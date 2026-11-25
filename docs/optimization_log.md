# Optimization log

Each change that was tried, the TPU v5e measurements behind it, and whether it was kept.
Unless noted, latencies are p50 for the full 24-layer BF16 model (or the named
microbenchmark), compiled with `torch.compile(backend="tpu", fullgraph=True)`.

## Baselines

Earlier compiled-HF measurements we were given at the start:

| Audio | B=1 | B=4 | B=8 | B=16 |
|---|---:|---:|---:|---:|
| 5 s (T=250) | 4.578 ms | 12.336 ms | 28.114 ms | 48.834 ms |
| 10 s (T=500) | 7.517 ms | 26.870 ms | 61.857 ms | 135.803 ms |
| 30 s (T=1500) | 51.034 ms | 179.288 ms | 360.660 ms | 727.246 ms |

Measured again on this machine (ms):

| | B=1 | B=4 | B=8 | B=16 |
|---|---:|---:|---:|---:|
| HF compiled, 5 s | 5.10 (29.6% MFU) | 13.01 | 31.61 | 53.55 |
| HF compiled, 10 s | 9.11 (33.8% MFU) | 32.93 | 60.99 | 117.58 |
| HF compiled, 30 s | 57.71 (17.3% MFU) | 142.54 | 269.99 | 545.15 (10.7 GiB) |
| HF eager, 5 / 10 / 30 s | 45.9 / 38.6 / 92.6 | 38.1 / 72.2 / 282.0 | – | – |

HF builds a `[T, T, 64]` relative-position tensor and a `[B, H, T, T]` bias in every layer.
That is why its MFU drops at 30 s.

## 1. Plain-PyTorch relative attention: kept, used by `auto`

**Idea:** compute `q @ Eᵀ` (`[B, H, T, 73]`) and gather into `[B, H, T, T]`, skipping the
`[T, T, 64]` tensor.

**Result** (`bench_attention.md`, batch 16): the best variant beats the HF-style reference at
every length (T=512: 1.67 vs 3.32 ms; T=1536: 18.4 vs 26.9 ms). But XLA's gather on TPU becomes
extremely slow on long inputs: `explicit_gather` took 8,563 ms and `sdpa_gather` 8,556–13,861 ms
at T ≥ 1500, against 22 ms for `index_select`.

**Decision:** `native_variant="auto"` uses gather up to T=512 and `index_select` above. Even so,
the native path alone is slow at 30 s (154 ms at batch 1), so long inputs use Pallas.

## 2. One packed Q/K/V matmul: not the default on its own

**Idea:** one `[3d, d]` matmul instead of three `[d, d]` matmuls.

**Result** (batch 16): T=1500 2.13 ms separate vs 2.68 ms packed; T=1536 1.32 vs 2.59 ms.

**Decision:** `prepack_qkv=False` by default. Packing does pay off once the kernel reads the
packed result directly (see 9).

## 3. Channels-last Conformer convolution: off by default

**Idea:** run the convolution in `[B, T, C]` layout with `F.linear` pointwise convs to avoid
transposes.

**Result** (`bench_convolution.md`, batch 16, T=250/500/1500): HF 0.65/0.92/2.02 ms.
`conv1d` was 1.06–1.21× slower, `shift` 1.44–1.99× slower. `unfold` ran out of HBM at compile
time at T=250 (31 GiB of temporaries) and was 4–6× slower elsewhere.

**Decision:** `convolution=False` by default. `TPUConvolutionModule` stays as an option.

## 4. Fused Pallas attention kernel: kept, after fixing four problems

1. **It never ran under `torch.compile`.** The `jax_op` registration happened lazily during
   Dynamo tracing and failed in `inspect.signature`. Fix: register eagerly in `optimize()`
   (`register_pallas_ops()`).
2. **Out of VMEM for any tile larger than 128** (29 MiB needed, 16 MiB allowed). Testing kernel
   variants directly in JAX on the TPU showed the cause was a value-returning `lax.cond` over
   the whole bias tile, not the gather as first assumed. Fix: write the bias into VMEM scratch
   under `pl.when`, replace the band `take_along_axis` with two lane rotations, and raise
   `vmem_limit_bytes` to 64 MiB.
3. **3–8× slower than XLA in the first benchmark.** JAX's reference Pallas flash attention (no
   bias) ran in 3.6 ms vs 5.1 ms for XLA at T=1536, so a fast kernel was possible; ours took
   41.8 ms. Even a minimal flash-attention core written our way took 39 ms. The cause was
   `jnp.tile(m[:, :1], (1, block_k))`: broadcasting a single lane is very slow in Mosaic. Fix:
   keep per-row statistics copied across all 128 lanes, and compute the far-region bias columns
   that way too, using the MXU. Result: 41.8 → 5.8 ms (measured in JAX directly).
4. **Benchmarks through PyTorch didn't change after the fix.** PyTorch's AOTAutograd cache
   (`/tmp/torchinductor_*`) identifies compiled programs by FX graph, which includes the custom
   op's name but not its code, so the old kernel was reused. Fix: op names include a hash of the
   kernel source and the jax/jaxlib/libtpu versions. All Pallas numbers were re-measured with
   clean caches.

**Result** (batch 16, clean caches): Pallas with 512×512 tiles beats the best native variant from
T=512 (1.03 vs 1.67 ms) and is 4.5× faster than the HF-style reference at T=1536 (6.0 vs
26.9 ms). Accuracy: cos ≥ 0.99997 for 5 tile shapes × 3 lengths.

**Decision:** kept.

## 5. Tile sizes and the `auto` threshold: kept

Tiles: 128×128 for T ≤ 128, 256×256 for T ≤ 256, 512×512 otherwise. 1024-wide tiles were slower
(more padding, fewer parallel grid steps).

Full model at T=250: native wins at batch ≤ 4 (3.93 vs 4.91 ms at batch 1). Pallas wins at
batch ≥ 8 (27.46 vs 28.43 ms at batch 8) and at every batch size from T=500.

**Decision:** `auto` uses Pallas when T > 256, or when T > 128 and batch ≥ 8.

## 6. Bugs found while benchmarking

- `fullgraph=True` compilation failed on the first trace. The packed-weight cache check compared
  `Tensor._version`, which becomes an unbacked symbol under Dynamo. Fix: check the cache only in
  eager mode. A regression test compiles at three shapes.
- `output_attentions=True` returned no weights under transformers 5.x's default `"sdpa"`. Fix:
  `optimize()` selects HF's `"eager"` attention for the fallback path.

## 7. Static bucketing: recommended, not forced

Padding T to a multiple of 128 avoids the kernel's own padding and misaligned layout operations.
Batch 16, bucketed vs exact: 5 s 44.3 vs 47.9 ms, 10 s 92.9 vs 97.8 ms, 30 s 370.5 vs
411.3 ms, 60 s 905 vs 981 ms. At batch 1 it costs 0–6% because of the extra frames.

These numbers are from before change 9. With the packed Q/K/V kernel, bucketing only helps at
30 s and above (see `performance.md`).

**Decision:** offered through `pad_to_bucket`, not done inside the model, so exact shapes still
give exact-shape results.

## 8. Whole-model vs per-layer compile: whole-model stays the default

Per-layer compile is 1.2–2.4× faster to compile at 5 s and at 30 s batch 1 (8.3 vs 20.0 s at 5 s
batch 1), about the same at 30 s batch 16. Latency is within ±6%.

**Decision:** whole-model by default; `compile_strategy="layer"` is available.

## 9. Kernel reads packed Q/K/V directly (w2v-bert-fast layout): kept, now the default

**Idea:** about 22% of device time went to head transposes, `.contiguous()` copies and padding
around the `[B, H, T, D]` kernel. If the kernel reads the packed `[B, T, 3·H·D]` projection
directly (two 64-wide heads per 128-lane block) and writes `[B, T, H·D]`, that work disappears.
Change 2 rejected packing because the packed matmul alone is slower; the gain in w2v-bert-fast
comes from skipping the layout changes.

**Result:** attention block including projection, batch 16: T=512 1.48 → 1.04 ms, T=1536
7.17 → 5.71 ms. Full model vs the previous default: 1.23–1.43× faster at batch ≥ 4 (30 s batch 16:
411 → 327 ms; 5 s batch 8: 27.9 → 20.4 ms).

**Accuracy:** same error as the previous path at T=512 (4 seeds). At T=320 the two compiled
programs differ only by BF16 rounding (error ratio 0.93–1.61 across seeds, no consistent
direction).

**Decision:** default on the Pallas path (`fused_qkv=True`). The padding and mixed-length TPU
tests were changed to compare against FP32 (and HF BF16). They previously compared BF16 against
BF16 with a 1e-2 bound, which is tighter than BF16 rounding noise.

**Measurement mistake:** one run reported 10.2 ms for 5 s, batch 1, because the CPU test suite
was running at the same time (short clips are limited by host-side dispatch). Measured again on
a quiet machine: 4.04 ms. Benchmarks are now run with nothing else on the host.

## 10. Pallas depthwise convolution (w2v-bert-fast port): off by default

The kernel works on `[B, T, C]` tiles plus 32 preceding rows, and does 31 shifted FP32
multiply-adds. Module benchmark (batch 16, T=250/500/1500): 0.70/0.99/2.41 ms vs HF
0.60/0.88/2.00 ms. The same module with XLA's depthwise conv takes 0.69/1.03/2.44 ms, so the
kernel itself is fine. The loss comes from the channels-last LayerNorm and pointwise layers
around it.

**Decision:** available as `conv_impl="pallas"`, off by default.

## Profile (30 s, batch 16; see `profile_summary.md`)

Measured before change 9: HF 540 ms per step, optimized 407 ms. The Pallas kernel was 27.7% of
device time; layout operations around attention (`copy`, `pad`, `reshape`, `as_strided`) were
about 22%. Change 9 removed most of them.

## Left out on purpose

- Padding to a multiple of 128 inside the model when batch ≥ 8. This is left to the caller
  (`pad_to_bucket`) so that exact-shape calls stay exact.
