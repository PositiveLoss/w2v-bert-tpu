# Architecture

`w2vbert_tpu` runs Hugging Face `facebook/w2v-bert-2.0` (`Wav2Vec2BertModel`, 24 Conformer
layers, 580,493,120 parameters) on one TPU v5e chip in BF16. It also runs models built on that
encoder, such as the fine-tuned speech recognizer `Wav2Vec2BertForCTC`.

It changes the model in place. `optimize(model)` replaces the class of some Hugging Face
submodules with a subclass that has a faster `forward`. Parameters are not touched, so parameter
names, `state_dict()` and `isinstance` checks all behave as before.

```
waveform ──► features.extract_features     (on the host; same output as SeamlessM4TFeatureExtractor)
         ──► batching.pad_to_bucket         (optional; pad to a fixed shape)
         ──► Wav2Vec2BertModel              (torch.compile(backend="tpu", fullgraph=True))
               feature_projection           (Hugging Face, unchanged)
               TPUEncoder                   (computes masks and position indices once)
                 24 × encoder layer         (Hugging Face, unchanged except:)
                     self_attn → TPUSelfAttention
                     conv_module            (Hugging Face; TPUConvolutionModule if convolution=True)
               adapter                      (CTC models only; Hugging Face, unchanged)
             lm_head                        (CTC models only; Hugging Face, unchanged)
         ──► batching.unpad_output          (optional; remove padding)
             or ctc.ctc_greedy_decode       (CTC models: logits → token ids)
```

`optimize()` looks for the `Wav2Vec2BertEncoder` anywhere inside the model it is given, so it
works the same on `Wav2Vec2BertModel`, `Wav2Vec2BertForCTC` (encoder at
`model.wav2vec2_bert.encoder`) or a bare encoder. It only replaces attention whose
`position_embeddings_type` is `relative_key`. The adapter's attention has no position embeddings,
so it stays as Hugging Face code.

`load()` reads `architectures` from the checkpoint's `config.json` and builds that class
(`Wav2Vec2BertForCTC` for a fine-tuned recognizer), falling back to `Wav2Vec2BertModel`.

Source files (`src/w2vbert_tpu/`):

| File | Contents |
|---|---|
| `model.py` | `load`, `optimize`, `TPUSelfAttention`, `TPUEncoder` |
| `ctc.py` | `ctc_output_lengths`, `ctc_greedy_decode` for fine-tuned CTC models |
| `attention.py` | Plain-PyTorch relative-key attention and the reference implementation |
| `kernels/relative_attention.py` | Pallas kernel with `[B, H, T, D]` inputs, tile selection |
| `kernels/fused_qkv_attention.py` | Pallas kernel that reads packed `[B, T, 3·H·D]` Q/K/V (default) |
| `kernels/causal_conv.py` | Pallas channels-last depthwise convolution (opt-in) |
| `convolution.py` | `TPUConvolutionModule` (opt-in) |
| `features.py` | `load_audio`, `extract_features` |
| `batching.py` | `pad_to_bucket`, `unpad_output`, `prewarm` |
| `compiler.py` | `configure_compiler`, compile caches, compile counters |
| `_tpu.py` | `torch_tpu()`: imports TorchTPU and returns `torch.tpu` |

## Relative-key attention

w2v-BERT 2.0 uses clipped relative-position embeddings. With `L = 64` positions to the left,
`R = 8` to the right and a table `E` of shape `[73, 64]`:

    score(i, j) = (q_i·k_j + q_i·E[clip(j − i, −L, R) + L]) / √D

Hugging Face looks up `E` for every `(i, j)` pair, building a `[T, T, 64]` tensor and then a
`[B, H, T, T]` bias in every layer. `TPUSelfAttention` never builds the `[T, T, 64]` tensor.
It has two implementations:

| Backend | How the position term is computed |
|---|---|
| `native` | `q @ Eᵀ` gives `[B, H, T, 73]`; those scores are gathered into the `[B, H, T, T]` logits |
| `pallas` | One fused kernel builds the bias per tile in on-chip memory (VMEM); it never goes to HBM |

`backend="auto"` picks Pallas when the input is on the TPU in BF16/FP32 and either T > 256, or
T > 128 with batch ≥ 8. Otherwise it uses `native`. The thresholds come from full-model
benchmarks (`benchmarks/results/benchmark_optimized.md`).

Inside `native`, `native_variant="auto"` picks:

- T ≤ 256: explicit matmul + softmax with a gather;
- 256 < T ≤ 512: SDPA with a gather;
- T > 512: explicit matmul with `index_select`.

XLA's gather on TPU becomes extremely slow on long inputs (8.5 s instead of 22 ms at T=1500,
batch 16), so the gather variants are never chosen above 512 frames.

### Pallas kernel (`kernels/relative_attention.py`)

The grid is `(B, H, T/block_q, T/block_k)`. The last axis walks over key tiles and accumulates
an online softmax.

1. On the first key tile of each query tile, the kernel computes `q @ Eᵀ` (`[block_q, 128]`, with
   `E` zero-padded to 128 rows). It also computes `q·E[0]` and `q·E[L+R]`, the only two values
   used far from the diagonal, and stores them in VMEM.
2. Each key tile falls into one of three cases, known from the tile position alone:
   - **far left** (every `j − i ≤ −L`) or **far right** (every `j − i ≥ R`): the bias is one
     value per row, copied across the tile;
   - **band** (near the diagonal): the bias is built with the Transformer-XL "relative shift"
     trick, using two lane rotations of the `q @ Eᵀ` scores.
3. The bias tile is written to VMEM, added to `q @ kᵀ`, masked by sequence length if needed, and
   folded into the FP32 online softmax.

Rules learned on hardware (details in `optimization_log.md`):

- Do not broadcast a single-column `[rows, 1]` value across lanes. Keep per-row statistics
  copied across all 128 lanes instead. The single-column version was 7× slower.
- Do not choose between two large tiles with a value-returning `lax.cond`. Mosaic reserves about
  29 MiB of VMEM for it and the kernel fails to compile. Write into scratch under `pl.when`.
- Set `vmem_limit_bytes` to 64 MiB. The 16 MiB default rejects 512-wide tiles, and v5e has
  128 MiB of VMEM.

Tile sizes: 128×128 for T ≤ 128, 256×256 for T ≤ 256, 512×512 above that. `pallas_tile_size`
overrides this. When T is not a multiple of the tile size, the input is padded and the kernel
masks the padded keys using each sequence's real length.

### Packed Q/K/V (`kernels/fused_qkv_attention.py`, default)

Following w2v-bert-fast, Q, K and V come from one `[3d, d]` matmul. The packed weight is cached
on the module, outside `state_dict`. The kernel reads the `[B, T, 3·H·64]` result directly.

With 64-dim heads, one 128-lane block holds two heads, so the grid is `(B, H/2, T/bq, T/bk)`.
The Q, K and V tiles for head pair `hp` are lane blocks `hp`, `H/2 + hp` and `H + hp` of the same
array. For each head, the kernel zeroes the other head's lanes before `q @ kᵀ` (exact), places
`E` in that head's lanes for `q @ Eᵀ`, and merges both heads' results with lane selects. The
output is `[B, T, H·64]`, which goes straight into `linear_out` with no transposes.

This path needs head size 64 and an even number of heads. Other shapes, or `fused_qkv=False`,
use the `[B, H, T, D]` kernel.

### Registering and caching the kernels

- `optimize()` registers the kernels with `torch.tpu.pallas.jax_op` before compiling. Dynamo
  cannot trace the registration (it calls `inspect.signature`), so it can't happen lazily.
- Each op name includes a hash of the kernel source files and the jax/jaxlib/libtpu versions.
  PyTorch's AOTAutograd cache identifies compiled programs by their FX graph, which contains the
  op name but not the kernel code. Without the hash, an edited kernel could silently reuse an
  old compiled version from `/tmp/torchinductor_*`.
- TorchTPU processes run JAX with `JAX_PLATFORMS=cpu`. JAX only traces and exports the kernels
  on the host; TorchTPU owns the TPU device.

## Encoder and masks

`TPUEncoder` replaces only the encoder's `forward`. Once per forward pass it computes the
sequence lengths, the additive key mask and, for the native backend, the position indices. It
passes these to every layer, so layers don't recompute them.

Nothing in the forward pass copies data back to the host (`.item()`, `.tolist()`,
`nonzero()`). Lengths stay on the device and all branching depends only on tensor shapes.

`output_attentions=True`, and any config that doesn't use `relative_key` positions, fall back
to the Hugging Face encoder and eager attention. `optimize()` sets the config's attention
implementation to `"eager"` because transformers 5.x silently returns no attention weights with
its default `"sdpa"`.

`optimize()` also sets `layerdrop`, `mask_time_prob` and `mask_feature_prob` to 0.

## Compile safety

- The packed Q/K/V weight and the padded `E` table are cached. The cache is checked for
  staleness only in eager mode. Under Dynamo it is read directly: checking
  `Tensor._version` while tracing created unbacked symbols and broke `fullgraph=True`.
- `configure_compiler()` raises Dynamo's recompile limit to 128 (1024 total), since each bucket
  shape is one compile. When the limit is hit it raises an error instead of silently falling
  back to eager. It also enables TorchTPU's tier-2 cache
  (`/dev/shm/torch_tpu_cache/w2vbert_tpu_tier2`) and tier-3 cache
  (`/tmp/w2vbert_tpu_tier3_cache`).

## Convolution

`TPUConvolutionModule` runs the Conformer convolution in `[B, T, C]` layout. The depthwise step
can use `conv_impl="conv1d"` (default), `"unfold"`, `"shift"` or `"pallas"`. The Pallas version
(`kernels/causal_conv.py`) is a port of w2v-bert-fast's Triton kernel: 256-row time tiles plus a
32-row block of preceding rows ("halo").

It is off by default. The compiled Hugging Face module was faster on v5e with every variant.
Pallas was 1.12–1.20× slower (`benchmarks/results/bench_convolution.md`).

## Bucketing

`batching.py` pads inputs to fixed shapes so a server compiles a known, small set of programs:

- frame buckets: 128, 256, 512, 768, 1024, 1536, 2048, 3072, 6144;
- batch buckets (optional, `batch_buckets=`): 1, 2, 4, 8, 16, 32.

Padded frames are masked out of attention. `unpad_output` removes padded rows and frames.
`prewarm(model)` compiles frames 128/256/512/1024/1536 × batch 1/4/16 by default and returns
the time each compile took.
