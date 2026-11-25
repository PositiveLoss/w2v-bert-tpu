# Performance

## Setup

- One TPU v5e chip: 197 TFLOPS BF16, 15.75 GiB usable HBM.
- `facebook/w2v-bert-2.0`, all 24 layers, BF16 weights and activations.
- Input: random `[B, T, 160]` features with an all-ones mask. T is the number of encoder
  frames: 50 frames per second of audio (5 s → T=250).
- Whole model compiled with `torch.compile(backend="tpu", dynamic=False, fullgraph=True)`.
- Latency: median (p50) wall time per synchronized call, 50 runs after 5 warmup runs.
- MFU (model FLOPs utilization): analytical FLOP count from `benchmarks/bench_model.py`
  divided by time and peak FLOPS.

Every number here is also in `benchmarks/results/benchmark_summary.md`.

## w2vbert_tpu vs compiled Hugging Face

p50 latency, HF → w2vbert_tpu (speedup):

| Audio | B=1 | B=4 | B=8 | B=16 |
|---|---|---|---|---|
| 2 s (T=100) | 3.20 → 3.18 ms (1.01×) | – | – | 19.74 → 19.24 ms (1.03×) |
| 5 s (T=250) | 5.10 → 4.04 ms (1.26×) | 13.01 → 12.01 ms (1.08×) | 31.61 → 20.40 ms (1.55×) | 53.55 → 38.30 ms (1.40×) |
| 10 s (T=500) | 9.11 → 7.02 ms (1.30×) | 32.93 → 20.89 ms (1.58×) | 60.99 → 39.95 ms (1.53×) | 117.58 → 78.79 ms (1.49×) |
| 30 s (T=1500) | 57.71 → 22.90 ms (2.52×) | 142.54 → 81.12 ms (1.76×) | 269.99 → 160.04 ms (1.69×) | 545.15 → 327.41 ms (1.67×) |

- Every shape compiled once, with no graph breaks.
- MFU, HF → w2vbert_tpu: 5 s B=16 45% → 63%; 10 s B=16 42% → 62%; 30 s B=1 17% → 44%;
  30 s B=16 29% → 49%.
- Peak HBM on long inputs: 30 s B=16 10.7 → 7.7 GiB; 30 s B=8 7.3 → 5.8 GiB. The packed Q/K/V
  weights add about 151 MB. Peak-HBM numbers for short clips include memory left over from
  earlier runs in the same process, so they can't be compared.
- Uncompiled Hugging Face is 4–11× slower than w2vbert_tpu at batch 1.

Where the speedup comes from:

1. **Attention.** Hugging Face builds a `[T, T, 64]` tensor and a `[B, H, T, T]` bias in every
   layer. This cost grows with the square of T, which is why HF's MFU falls from 34% at 10 s to
   17% at 30 s (batch 1). The Pallas kernel builds the bias in small tiles in on-chip memory.
2. **Packed Q/K/V read in place.** The kernel reads Q, K and V from one packed projection and
   writes output in the layout `linear_out` expects. This removed the transposes, copies and
   padding that took about 22% of device time at 30 s, batch 16. It is worth 1.2–1.4× over the
   earlier kernel at batch ≥ 4.

Short single clips (≤ 5 s, batch ≤ 4) use the plain-PyTorch attention path. There, most of the
time goes to the feed-forward and convolution blocks (unchanged Hugging Face code) and to
per-call overhead, so the gain is only 1.0–1.3×.

## Attention alone (one layer, batch 16)

| T | HF-style reference | Best native variant | Pallas `[B,H,T,D]`, 512×512 tiles |
|---:|---:|---:|---:|
| 512 | 3.32 ms | 1.67 ms | 1.03 ms |
| 1536 | 26.92 ms | 18.39 ms | 6.01 ms |

Including the Q/K/V projection and layout changes, the packed-Q/K/V kernel is faster again:

| Case | `[B,H,T,D]` kernel | Packed Q/K/V kernel |
|---|---:|---:|
| T=512, B=16 | 1.48 ms | 1.04 ms |
| T=1500, B=16 | 7.97 ms | 6.97 ms |
| T=1536, B=16 | 7.17 ms | 5.71 ms |
| T=3072, B=8 | 10.53 ms | 8.84 ms |

All variants and tile sizes: `benchmarks/results/bench_attention.md`.

## Long inputs (measured before the packed Q/K/V kernel)

p50 latency, with throughput in hours of audio processed per hour in parentheses. These numbers
are from the earlier kernel; current latencies are lower.

| Audio | B=1 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| 2 s | 3.18 ms (630) | 13.60 ms (1177) | 19.26 ms (1662) | 40.74 ms (1571) |
| 20 s | 14.67 ms (1364) | 116.29 ms (1376) | 239.11 ms (1338) | 491.64 ms (1302) |
| 60 s | 54.81 ms (1095) | 490.08 ms (979) | 980.00 ms (980) | 1945.28 ms (987) |
| 120 s | 149.67 ms (802) | 1312.37 ms (732) | 2573.23 ms (746) | skipped (not enough HBM) |

Every cell compiled once with no graph breaks. Memory stays low because nothing of size T×T
is stored: 120 s at batch 16 and 60 s at batch 32 both peak at 6.3 GiB. The best throughput
in this sweep was about 1,670 hours of audio per hour (5 s clips, batch 16).

## Bucketing (padding to fixed shapes)

`pad_to_bucket` pads T up to a fixed size. Sizes that are multiples of 128 let the kernel skip
its own padding step and avoid misaligned memory layouts.

| Audio | B=1 exact → bucketed | B=16 exact → bucketed |
|---|---|---|
| 2 s (T=100 → 128) | 3.18 → 3.69 ms | 19.24 → 23.39 ms |
| 5 s (250 → 256) | 4.04 → 4.09 ms | 38.30 → 40.23 ms |
| 10 s (500 → 512) | 7.02 → 7.22 ms | 78.79 → 80.95 ms |
| 30 s (1500 → 1536) | 22.90 → 23.55 ms | 327.41 → 309.86 ms |
| 60 s (3000 → 3072) | 55.06* → 55.42 ms | 980.56* → 768.64 ms |

\* Exact-length 60 s numbers were measured before the packed Q/K/V kernel.

Advice:

- Bucket long batched inputs. At 30 s, batch 16, it is 5% faster.
- Use buckets to limit how many shapes get compiled.
- For 5–10 s clips, padding is now 3–5% slower at batch 16. Bucket for the compile savings,
  not for speed.
- For very short clips (2 s), padding 100 → 128 frames costs 16–21%. Use finer buckets or exact
  shapes.

Run-to-run noise between processes is about ±3%. The first configuration measured in a new
process can be slow: one sweep reported 5.81 ms for 2 s, batch 1, while a controlled re-run
gave 3.18 ms.

## Compile time

- First call of a new shape (compile + run): 8–30 s for the whole model. TorchTPU's on-disk
  caches make later processes much faster.
- `compile_strategy="layer"` compiles one layer and reuses it for all 24: 8.3 s instead of
  20.0 s for the first call at 5 s, batch 1. Latency is within ±6% of the whole-model compile.
- `prewarm()` compiles a list of buckets ahead of time.

## Profile (30 s, batch 16)

From `benchmarks/results/profile_summary.md` (`torch.profiler` with `ProfilerActivity.TPU`),
taken before the packed Q/K/V kernel: 540 ms per step for HF, 407 ms for w2vbert_tpu. The Pallas
kernel took 27.7% of device time (113 ms per step, 4.7 ms per layer). Layout operations around
it (`copy`, `pad`, `reshape`, `as_strided`) took about 22%. The packed Q/K/V kernel removed most
of those (407 → 327 ms end to end). Weight prefetch copies run in parallel with compute.

## Limitations

- Gains on short single clips are small. Most of the remaining time is in feed-forward and
  convolution code that XLA already compiles well, plus per-call overhead.
- The Pallas depthwise convolution matches XLA's depthwise convolution, but the channels-last
  module around it is 12–20% slower than Hugging Face's, so it is off by default.
- One chip only. There is no sharding across chips.
- All numbers come from one machine and one software stack (see `environment.md`), with one
  measurement run per configuration (50 calls each; 10 for the long-input sweep).
