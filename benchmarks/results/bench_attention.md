# Relative-key attention microbenchmark (TPU v5e, BF16, B=16, H=16, D=64, compiled)

p50 latency in ms (50 runs; `*_gather` at T>=1500: 3 runs). Lower is better; **bold** = fastest.

| Variant | T=250 | T=256 | T=500 | T=512 | T=1500 | T=1536 |
|---|---:|---:|---:|---:|---:|---:|
| hf_ref_compiled | **0.6829** | 0.7087 | 3.5098 | 3.3196 | 27.2217 | 26.9198 |
| explicit_gather | 0.7004 | 0.6195 | 2.3717 | 2.3815 | 8563.0134 | 8955.428 |
| explicit_index | 0.8922 | 0.8067 | 2.9954 | 2.6609 | 22.2136 | 19.1252 |
| sdpa_gather | 0.851 | 0.716 | **1.6015** | 1.6996 | 8555.6006 | 13860.63 |
| sdpa_index | 1.4323 | 0.9767 | 3.9722 | 2.8714 | 41.9216 | 18.3892 |
| pallas_128x128 | 1.5682 | 1.2009 | 4.131 | 3.3095 | 25.1684 | 22.8388 |
| pallas_256x256 | 0.9869 | **0.6176** | 2.406 | 1.7112 | 11.6031 | 9.5655 |
| pallas_256x512 | 1.7824 | 1.6573 | 1.9456 | 1.266 | 8.572 | 6.5442 |
| pallas_512x256 | 1.9755 | 1.8409 | 2.1525 | 1.4471 | 9.936 | 7.8857 |
| pallas_512x512 | 1.481 | 1.356 | 1.6688 | **1.028** | **8.0812** | **6.0074** |
| pallas_512x1024 | 3.5796 | 3.4389 | 4.0029 | 3.6442 | 11.0591 | 10.7559 |
| pallas_1024x512 | 3.7532 | 3.6468 | 4.2029 | 3.8317 | 11.6888 | 11.3954 |
| pallas_1024x1024 | 2.8429 | 2.7093 | 3.2606 | 2.9136 | 11.1921 | 10.8713 |

## Q/K/V projection (T=1500/1536)

| T | Variant | p50 (ms) |
|---:|---|---:|
| 1500 | qkv_separate | 2.1314 |
| 1500 | qkv_runtime_cat | 2.6804 |
| 1500 | qkv_prepacked | 2.6815 |
| 1536 | qkv_separate | 1.3174 |
| 1536 | qkv_runtime_cat | 2.6113 |
| 1536 | qkv_prepacked | 2.5888 |

Notes:
- `*_gather` variants hit an XLA TPU gather cliff at long T (8.5-13.9 s at T>=1500).
- Pallas rows were measured after fixing a stale AOTAutograd cache (kernel op names are now
  content-hashed); earlier Pallas timings were discarded.
- T=1500 is padded to the tile multiple and runs the masked kernel; T=1536 runs unmasked.
