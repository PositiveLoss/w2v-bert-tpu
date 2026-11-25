# Benchmark Report (`benchmark_extended_part2`)

- **Accelerator**: TPU v5e (1 chip)
- **Model**: `facebook/w2v-bert-2.0` (580,493,120 params, 24 layers)
- **Frameworks**: torch `2.14.1+cpu`, torch_tpu `0.3.0`, libtpu `0.0.50`, transformers `5.24.1`, jax `0.11.2`

| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) | p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_auto | ok | 20 | 1000 | 1 | 21.7784 | 14.667 | 15.656 | 15.761 | 1363.57 | 1.298 | 43.65 | 1 | 0 |
| optimized_auto | ok | 20 | 1000 | 8 | 23.9298 | 116.29 | 116.856 | 116.934 | 1375.88 | 1.876 | 44.05 | 1 | 0 |
| optimized_auto | ok | 20 | 1000 | 16 | 26.4792 | 239.114 | 239.259 | 239.282 | 1338.28 | 2.605 | 42.84 | 1 | 0 |
| optimized_auto | ok | 20 | 1000 | 32 | 24.8605 | 491.638 | 491.989 | 492.022 | 1301.77 | 3.78 | 41.68 | 1 | 0 |
| optimized_auto | ok | 60 | 3000 | 1 | 8.0099 | 54.812 | 55.24 | 55.299 | 1094.65 | 3.303 | 40.51 | 1 | 0 |
| optimized_auto | ok | 60 | 3000 | 8 | 38.8089 | 490.076 | 490.259 | 490.279 | 979.44 | 4.156 | 36.24 | 1 | 0 |
| optimized_auto | ok | 60 | 3000 | 16 | 9.1227 | 979.998 | 980.211 | 980.259 | 979.59 | 5.469 | 36.25 | 1 | 0 |
| optimized_auto | skipped | 60 | 3000 | 32 | - | - | - | - | - | - | - | - | - |
| optimized_auto | ok | 120 | 6000 | 1 | 25.012 | 149.673 | 150.535 | 150.567 | 801.75 | 4.681 | 35.67 | 1 | 0 |
| optimized_auto | ok | 120 | 6000 | 8 | 30.6664 | 1312.374 | 1312.522 | 1312.537 | 731.5 | 5.992 | 32.54 | 1 | 0 |
| optimized_auto | skipped | 120 | 6000 | 16 | - | - | - | - | - | - | - | - | - |
| optimized_auto | skipped | 120 | 6000 | 32 | - | - | - | - | - | - | - | - | - |
