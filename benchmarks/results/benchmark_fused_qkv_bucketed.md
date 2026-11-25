# Benchmark Report (`benchmark_fused_qkv_bucketed`)

- **Accelerator**: TPU v5e (1 chip)
- **Model**: `facebook/w2v-bert-2.0` (580,493,120 params, 24 layers)
- **Frameworks**: torch `2.14.1+cpu`, torch_tpu `0.3.0`, libtpu `0.0.50`, transformers `5.24.1`, jax `0.11.2`

| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) | p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_auto | ok | 5 | 250 | 1 | 14.8741 | 4.087 | 4.334 | 4.562 | 1223.52 | 1.359 | 37.78 | 1 | 0 |
| optimized_auto | ok | 5 | 250 | 16 | 22.7298 | 40.231 | 41.136 | 41.277 | 1988.49 | 1.756 | 61.41 | 1 | 0 |
| optimized_auto | ok | 10 | 500 | 1 | 19.4858 | 7.216 | 7.762 | 7.913 | 1385.75 | 1.729 | 43.7 | 1 | 0 |
| optimized_auto | ok | 10 | 500 | 16 | 21.7007 | 80.953 | 82.483 | 82.487 | 1976.47 | 2.41 | 62.33 | 1 | 0 |
| optimized_auto | ok | 30 | 1500 | 1 | 20.1625 | 23.554 | 25.199 | 25.218 | 1273.65 | 2.265 | 43.5 | 1 | 0 |
| optimized_auto | ok | 30 | 1500 | 16 | 36.7966 | 309.86 | 310.209 | 310.475 | 1549.09 | 3.746 | 52.9 | 1 | 0 |
| optimized_auto | ok | 60 | 3000 | 1 | 69.4941 | 55.422 | 59.715 | 59.999 | 1082.59 | 2.921 | 41.22 | 1 | 0 |
| optimized_auto | ok | 60 | 3000 | 16 | 24.7048 | 768.635 | 768.881 | 768.976 | 1248.97 | 5.778 | 47.56 | 1 | 0 |
