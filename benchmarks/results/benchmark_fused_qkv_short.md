# Benchmark Report (`benchmark_fused_qkv_short`)

- **Accelerator**: TPU v5e (1 chip)
- **Model**: `facebook/w2v-bert-2.0` (580,493,120 params, 24 layers)
- **Frameworks**: torch `2.14.1+cpu`, torch_tpu `0.3.0`, libtpu `0.0.50`, transformers `5.24.1`, jax `0.11.2`

| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) | p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_auto | ok | 2 | 100 | 1 | 12.2656 | 3.181 | 3.387 | 3.684 | 628.79 | 1.333 | 18.72 | 1 | 0 |
| optimized_auto | ok | 2 | 100 | 4 | 10.3722 | 5.903 | 6.223 | 7.91 | 1355.24 | 1.526 | 40.34 | 1 | 0 |
| optimized_auto | ok | 2 | 100 | 8 | 11.7391 | 10.631 | 11.071 | 12.08 | 1505.07 | 1.788 | 44.8 | 1 | 0 |
| optimized_auto | ok | 2 | 100 | 16 | 12.2423 | 19.241 | 20.425 | 21.363 | 1663.08 | 2.166 | 49.51 | 1 | 0 |
| optimized_auto | ok | 5 | 250 | 1 | 9.1542 | 4.043 | 4.226 | 5.241 | 1236.74 | 2.104 | 37.28 | 1 | 0 |
| optimized_auto | ok | 5 | 250 | 4 | 12.4436 | 12.014 | 12.517 | 13.608 | 1664.72 | 2.39 | 50.18 | 1 | 0 |
| optimized_auto | ok | 5 | 250 | 8 | 11.3805 | 20.401 | 21.29 | 22.377 | 1960.66 | 2.705 | 59.1 | 1 | 0 |
| optimized_auto | ok | 5 | 250 | 16 | 12.0282 | 38.304 | 39.953 | 40.41 | 2088.53 | 3.156 | 62.96 | 1 | 0 |
