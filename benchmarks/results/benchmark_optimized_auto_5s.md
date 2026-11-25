# Benchmark Report (`benchmark_optimized_auto_5s`)

- **Accelerator**: TPU v5e (1 chip)
- **Model**: `facebook/w2v-bert-2.0` (580,493,120 params, 24 layers)
- **Frameworks**: torch `2.14.1+cpu`, torch_tpu `0.3.0`, libtpu `0.0.50`, transformers `5.24.1`, jax `0.11.2`

| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) | p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_auto | ok | 5 | 250 | 8 | 18.2826 | 27.926 | 28.862 | 29.286 | 1432.36 | 1.626 | 43.18 | 1 | 0 |
| optimized_auto | ok | 5 | 250 | 16 | 14.9607 | 47.94 | 49.179 | 49.522 | 1668.77 | 2.157 | 50.3 | 1 | 0 |
