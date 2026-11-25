# Benchmark Report (`benchmark_unbucketed_60s`)

- **Accelerator**: TPU v5e (1 chip)
- **Model**: `facebook/w2v-bert-2.0` (580,493,120 params, 24 layers)
- **Frameworks**: torch `2.14.1+cpu`, torch_tpu `0.3.0`, libtpu `0.0.50`, transformers `5.24.1`, jax `0.11.2`

| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) | p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_auto | ok | 60 | 3000 | 1 | 26.8044 | 55.056 | 56.734 | 56.94 | 1089.8 | 1.493 | 40.33 | 1 | 0 |
| optimized_auto | ok | 60 | 3000 | 16 | 29.2949 | 980.56 | 980.771 | 980.809 | 979.03 | 5.229 | 36.23 | 1 | 0 |
