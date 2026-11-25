# Benchmark Report (`benchmark_bucketed_2s`)

- **Accelerator**: TPU v5e (1 chip)
- **Model**: `facebook/w2v-bert-2.0` (580,493,120 params, 24 layers)
- **Frameworks**: torch `2.14.1+cpu`, torch_tpu `0.3.0`, libtpu `0.0.50`, transformers `5.24.1`, jax `0.11.2`

| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) | p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_auto | ok | 2 | 100 | 1 | 19.8293 | 3.69 | 4.119 | 4.17 | 542.06 | 1.188 | 20.7 | 1 | 0 |
| optimized_auto | ok | 2 | 100 | 16 | 24.8888 | 23.39 | 24.59 | 24.849 | 1368.1 | 1.536 | 52.25 | 1 | 0 |
