# Benchmark Report (`benchmark_optimized_2s`)

- **Accelerator**: TPU v5e (1 chip)
- **Model**: `facebook/w2v-bert-2.0` (580,493,120 params, 24 layers)
- **Frameworks**: torch `2.14.1+cpu`, torch_tpu `0.3.0`, libtpu `0.0.50`, transformers `5.24.1`, jax `0.11.2`

| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) | p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_auto | ok | 2 | 100 | 1 | 7.3891 | 3.175 | 3.363 | 3.622 | 629.97 | 1.19 | 18.75 | 1 | 0 |
| optimized_auto | ok | 2 | 100 | 16 | 7.684 | 19.257 | 20.577 | 21.454 | 1661.76 | 1.646 | 49.47 | 1 | 0 |
