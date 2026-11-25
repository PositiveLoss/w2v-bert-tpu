# Benchmark Report (`benchmark_extended_part3`)

- **Accelerator**: TPU v5e (1 chip)
- **Model**: `facebook/w2v-bert-2.0` (580,493,120 params, 24 layers)
- **Frameworks**: torch `2.14.1+cpu`, torch_tpu `0.3.0`, libtpu `0.0.50`, transformers `5.24.1`, jax `0.11.2`

| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) | p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_auto | ok | 2 | 100 | 32 | 8.2132 | 40.742 | 42.188 | 42.222 | 1570.88 | 1.518 | 46.76 | 1 | 0 |
| optimized_auto | ok | 5 | 250 | 32 | 25.9344 | 97.706 | 98.196 | 98.271 | 1637.56 | 2.024 | 49.36 | 1 | 0 |
| optimized_auto | ok | 10 | 500 | 32 | 26.5028 | 207.518 | 207.847 | 207.89 | 1542.04 | 2.754 | 47.44 | 1 | 0 |
| optimized_auto | ok | 30 | 1500 | 32 | 32.9539 | 857.33 | 857.775 | 857.999 | 1119.76 | 4.771 | 37.25 | 1 | 0 |
