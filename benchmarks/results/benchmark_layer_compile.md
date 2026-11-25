# Benchmark Report (`benchmark_layer_compile`)

- **Accelerator**: TPU v5e (1 chip)
- **Model**: `facebook/w2v-bert-2.0` (580,493,120 params, 24 layers)
- **Frameworks**: torch `2.14.1+cpu`, torch_tpu `0.3.0`, libtpu `0.0.50`, transformers `5.24.1`, jax `0.11.2`

| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) | p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_auto | ok | 5 | 250 | 1 | 8.2507 | 4.159 | 5.192 | 5.763 | 1202.25 | 1.215 | 36.24 | 1 | 0 |
| optimized_auto | ok | 5 | 250 | 16 | 8.5086 | 46.988 | 48.349 | 48.786 | 1702.56 | 1.678 | 51.32 | 1 | 0 |
| optimized_auto | ok | 30 | 1500 | 1 | 17.0332 | 23.657 | 23.786 | 23.792 | 1268.14 | 1.739 | 42.18 | 1 | 0 |
| optimized_auto | ok | 30 | 1500 | 16 | 9.1536 | 411.06 | 411.523 | 411.936 | 1167.71 | 3.707 | 38.84 | 1 | 0 |
