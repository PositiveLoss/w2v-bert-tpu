# Benchmark Report (`baseline_hf_eager`)

- **Accelerator**: TPU v5e (1 chip)
- **Model**: `facebook/w2v-bert-2.0` (580,493,120 params, 24 layers)
- **Frameworks**: torch `2.14.1+cpu`, torch_tpu `0.3.0`, libtpu `0.0.50`, transformers `5.24.1`, jax `0.11.2`

| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) | p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hf_eager | ok | 5 | 250 | 1 | 0.9853 | 45.927 | 48.266 | 48.752 | 108.87 | 1.128 | 3.28 | 0 | 0 |
| hf_eager | ok | 5 | 250 | 4 | 1.3089 | 38.143 | 41.176 | 42.123 | 524.34 | 1.231 | 15.81 | 0 | 0 |
| hf_eager | ok | 10 | 500 | 1 | 0.8783 | 38.606 | 48.537 | 49.109 | 259.02 | 1.213 | 7.97 | 0 | 0 |
| hf_eager | ok | 10 | 500 | 4 | 1.1344 | 72.228 | 72.348 | 72.385 | 553.8 | 1.465 | 17.04 | 0 | 0 |
| hf_eager | ok | 30 | 1500 | 1 | 1.9609 | 92.602 | 92.851 | 92.951 | 323.97 | 1.808 | 10.78 | 0 | 0 |
| hf_eager | ok | 30 | 1500 | 4 | 2.2613 | 281.955 | 284.771 | 286.574 | 425.6 | 2.965 | 14.16 | 0 | 0 |
