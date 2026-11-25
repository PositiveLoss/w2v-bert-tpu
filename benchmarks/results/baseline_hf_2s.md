# Benchmark Report (`baseline_hf_2s`)

- **Accelerator**: TPU v5e (1 chip)
- **Model**: `facebook/w2v-bert-2.0` (580,493,120 params, 24 layers)
- **Frameworks**: torch `2.14.1+cpu`, torch_tpu `0.3.0`, libtpu `0.0.50`, transformers `5.24.1`, jax `0.11.2`

| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) | p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hf_compiled | ok | 2 | 100 | 1 | 12.4945 | 3.204 | 4.517 | 5.211 | 624.15 | 1.173 | 18.58 | 1 | 0 |
| hf_compiled | ok | 2 | 100 | 16 | 13.162 | 19.737 | 20.769 | 22.171 | 1621.3 | 1.639 | 48.26 | 1 | 0 |
