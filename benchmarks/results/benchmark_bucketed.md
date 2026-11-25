# Benchmark Report (`benchmark_bucketed`)

- **Accelerator**: TPU v5e (1 chip)
- **Model**: `facebook/w2v-bert-2.0` (580,493,120 params, 24 layers)
- **Frameworks**: torch `2.14.1+cpu`, torch_tpu `0.3.0`, libtpu `0.0.50`, transformers `5.24.1`, jax `0.11.2`

| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) | p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimized_auto | ok | 5 | 250 | 1 | 21.6042 | 4.07 | 4.14 | 4.143 | 1228.62 | 1.217 | 37.94 | 1 | 0 |
| optimized_auto | ok | 5 | 250 | 16 | 24.6846 | 44.346 | 46.07 | 46.553 | 1803.98 | 1.643 | 55.71 | 1 | 0 |
| optimized_auto | ok | 10 | 500 | 1 | 22.7592 | 7.536 | 9.532 | 10.082 | 1326.89 | 1.624 | 41.84 | 1 | 0 |
| optimized_auto | ok | 10 | 500 | 16 | 24.1036 | 92.898 | 93.741 | 93.872 | 1722.32 | 2.327 | 54.32 | 1 | 0 |
| optimized_auto | ok | 30 | 1500 | 1 | 24.8104 | 25.457 | 26.788 | 26.855 | 1178.44 | 2.189 | 40.25 | 1 | 0 |
| optimized_auto | ok | 30 | 1500 | 16 | 30.2788 | 370.492 | 370.837 | 370.851 | 1295.57 | 3.91 | 44.25 | 1 | 0 |
| optimized_auto | ok | 60 | 3000 | 1 | 24.4759 | 56.129 | 56.975 | 57.017 | 1068.96 | 3.095 | 40.7 | 1 | 0 |
| optimized_auto | ok | 60 | 3000 | 16 | 27.3571 | 905.407 | 905.731 | 905.887 | 1060.3 | 6.109 | 40.37 | 1 | 0 |
