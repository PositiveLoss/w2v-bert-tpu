# TPU v5e profile summary (`torch.profiler`, `ProfilerActivity.TPU`)

Per forward step, from `key_averages()`. *Program* is the compiled executable's device
span. Synchronous HLO ops are grouped by instruction prefix (`convolution` = MXU
matmuls, `custom-call` = Pallas kernel, `fusion` = XLA elementwise/reduction fusions).
Async `*-start`/`*-done` copies (weight prefetch) overlap compute and are listed apart.

## hf_compiled: 5 s (T=250), B=1

Program: **3.385 ms/step**; synchronous ops 3.323 ms (1399 distinct).

| HLO group | ms/step | % of sync ops |
|---|---:|---:|
| `fusion` | 1.425 | 42.87 |
| `bitcast_add_fusion` | 1.231 | 37.04 |
| `copy` | 0.182 | 5.47 |
| `compare_reduce_fusion` | 0.177 | 5.32 |
| `select_convert_fusion` | 0.138 | 4.16 |
| `broadcast_multiply_fusion` | 0.083 | 2.49 |
| `multiply_reduce_fusion` | 0.063 | 1.88 |
| `convert_reduce_fusion` | 0.01 | 0.29 |
| `pad` | 0.005 | 0.15 |
| `iota_compare_fusion` | 0.003 | 0.1 |
| `gather_fusion` | 0.003 | 0.08 |
| `slice` | 0.002 | 0.07 |
| `addmm` | 0.002 | 0.05 |
| `custom-call` | 0.001 | 0.02 |
| `not_reduce_fusion` | 0.0 | 0.0 |
| `iota` | 0.0 | 0.0 |

Overlapped async spans: `copy-start` 34.444 ms, `slice-start` 37.265 ms, `copy-done` 0.01 ms, `slice-done` 0.01 ms.

| Top op | Calls | µs/step | % |
|---|---:|---:|---:|
| `%bitcast_add_fusion.72` (bf16[1,250,1024] | 1 | 28.19 | 0.85 |
| `%bitcast_add_fusion.49` bf16[1,250,1024] | 1 | 25.93 | 0.78 |
| `%bitcast_add_fusion.27` bf16[1,250,1024] | 1 | 25.93 | 0.78 |
| `%bitcast_add_fusion.69` bf16[1,250,1024] | 1 | 25.92 | 0.78 |
| `%bitcast_add_fusion.59` bf16[1,250,1024] | 1 | 25.92 | 0.78 |
| `%bitcast_add_fusion.37` bf16[1,250,1024] | 1 | 25.92 | 0.78 |
| `%bitcast_add_fusion.57` bf16[1,250,1024] | 1 | 25.9 | 0.78 |
| `%bitcast_add_fusion.55` bf16[1,250,1024] | 1 | 25.89 | 0.78 |
| `%bitcast_add_fusion.47` bf16[1,250,1024] | 1 | 25.89 | 0.78 |
| `%bitcast_add_fusion.39` bf16[1,250,1024] | 1 | 25.89 | 0.78 |

## hf_compiled: 10 s (T=500), B=16

Program: **113.221 ms/step**; synchronous ops 112.596 ms (2271 distinct).

| HLO group | ms/step | % of sync ops |
|---|---:|---:|
| `fusion` | 50.218 | 44.6 |
| `copy` | 18.489 | 16.42 |
| `select_bitcast_fusion` | 9.608 | 8.53 |
| `scaled_dot_product_cudnn_attention` | 8.37 | 7.43 |
| `convolution_multiply_fusion` | 4.772 | 4.24 |
| `convolution_convert_fusion` | 4.485 | 3.98 |
| `reshape` | 4.085 | 3.63 |
| `multiply_reduce_fusion` | 4.079 | 3.62 |
| `copy_add_fusion` | 4.045 | 3.59 |
| `as_strided` | 3.592 | 3.19 |
| `convert_reduce_fusion` | 0.772 | 0.69 |
| `gather_fusion` | 0.076 | 0.07 |
| `broadcast_multiply_fusion` | 0.004 | 0.0 |
| `addmm` | 0.001 | 0.0 |
| `custom-call` | 0.001 | 0.0 |
| `iota` | 0.0 | 0.0 |

Overlapped async spans: `copy-start` 101.782 ms, `copy-done` 0.58 ms, `slice-start` 178.721 ms, `slice-done` 0.002 ms.

| Top op | Calls | µs/step | % |
|---|---:|---:|---:|
| `%fusion.2` bf16[8000,4096] | 1 | 408.48 | 0.36 |
| `%select_bitcast_fusion.23` bf16[16,16,500,500] | 1 | 400.79 | 0.36 |
| `%select_bitcast_fusion.16` bf16[16,16,500,500] | 1 | 400.75 | 0.36 |
| `%select_bitcast_fusion.3` bf16[16,16,500,500] | 1 | 400.74 | 0.36 |
| `%select_bitcast_fusion.13` bf16[16,16,500,500] | 1 | 400.55 | 0.36 |
| `%select_bitcast_fusion.14` bf16[16,16,500,500] | 1 | 400.43 | 0.36 |
| `%select_bitcast_fusion.6` bf16[16,16,500,500] | 1 | 400.42 | 0.36 |
| `%select_bitcast_fusion.15` bf16[16,16,500,500] | 1 | 400.38 | 0.36 |
| `%select_bitcast_fusion.8` bf16[16,16,500,500] | 1 | 400.38 | 0.36 |
| `%select_bitcast_fusion.19` bf16[16,16,500,500] | 1 | 400.36 | 0.36 |

## hf_compiled: 30 s (T=1500), B=16

Program: **539.928 ms/step**; synchronous ops 538.45 ms (2269 distinct).

| HLO group | ms/step | % of sync ops |
|---|---:|---:|
| `fusion` | 151.695 | 28.17 |
| `copy` | 131.011 | 24.33 |
| `select_bitcast_fusion` | 86.221 | 16.01 |
| `scaled_dot_product_cudnn_attention` | 52.35 | 9.72 |
| `convolution_multiply_fusion` | 42.99 | 7.98 |
| `reshape` | 22.828 | 4.24 |
| `as_strided` | 14.327 | 2.66 |
| `convolution_convert_fusion` | 13.487 | 2.5 |
| `multiply_reduce_fusion` | 12.218 | 2.27 |
| `copy_add_fusion` | 8.658 | 1.61 |
| `convert_reduce_fusion` | 2.42 | 0.45 |
| `gather_fusion` | 0.233 | 0.04 |
| `broadcast_multiply_fusion` | 0.01 | 0.0 |
| `custom-call` | 0.001 | 0.0 |
| `addmm` | 0.001 | 0.0 |
| `iota` | 0.0 | 0.0 |

Overlapped async spans: `copy-start` 696.928 ms, `copy-done` 1.424 ms, `slice-start` 1035.819 ms, `slice-done` 0.02 ms.

| Top op | Calls | µs/step | % |
|---|---:|---:|---:|
| `%select_bitcast_fusion` bf16[16,16,1500,1500] | 1 | 3598.08 | 0.67 |
| `%select_bitcast_fusion.6` bf16[16,16,1500,1500] | 1 | 3593.59 | 0.67 |
| `%select_bitcast_fusion.16` bf16[16,16,1500,1500] | 1 | 3593.26 | 0.67 |
| `%select_bitcast_fusion.20` bf16[16,16,1500,1500] | 1 | 3593.23 | 0.67 |
| `%select_bitcast_fusion.1` bf16[16,16,1500,1500] | 1 | 3593.04 | 0.67 |
| `%select_bitcast_fusion.14` bf16[16,16,1500,1500] | 1 | 3592.77 | 0.67 |
| `%select_bitcast_fusion.4` bf16[16,16,1500,1500] | 1 | 3592.68 | 0.67 |
| `%select_bitcast_fusion.18` bf16[16,16,1500,1500] | 1 | 3592.67 | 0.67 |
| `%select_bitcast_fusion.2` bf16[16,16,1500,1500] | 1 | 3592.65 | 0.67 |
| `%select_bitcast_fusion.22` bf16[16,16,1500,1500] | 1 | 3592.63 | 0.67 |

## optimized_auto: 5 s (T=250), B=1

Program: **2.461 ms/step**; synchronous ops 2.405 ms (1296 distinct).

| HLO group | ms/step | % of sync ops |
|---|---:|---:|
| `bitcast_add_fusion` | 1.223 | 50.86 |
| `fusion` | 0.775 | 32.23 |
| `copy` | 0.16 | 6.67 |
| `select_convert_fusion` | 0.139 | 5.77 |
| `multiply_reduce_fusion` | 0.082 | 3.4 |
| `convert_reduce_fusion` | 0.01 | 0.43 |
| `pad` | 0.005 | 0.2 |
| `broadcast_multiply_fusion` | 0.004 | 0.17 |
| `slice` | 0.002 | 0.1 |
| `addmm` | 0.002 | 0.08 |
| `custom-call` | 0.001 | 0.03 |
| `maximum_convert_fusion` | 0.001 | 0.02 |
| `broadcast_divide_fusion` | 0.0 | 0.02 |
| `compare_select_fusion` | 0.0 | 0.01 |
| `convert` | 0.0 | 0.01 |
| `not_reduce_fusion` | 0.0 | 0.0 |

Overlapped async spans: `copy-start` 19.039 ms, `slice-start` 31.397 ms, `copy-done` 0.008 ms, `slice-done` 0.008 ms.

| Top op | Calls | µs/step | % |
|---|---:|---:|---:|
| `%bitcast_add_fusion.96` (bf16[1,250,1024] | 1 | 28.48 | 1.18 |
| `%bitcast_add_fusion.93` bf16[1,250,1024] | 1 | 25.87 | 1.08 |
| `%bitcast_add_fusion.49` bf16[1,250,1024] | 1 | 25.87 | 1.08 |
| `%bitcast_add_fusion.91` bf16[1,250,1024] | 1 | 25.86 | 1.08 |
| `%bitcast_add_fusion.89` bf16[1,250,1024] | 1 | 25.85 | 1.07 |
| `%bitcast_add_fusion.94` bf16[1,250,1024] | 1 | 25.84 | 1.07 |
| `%bitcast_add_fusion.85` bf16[1,250,1024] | 1 | 25.82 | 1.07 |
| `%bitcast_add_fusion.83` bf16[1,250,1024] | 1 | 25.82 | 1.07 |
| `%bitcast_add_fusion.53` bf16[1,250,1024] | 1 | 25.82 | 1.07 |
| `%bitcast_add_fusion.81` bf16[1,250,1024] | 1 | 25.81 | 1.07 |

## optimized_auto: 10 s (T=500), B=16

Program: **94.244 ms/step**; synchronous ops 94.206 ms (2060 distinct).

| HLO group | ms/step | % of sync ops |
|---|---:|---:|
| `fusion` | 51.52 | 54.69 |
| `pallas_relative_key_attention_masked_jax` | 11.09 | 11.77 |
| `copy` | 5.872 | 6.23 |
| `convolution_convert_fusion` | 4.671 | 4.96 |
| `pad` | 4.596 | 4.88 |
| `reshape` | 4.142 | 4.4 |
| `multiply_reduce_fusion` | 4.08 | 4.33 |
| `copy_add_fusion` | 4.058 | 4.31 |
| `as_strided` | 2.331 | 2.47 |
| `slice_bitcast_fusion` | 1.067 | 1.13 |
| `convert_reduce_fusion` | 0.772 | 0.82 |
| `broadcast_multiply_fusion` | 0.004 | 0.0 |
| `custom-call` | 0.001 | 0.0 |
| `addmm` | 0.001 | 0.0 |
| `maximum_convert_fusion` | 0.0 | 0.0 |
| `or_not_fusion` | 0.0 | 0.0 |

Overlapped async spans: `copy-start` 74.369 ms, `copy-done` 0.002 ms, `slice-start` 62.81 ms, `slice-done` 0.001 ms.

| Top op | Calls | µs/step | % |
|---|---:|---:|---:|
| `%pallas_relative_key_attention_masked_jax.25` bf16[16,16,512,64] | 1 | 463.05 | 0.49 |
| `%pallas_relative_key_attention_masked_jax.26` bf16[16,16,512,64] | 1 | 462.67 | 0.49 |
| `%pallas_relative_key_attention_masked_jax.24` bf16[16,16,512,64] | 1 | 462.54 | 0.49 |
| `%pallas_relative_key_attention_masked_jax.36` bf16[16,16,512,64] | 1 | 462.06 | 0.49 |
| `%pallas_relative_key_attention_masked_jax.40` bf16[16,16,512,64] | 1 | 462.04 | 0.49 |
| `%pallas_relative_key_attention_masked_jax.29` bf16[16,16,512,64] | 1 | 462.01 | 0.49 |
| `%pallas_relative_key_attention_masked_jax.41` bf16[16,16,512,64] | 1 | 462.01 | 0.49 |
| `%pallas_relative_key_attention_masked_jax.43` bf16[16,16,512,64] | 1 | 462.01 | 0.49 |
| `%pallas_relative_key_attention_masked_jax.34` bf16[16,16,512,64] | 1 | 462.0 | 0.49 |
| `%pallas_relative_key_attention_masked_jax.32` bf16[16,16,512,64] | 1 | 461.99 | 0.49 |

## optimized_auto: 30 s (T=1500), B=16

Program: **407.054 ms/step**; synchronous ops 406.826 ms (2118 distinct).

| HLO group | ms/step | % of sync ops |
|---|---:|---:|
| `fusion` | 161.381 | 39.67 |
| `pallas_relative_key_attention_masked_jax` | 112.869 | 27.74 |
| `copy` | 33.537 | 8.24 |
| `reshape` | 20.859 | 5.13 |
| `pad` | 20.405 | 5.02 |
| `as_strided` | 14.309 | 3.52 |
| `convolution_convert_fusion` | 13.491 | 3.32 |
| `multiply_reduce_fusion` | 12.222 | 3.0 |
| `copy_add_fusion` | 8.646 | 2.13 |
| `slice_bitcast_fusion` | 6.683 | 1.64 |
| `convert_reduce_fusion` | 2.413 | 0.59 |
| `broadcast_multiply_fusion` | 0.01 | 0.0 |
| `custom-call` | 0.001 | 0.0 |
| `addmm` | 0.001 | 0.0 |
| `maximum_convert_fusion` | 0.0 | 0.0 |
| `or_not_fusion` | 0.0 | 0.0 |

Overlapped async spans: `copy-start` 391.431 ms, `copy-done` 0.12 ms, `slice-start` 339.386 ms, `slice-done` 0.075 ms.

| Top op | Calls | µs/step | % |
|---|---:|---:|---:|
| `%pallas_relative_key_attention_masked_jax.44` bf16[16,16,1536,64] | 1 | 4703.55 | 1.16 |
| `%pallas_relative_key_attention_masked_jax.26` bf16[16,16,1536,64] | 1 | 4702.95 | 1.16 |
| `%pallas_relative_key_attention_masked_jax.29` bf16[16,16,1536,64] | 1 | 4702.88 | 1.16 |
| `%pallas_relative_key_attention_masked_jax.31` bf16[16,16,1536,64] | 1 | 4702.88 | 1.16 |
| `%pallas_relative_key_attention_masked_jax.41` bf16[16,16,1536,64] | 1 | 4702.88 | 1.16 |
| `%pallas_relative_key_attention_masked_jax.46` bf16[16,16,1536,64] | 1 | 4702.88 | 1.16 |
| `%pallas_relative_key_attention_masked_jax.32` bf16[16,16,1536,64] | 1 | 4702.87 | 1.16 |
| `%pallas_relative_key_attention_masked_jax.35` bf16[16,16,1536,64] | 1 | 4702.87 | 1.16 |
| `%pallas_relative_key_attention_masked_jax.39` bf16[16,16,1536,64] | 1 | 4702.87 | 1.16 |
| `%pallas_relative_key_attention_masked_jax.27` bf16[16,16,1536,64] | 1 | 4702.86 | 1.16 |

