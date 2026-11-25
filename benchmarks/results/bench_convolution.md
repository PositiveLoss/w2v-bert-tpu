# Conformer convolution microbenchmark (TPU v5e, BF16, B=16, compiled, 50 runs)

| Frames | Variant | p50 (ms) | vs HF | Peak HBM (GiB) | Status |
|---:|---|---:|---:|---:|---|
| 250 | hf_conv1d | 0.6498 | 1.00x | 0.0699 | ok |
| 250 | tpu_conv1d | 0.6892 | 1.06x | 0.0654 | ok |
| 250 | tpu_shift | 0.9343 | 1.44x | 0.0682 | ok |
| 250 | tpu_unfold | - | - | - | compile-time HBM OOM (31.4 GiB temporaries) |
| 500 | hf_conv1d | 0.9175 | 1.00x | 0.168 | ok |
| 500 | tpu_conv1d | 1.0014 | 1.09x | 0.1259 | ok |
| 500 | tpu_shift | 1.5692 | 1.71x | 0.1294 | ok |
| 500 | tpu_unfold | 5.8751 | 6.40x | 1.5581 | ok |
| 1500 | hf_conv1d | 2.0194 | 1.00x | 1.9711 | ok |
| 1500 | tpu_conv1d | 2.4505 | 1.21x | 1.7477 | ok |
| 1500 | tpu_shift | 4.0145 | 1.99x | 1.7526 | ok |
| 1500 | tpu_unfold | 11.8887 | 5.89x | 3.231 | ok |

Decision: keep the compiled Hugging Face convolution (`optimize(convolution=False)` default).

## Pallas channels-last causal depthwise kernel (`conv_impl="pallas"`, port of w2v-bert-fast)

Same harness, B=16, p50 ms:

| Frames | HF module | `TPUConvolutionModule` + Pallas depthwise | + XLA `conv1d` depthwise |
|---:|---:|---:|---:|
| 250 | 0.601 | 0.702 (1.17x) | 0.688 |
| 500 | 0.884 | 0.987 (1.12x) | 1.030 |
| 1500 | 2.003 | 2.407 (1.20x) | 2.443 |

The Pallas depthwise kernel matches XLA's depthwise conv; the channels-last module loses on the
surrounding LayerNorm/pointwise chain. Decision: opt-in only; the HF module stays the default.
