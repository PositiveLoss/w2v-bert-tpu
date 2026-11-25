# Numerical correctness

Unless noted otherwise, TPU results were measured on TPU v5e in BF16 with
`torch.compile(backend="tpu", fullgraph=True)`, and compared with Hugging Face
`Wav2Vec2BertModel` running in FP32 on CPU.

Metrics: `max_abs` is the largest absolute difference, `rel_l2` is
`‖output − reference‖ / ‖reference‖`, and `cos` is cosine similarity.

## Pass criteria

| Check | Criterion | Result |
|---|---|---|
| Full 24-layer output, real audio | cos ≥ 0.999, max_abs ≤ 0.15, all values finite | Pass (table below) |
| One attention layer, native and Pallas | cos ≥ 0.9995 | Pass for every native variant and Pallas, T=1…1536, with masking (`test_tpu_native_and_pallas_attention_parity`). Pallas tile sweep: cos ≥ 0.99997 |
| Conformer convolution | cos ≥ 0.9995 | Default path is unmodified Hugging Face code. Opt-in `TPUConvolutionModule` matches HF to 1e-11 in FP64 on CPU and rel_l2 < 1.5e-2 in BF16 on TPU |
| `state_dict()` keys and values | Identical to HF | Pass (`tests/test_checkpoint.py`, `tests/test_model.py`) |
| Compilation | No graph breaks, one compile per shape | Pass for every benchmarked shape |

## Full model on real audio

Clip: `/content/example_1.wav`, 8.5 s at 16 kHz → 425 encoder frames, batch 1. Test:
`tests/test_checkpoint.py::test_pretrained_real_audio_and_layers_tpu`.

`last_hidden_state` compared with FP32 Hugging Face on CPU:

| Path | max_abs | rel_l2 | cos | Worst layer rel_l2 / cos |
|---|---:|---:|---:|---|
| HF eager, BF16 on TPU (not compiled) | 1.65e-1 | 2.23e-2 | 0.999754 | 7.03e-2 / 0.997522 |
| HF compiled, BF16 on TPU | 9.69e-2 | 2.11e-2 | 0.999777 | – |
| `backend="native"` | 9.69e-2 | 2.11e-2 | 0.999777 | 4.70e-2 / 0.998891 |
| `backend="pallas"` | 9.93e-2 | 2.17e-2 | 0.999764 | 5.17e-2 / 0.998657 |
| `backend="auto"` (picks Pallas at T=425) | 9.93e-2 | 2.17e-2 | 0.999764 | 5.17e-2 / 0.998657 |

Per-layer hidden states are compared against Hugging Face's own BF16 result for the same layer,
not against a fixed bound: `rel_l2 ≤ max(2.5e-2, 1.25 × HF BF16)` and
`cos ≥ min(0.999, HF BF16 − 1e-3)`. In early layers BF16 itself does worse than a fixed 2.5e-2,
so a fixed bound would fail even for unmodified HF. The final-output criteria are fixed.

A padded batch of four inputs with lengths 320, 250, 73 and 1 frames passes for both native and
Pallas, with rel_l2 < 2.5e-2 on the real (non-padded) frames.

## Pallas kernel, one layer

`B=2, H=16, D=64`, random BF16 inputs, second sequence masked to half length, compared with the
FP32 `relative_key_reference`. Each kernel compiled from a clean cache.

| Tile | T=250 cos / max_abs | T=500 | T=1500 |
|---|---|---|---|
| 128×128 | 0.999982 / 0.0116 | 0.999971 / 0.0104 | 0.999994 / 0.0113 |
| 256×256 | 0.999982 / 0.0116 | 0.999971 / 0.0104 | 0.999995 / 0.0113 |
| 256×128 | 0.999982 / 0.0116 | 0.999971 / 0.0104 | 0.999994 / 0.0113 |
| 512×256 | 0.999982 / 0.0116 | 0.999971 / 0.0104 | 0.999995 / 0.0113 |
| 512×512 | 0.999982 / 0.0116 | 0.999971 / 0.0104 | 0.999994 / 0.0113 |

On CPU, in Pallas interpret mode with FP32 (`tests/test_attention.py`), the kernel matches the
reference to 1e-4 for lengths 1–1100 and tiles 128×128, 256×256, 256×128, 128×256 and 512×256.
Rows where every key is masked return zeros, not NaN.

## Packed Q/K/V kernel (default Pallas path)

- CPU interpret mode, FP32, H=4: matches the reference to 1e-4 for T=128–1024, with and without
  masking, tiles 128–512 (`test_fused_qkv_interpret_matches_reference`).
- TPU, attention block including the projection, batch 16: at most 0.002 different from the
  `[B, H, T, D]` kernel path, and 0.007–0.014 max_abs from the FP32 reference.
- Full model, 4 random seeds, batch 4: same rel_l2 as the `[B, H, T, D]` path at T=512. At
  T=320 (not a multiple of the tile) the two compiled programs differ by BF16 rounding only. The
  ratio of their errors ranges from 0.93 to 1.61 with no consistent direction. The real-audio
  result is unchanged (max_abs 9.93e-2, cos 0.999764).
- Mixed-length batch (320/250/73/1 frames): Pallas rel_l2 2.53e-2, unmodified HF in BF16
  2.12e-2 on the same input. The test allows up to 1.25× HF BF16.
- Padding test: the check is that bucketing does not make results worse compared with FP32. It
  does not require T=250 and T=256 to give identical output. Those are different compiled
  programs, and their BF16 outputs differ by about 2e-2 even in rows with no padding.

## CPU FP64 checks

`tests/test_attention.py` checks every native variant against the reference in FP64 for several
lengths and clip sizes.

`tests/test_model.py` checks the optimized model against Hugging Face in FP64 on CPU
(`rtol=1e-9`, `atol=1e-10`). It covers packed and separate Q/K/V, `convolution=True/False`,
padding masks, `num_layers`, `output_attentions=True` (weights equal HF eager) and
`torch.compile(fullgraph=True)` at three sequence lengths.

## Training gradients

`tests/test_training.py`, on CPU:

- **Pallas backward kernels**, in interpret mode, FP32, against PyTorch autograd of the
  reference: dq, dk, dv and dE agree to 1e-4 relative error, for T = 250–600, tiles 128–256 and
  key lengths down to 1. Measured errors were 4e-7 to 2e-5, the same for 512×512 tiles at
  T = 1100.
- **Whole model through the kernels:** a `Wav2Vec2BertForCTC` with an adapter, encoder and
  adapter forced onto the kernel path. Every parameter's gradient of the CTC loss matches Hugging
  Face's to 1e-4 of its largest value (about 1e-6 in practice), including both layers'
  `distance_embedding`. `linear_k.bias` gets a gradient that is 0 in exact arithmetic (softmax
  ignores a per-row constant), so it is checked against an absolute 1e-8.
- **Native path**, FP64: base model and CTC model, every parameter's gradient equal to Hugging
  Face's at `rtol=1e-9`.
- **Stale caches:** the gradient path ignores the prepacked weights, and `model.eval()` refreshes
  them in place.

None of this has run on a TPU yet.

## Notes

- Unmodified Hugging Face in BF16 (`backend="eager"`) fails the max_abs ≤ 0.15 criterion on the
  real clip (0.165). The optimized paths are closer to FP32 because they compute softmax in FP32.
- In the Pallas kernel, padded query rows produce values that `unpad_output` slices off. Rows
  with no valid keys produce zeros, not NaN.
