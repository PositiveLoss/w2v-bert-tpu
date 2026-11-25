# w2v-bert-tpu

Fast inference for [`facebook/w2v-bert-2.0`](https://huggingface.co/facebook/w2v-bert-2.0)
(Hugging Face `Wav2Vec2BertModel`: 24 Conformer layers, 580M parameters) and models fine-tuned
from it, such as the speech recognizer
[`speech-uk/w2v-bert-v3`](https://huggingface.co/speech-uk/w2v-bert-v3) (`Wav2Vec2BertForCTC`),
on a single TPU v5e chip. It uses TorchTPU (`torch.compile(backend="tpu")`) and a Pallas kernel
for the model's relative-position attention.

What you get:

- **Same model, same weights.** `optimize(model)` swaps the classes of some Hugging Face modules
  in place. Parameter names and `state_dict()` are unchanged, so existing checkpoints load as is.
- **Faster than compiled Hugging Face at every measured shape**: 1.01–1.03× on 2 s clips,
  1.08–1.55× at 5 s, 1.30–1.58× at 10 s and 1.67–2.52× at 30 s. Peak memory is also lower on
  long inputs (30 s, batch 16: 7.7 GiB instead of 10.7 GiB).
- **No approximations.** On a real 8.5 s clip, BF16 output on TPU matches FP32 Hugging Face on
  CPU with cosine similarity 0.99976 and max abs error 0.099. That is closer to FP32 than
  Hugging Face's own BF16 output (max abs error 0.165).
- **Compiles cleanly** with `fullgraph=True`: no graph breaks, one compile per input shape, no
  device-to-host syncs inside the forward pass.

## Install

```bash
pip install -e ".[audio]"
```

You also need a TPU v5e host with TorchTPU (`torch_tpu`), `libtpu` and a matching `jax`
already installed. The `pallas` extra only asks for `jax>=0.9.2`, so on a fresh environment it
can pull a jax that is too new for your libtpu. See [Requirements](#requirements).

## Usage

```python
import torch
import w2vbert_tpu

# Loads in BF16, moves to the TPU, optimizes and compiles.
model = w2vbert_tpu.load("facebook/w2v-bert-2.0")

audio, sr = w2vbert_tpu.load_audio("speech.wav")  # mono, resampled to 16 kHz
# Same features as Hugging Face's SeamlessM4TFeatureExtractor.
inputs = w2vbert_tpu.extract_features(audio, sampling_rate=sr)

with torch.inference_mode():
    hidden = model(
        input_features=inputs["input_features"].to("tpu", torch.bfloat16),
        attention_mask=inputs["attention_mask"].to("tpu"),
    ).last_hidden_state  # [B, T, 1024]
```

Each new input shape triggers a compile (8–30 s). For serving, pad inputs to a fixed set of
sizes ("buckets") so the number of compiles stays small:

```python
batch = w2vbert_tpu.pad_to_bucket(inputs["input_features"], inputs["attention_mask"])
with torch.inference_mode():
    out = model(
        input_features=batch.input_features.to("tpu", torch.bfloat16),
        attention_mask=batch.attention_mask.to("tpu"),
    ).last_hidden_state
out = w2vbert_tpu.unpad_output(out, batch.original_batch, batch.original_frames)
```

The default frame buckets are 128, 256, 512, 768, 1024, 1536, 2048, 3072 and 6144. Padding to
a multiple of 128 also makes long batched inputs about 5% faster (30 s, batch 16). For short
clips the extra padding can cost more than it saves; see
[`docs/performance.md`](docs/performance.md). `w2vbert_tpu.prewarm(model)` compiles a list of
buckets ahead of time.

### Fine-tuned speech recognition (CTC)

`load()` builds the class named in the checkpoint's `config.json`, so a fine-tuned CTC model
loads as `Wav2Vec2BertForCTC` with its head and adapter. The encoder is optimized as usual. The
adapter's attention (one layer, about 2% of the compute) runs through the same Pallas kernel on
long inputs. Hugging Face's version stores a `[B, 16, T/2, T/2]` score matrix (4.3 GiB at
120 s, batch 16); the kernel never does. `adapter=False` keeps Hugging Face's version. The head
is unchanged.

```python
import sentencepiece as spm
from huggingface_hub import hf_hub_download

model = w2vbert_tpu.load("speech-uk/w2v-bert-v3")
tokenizer = spm.SentencePieceProcessor(
    model_file=hf_hub_download("speech-uk/w2v-bert-v3", "tokenizer.model")
)

batch = w2vbert_tpu.pad_to_bucket(inputs["input_features"], inputs["attention_mask"])
with torch.inference_mode():
    logits = model(
        input_features=batch.input_features.to("tpu", torch.bfloat16),
        attention_mask=batch.attention_mask.to("tpu"),
    ).logits

# The adapter halves the frame rate, so count valid logit frames from the original mask.
lengths = w2vbert_tpu.ctc_output_lengths(model.config, inputs["attention_mask"].sum(-1))
token_ids = w2vbert_tpu.ctc_greedy_decode(logits, lengths, blank_id=0, ignore_ids=[1])
texts = [tokenizer.decode_ids(ids).strip() for ids in token_ids]
```

`ctc_greedy_decode` picks the best token per frame, merges repeats and drops blanks. It skips
padded rows and frames itself, so there is no need for `unpad_output` here. Use your model's
blank id (`config.pad_token_id`) and any ids to drop (here `1` is `<unk>`).

Checked on CPU in FP32 with `speech-uk/w2v-bert-v3` and the six example clips from the
[Yehor/w2v-bert-uk-v3](https://huggingface.co/spaces/Yehor/w2v-bert-uk-v3) Space: the optimized
model gives exactly the same text as unmodified Hugging Face, alone and in a padded batch.
`extract_features` matches the model's own `asr_features` extractor to within 7e-4.

### Options

If you already have a loaded model, call
`w2vbert_tpu.optimize(model, compile=True)`. Main options:

| Option | Default | Meaning |
|---|---|---|
| `backend` | `"auto"` | Attention: `"auto"`, `"pallas"`, `"native"`, `"sdpa"`, or `"eager"` (unmodified HF) |
| `compile` | `False` in `optimize`, `True` in `load` | Wrap with `torch.compile(backend="tpu", fullgraph=True)` |
| `compile_strategy` | `"whole"` | `"layer"` compiles one layer and reuses it: faster to compile, same speed |
| `fused_qkv` | `True` | Pallas kernel reads the packed Q/K/V projection directly |
| `convolution` | `False` | Replace the Conformer conv with a channels-last version (slower on v5e) |
| `num_layers` | `None` | Keep only the first N encoder layers |
| `adapter` | `True` | CTC models: run the adapter's attention through the Pallas kernel on long inputs |

`output_attentions=True` works, but falls back to Hugging Face's eager attention for that call.

There is a runnable script in [`examples/inference.py`](examples/inference.py), and a Colab
notebook that installs everything and transcribes sample clips with `speech-uk/w2v-bert-v3` in
[`examples/inference.ipynb`](examples/inference.ipynb).

The performance numbers below are for the base model. A CTC model adds one adapter layer and a
linear head on top of the same 24-layer encoder, but it hasn't been benchmarked on TPU yet. To
measure it, including the adapter's memory with and without the kernel:

```bash
python benchmarks/bench_model.py --model-id speech-uk/w2v-bert-v3 \
    --variants hf_compiled optimized_auto optimized_auto_hf_adapter \
    --durations 5 30 60 120 --batches 1 16 --output-prefix benchmark_ctc
python benchmarks/profile_model.py --model-id speech-uk/w2v-bert-v3 \
    --variants optimized_auto --configs 5:1 30:16 120:16 --output-prefix profile_ctc
```

## Performance

TPU v5e, BF16, median (p50) latency per forward pass:

| Audio | Batch | HF compiled | w2vbert_tpu | Speedup |
|---|---:|---:|---:|---:|
| 5 s | 1 | 5.10 ms | 4.04 ms | 1.26× |
| 5 s | 16 | 53.55 ms | 38.30 ms | 1.40× |
| 10 s | 1 | 9.11 ms | 7.02 ms | 1.30× |
| 10 s | 16 | 117.58 ms | 78.79 ms | 1.49× |
| 30 s | 1 | 57.71 ms | 22.90 ms | 2.52× |
| 30 s | 16 | 545.15 ms | 327.41 ms (309.86 ms bucketed) | 1.67× |

The full table, including batch 4 and 8, hardware utilization (MFU) and memory, is in
[`benchmarks/results/benchmark_summary.md`](benchmarks/results/benchmark_summary.md).

## How it works

w2v-BERT 2.0 adds a learned relative-position term to every attention score. Hugging Face
computes it by building a `[T, T, 64]` tensor and a `[B, H, T, T]` bias in every layer, so
memory and time grow with the square of the input length. This package never builds either:

- **Longer inputs** (more than 256 frames, or more than 128 frames at batch ≥ 8) use a Pallas
  kernel based on [w2v-bert-fast](https://github.com/Wataru-Nakata/w2v-bert-fast). It builds the
  position bias one tile at a time in on-chip memory and uses an FP32 online softmax. It reads
  Q, K and V straight from one packed projection and writes output in the layout the next layer
  expects, so no transposes or copies are needed.
- **Shorter inputs** use plain PyTorch: compute the 73 possible position scores per query
  (`q @ Eᵀ`) and gather them into the attention logits.
- Everything else (feed-forward, convolution, LayerNorm) is the unchanged Hugging Face code,
  compiled by XLA. A Pallas depthwise convolution exists (`convolution=True,
  conv_impl="pallas"`), but it was 12–20% slower than XLA on v5e, so it is off by default.

More detail:

- [`docs/architecture.md`](docs/architecture.md): how the code is organized and how the kernel works
- [`docs/performance.md`](docs/performance.md): all benchmarks and tuning advice
- [`docs/numerical_correctness.md`](docs/numerical_correctness.md): accuracy checks
- [`docs/optimization_log.md`](docs/optimization_log.md): what was tried, what worked, what didn't
- [`docs/environment.md`](docs/environment.md): the test machine and known setup problems

## Requirements

Tested on one TPU v5e chip with Python 3.13, torch 2.14.1, torch_tpu 0.3.0, libtpu 0.0.50,
jax/jaxlib 0.11.2 and transformers 5.24.1.

Use jax 0.11.2 with libtpu 0.0.50. Newer jax (0.12+) needs libtpu ≥ 0.0.53 to compile Pallas
kernels. `scripts/setup_colab.sh` checks this and pins jax if needed.

Set these in every process that uses the TPU:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4   # recommended by TorchTPU
export JAX_PLATFORMS=cpu   # JAX only compiles the Pallas kernels; TorchTPU owns the chip
```

Only one process can use the TPU at a time. Run tests and benchmarks one after another.

## Development

```bash
pip install -e ".[audio,dev]"
ruff check . && ruff format --check .
pyright
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest          # CPU tests
W2VBERT_RUN_TPU_TESTS=1 pytest -v               # CPU + TPU tests
python benchmarks/bench_model.py --variants hf_compiled optimized_auto
python benchmarks/summarize.py
```

Some tests and `examples/inference.py` expect a 16 kHz audio file at `/content/example_1.wav`
(the Colab path used during development). Pass `--audio` to the example to use another file.

## License

Copyright 2026 Yehor Smoliakov. Apache 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
