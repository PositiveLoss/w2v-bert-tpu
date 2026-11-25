# Environment

The machine used for development, testing and every benchmark in this repository. It was a
Google Colab TPU runtime, which is why some paths start with `/content`.

## Hardware

- **TPU:** one Google Cloud TPU v5e chip (`/dev/vfio/0`).
- **HBM:** 16 GiB, of which 15.75 GiB (16,909,336,064 bytes) is usable according to
  `torch.accelerator.get_memory_info()`.
- **Peak BF16 compute:** 197 TFLOPS.
- **Host:** Linux x86_64, 24 CPUs, 47 GiB RAM.

## Software

| Package | Version | Used for |
|---|---|---|
| Python | 3.13.18 | |
| `torch` | 2.14.1+cpu | PyTorch; the TPU appears as device `tpu` |
| `torch_tpu` | 0.3.0 | TorchTPU: `torch.compile(backend="tpu")`, `torch.tpu.pallas.jax_op` |
| `libtpu` | 0.0.50 | TPU runtime |
| `transformers` | 5.24.1 | `Wav2Vec2BertModel`, `SeamlessM4TFeatureExtractor` |
| `jax` / `jaxlib` | 0.11.2 | Compiling Pallas kernels (see below) |
| `torchaudio` | 2.11.0+cpu | Resampling audio that isn't 16 kHz |
| `soundfile` | 0.14.0 | Reading audio files |
| `scipy` | 1.16.3 | Resampling fallback when torchaudio is missing |
| `xprof` / `tensorboard` | 2.24.2 / 2.21.0 | Profiling |
| `pytest` / `ruff` | 8.4.2 / 0.17.6 | Tests and linting |

`scripts/check_environment.py` prints the installed versions and the TPU details.

## Known problems and fixes

### jax version must match libtpu

The runtime originally came with jax/jaxlib 0.11.2 and libtpu 0.0.50. Installing another package
upgraded jax to 0.12.1, which refuses to compile Pallas kernels for TPU unless libtpu is at
least 0.0.53. Going back to jax/jaxlib 0.11.2 fixed it. `scripts/setup_colab.sh` does this
automatically when it finds libtpu 0.0.50.

### Set `JAX_PLATFORMS=cpu`

In processes that use TorchTPU, set `JAX_PLATFORMS=cpu`. JAX then compiles the Pallas kernels
on the host without trying to open the TPU, which TorchTPU already holds. The tests
(`tests/conftest.py`), the example and the TPU benchmark scripts set it themselves if it is unset.

### Only one process can use the TPU

Only one process at a time can hold `/dev/vfio/0`. A Colab notebook kernel that has imported
`torch_tpu` keeps the device until it is restarted, and CLI tests then fail to get it.
`scripts/setup_colab.sh` kills the notebook kernel for this reason. Run tests and benchmarks
one at a time.

### Preload tcmalloc

TorchTPU recommends `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4` to avoid
lock contention in glibc's allocator.

### Compile caches and edited kernels

`configure_compiler()` (called by `optimize(compile=True)`) turns on TorchTPU's compile caches:

- tier 2: `/dev/shm/torch_tpu_cache/w2vbert_tpu_tier2`
- tier 3: `/tmp/w2vbert_tpu_tier3_cache`

These are keyed by content and safe to keep.

PyTorch's AOTAutograd cache (`/tmp/torchinductor_<user>`) is different: it is keyed by the FX
graph, which has the custom op's name but not the Pallas kernel code. That's why the op names
include a hash of the kernel source and the jax/jaxlib/libtpu versions. If you change a kernel
in a way the hash doesn't cover, delete `/tmp/torchinductor_*` before benchmarking.

### Testing Pallas kernels directly

When no TorchTPU process is running, JAX can use the TPU directly (`JAX_PLATFORMS=tpu`). This
compiles and times kernel variants in seconds, and is how the VMEM and lane-broadcast problems
in `optimization_log.md` were tracked down.

## Test audio

Some tests (`tests/test_checkpoint.py`, `tests/test_features.py`) and the default of
`examples/inference.py` use `/content/example_1.wav`: mono, 16 kHz, 136,245 samples (8.5 s).
That gives 850 mel frames and 425 encoder frames of 160 features. The file is not in the
repository (`*.wav` is in `.gitignore`).

## Reference repositories

Source code consulted during development, at these commits:

| Repository | Commit |
|---|---|
| [w2v-bert-fast](https://github.com/Wataru-Nakata/w2v-bert-fast) | `df6e9e4569edef18053008430e667aec3c98eba6` |
| [torch_tpu](https://github.com/google-pytorch/torch_tpu) | `536e7d6451db14a533327f70e5d0458582744718` |
| [jax](https://github.com/jax-ml/jax) | `59b4bfd116bd4fd254cc8eccfd3a202d59b88c84` |
| [tokamax](https://github.com/openxla/tokamax) | `0c901d55d2ffe2ff25e32a3e176eac0bd94b4ef9` |
