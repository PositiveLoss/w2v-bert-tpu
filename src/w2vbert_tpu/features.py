"""SeamlessM4T-compatible log-mel feature extraction for w2v-BERT 2.0.

Produces 160-dimensional stacked mel-filterbank features matching Hugging Face's
``SeamlessM4TFeatureExtractor`` configuration for ``facebook/w2v-bert-2.0``:

* 16 kHz mono float waveforms scaled by ``2**15`` (Kaldi int16 convention)
* 25 ms frames (400 samples) every 10 ms (160 samples), uncentered
* Per-frame DC removal, pre-emphasis ``0.97``, non-periodic Povey window, 512-point power spectrum
* 80 Kaldi mel bins (20 Hz to 8 kHz), ``log(max(mel, 1.1920929e-07))``
* Per-bin normalization over each utterance's valid frames using unbiased variance + ``1e-7``
* Right-padding with ``1.0`` to the longest utterance (and an even frame count), stacking pairs
  of consecutive 80-bin frames into 160-dim features, with ``attention_mask`` keeping every
  2nd frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

SAMPLE_RATE = 16000
FRAME_LENGTH = 400
HOP_LENGTH = 160
FFT_LENGTH = 512
NUM_MEL_BINS = 80
STRIDE = 2
PADDING_VALUE = 1.0
MEL_FLOOR = 1.192092955078125e-07

_CONSTANTS_CACHE: dict[tuple[str, torch.dtype], tuple[Tensor, Tensor]] = {}


def _get_constants(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Lazily compute and cache the Povey window and Kaldi mel filter bank."""
    key = (str(device), dtype)
    if key not in _CONSTANTS_CACHE:
        from transformers.audio_utils import mel_filter_bank, window_function

        mel = mel_filter_bank(
            num_frequency_bins=FFT_LENGTH // 2 + 1,
            num_mel_filters=NUM_MEL_BINS,
            min_frequency=20,
            max_frequency=SAMPLE_RATE // 2,
            sampling_rate=SAMPLE_RATE,
            norm=None,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        )
        window = window_function(FRAME_LENGTH, "povey", periodic=False)
        _CONSTANTS_CACHE[key] = (
            torch.as_tensor(window, dtype=dtype, device=device),
            torch.as_tensor(mel, dtype=dtype, device=device),
        )
    return _CONSTANTS_CACHE[key]


def frame_count(samples: Tensor) -> Tensor:
    """Compute the number of uncentered 25 ms frames for each sample count."""
    return torch.where(
        samples >= FRAME_LENGTH,
        1 + (samples - FRAME_LENGTH) // HOP_LENGTH,
        torch.zeros_like(samples),
    )


def log_mel(waves: Tensor) -> Tensor:
    """Compute ``[B, F, 80]`` Kaldi log-mel features from zero-padded ``[B, N]`` 16 kHz audio."""
    window, mel = _get_constants(waves.device, waves.dtype)
    frames = (waves * 32768.0).unfold(-1, FRAME_LENGTH, HOP_LENGTH)
    frames = frames - frames.mean(dim=-1, keepdim=True)
    frames = torch.cat(
        (frames[..., :1] * (1.0 - 0.97), frames[..., 1:] - 0.97 * frames[..., :-1]),
        dim=-1,
    )
    power = torch.fft.rfft(frames * window, n=FFT_LENGTH).abs().pow(2)
    return torch.matmul(power, mel).clamp_min(MEL_FLOOR).log()


def load_audio(
    path: str | Path,
    *,
    target_sampling_rate: int = SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    """Load an audio file from disk as a mono float32 tensor, resampling to 16 kHz if needed."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    if tensor.ndim == 2:
        tensor = tensor.mean(dim=-1)
    if sr != target_sampling_rate:
        tensor = _resample(tensor.unsqueeze(0), sr, target_sampling_rate).squeeze(0)
        sr = target_sampling_rate
    return tensor, int(sr)


def _resample(waves: Tensor, orig_freq: int, new_freq: int) -> Tensor:
    if orig_freq == new_freq:
        return waves
    try:
        import torchaudio

        return torchaudio.functional.resample(waves, orig_freq, new_freq)
    except ImportError:
        import math

        from scipy.signal import resample_poly

        gcd = math.gcd(orig_freq, new_freq)
        up = new_freq // gcd
        down = orig_freq // gcd
        resampled = resample_poly(waves.cpu().numpy(), up=up, down=down, axis=-1)
        return torch.from_numpy(np.asarray(resampled, dtype=np.float32)).to(
            device=waves.device, dtype=waves.dtype
        )


def _coerce_waveforms(
    waves: Tensor | np.ndarray | Sequence[Tensor | np.ndarray],
    lengths: Tensor | Sequence[int] | None,
) -> tuple[Tensor, Tensor]:
    if isinstance(waves, np.ndarray):
        waves = torch.from_numpy(np.asarray(waves))
    if isinstance(waves, (list, tuple)):
        if not waves:
            raise ValueError("waves sequence must not be empty")
        tensors = [torch.as_tensor(w).reshape(-1) for w in waves]
        if lengths is None:
            lengths_t = torch.tensor([t.numel() for t in tensors], dtype=torch.long)
        else:
            lengths_t = torch.as_tensor(lengths, dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return padded, lengths_t
    if not isinstance(waves, Tensor):
        raise TypeError(f"Unsupported waves type: {type(waves).__name__}")
    if waves.ndim == 1:
        waves = waves.unsqueeze(0)
    elif waves.ndim != 2:
        raise ValueError(
            "waves must be 1-D [N], 2-D [B, N], or a sequence of 1-D waveforms; "
            f"got shape {tuple(waves.shape)}"
        )
    if lengths is None:
        lengths_t = torch.full((waves.size(0),), waves.size(1), dtype=torch.long)
    else:
        lengths_t = torch.as_tensor(lengths, dtype=torch.long)
        if lengths_t.shape != (waves.size(0),):
            raise ValueError(
                f"lengths must have shape ({waves.size(0)},), got {tuple(lengths_t.shape)}"
            )
    return waves, lengths_t


def extract_features(
    waves: Tensor | np.ndarray | Sequence[Tensor | np.ndarray],
    lengths: Tensor | Sequence[int] | None = None,
    sampling_rate: int = SAMPLE_RATE,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """Extract w2v-BERT 2.0 ``input_features`` ``[B, T, 160]`` and ``attention_mask`` ``[B, T]``.

    Args:
        waves: A 1-D mono waveform ``[N]``, a 2-D batch ``[B, N]``, or a sequence of 1-D waveforms.
        lengths: Optional sample counts ``[B]`` for padded 2-D ``waves``.
        sampling_rate: Input sample rate in Hz; resampled to 16 kHz when different.
        device: Target compute device (defaults to input tensor device or CPU).
        dtype: Computation dtype for log-mel extraction (default ``torch.float32``).

    Returns:
        Dictionary with ``input_features`` (``[B, T, 160]``) and ``attention_mask``
        (``[B, T]``, ``torch.long``).
    """
    if sampling_rate <= 0:
        raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")
    waves_t, lengths_t = _coerce_waveforms(waves, lengths)
    if not torch.isfinite(waves_t).all():
        raise ValueError("waves contains NaN or Inf values")

    if device is not None:
        waves_t = waves_t.to(device)
    waves_t = waves_t.to(dtype)
    lengths_t = lengths_t.to(device=waves_t.device, dtype=torch.long)

    if sampling_rate != SAMPLE_RATE:
        waves_t = _resample(waves_t, sampling_rate, SAMPLE_RATE)
        lengths_t = (lengths_t * SAMPLE_RATE + sampling_rate - 1) // sampling_rate
        lengths_t = lengths_t.clamp_max(waves_t.size(1))

    frames = frame_count(lengths_t)
    if bool((frames < 2).any()):
        min_samples = FRAME_LENGTH + HOP_LENGTH
        raise ValueError(
            f"Each waveform needs at least 2 mel frames ({min_samples} samples at {SAMPLE_RATE} Hz)"
        )

    features = log_mel(waves_t)  # [B, F, 80]
    valid = torch.arange(features.size(1), device=waves_t.device)[None, :] < frames[:, None]
    count = frames[:, None, None].to(features.dtype)
    masked = features.masked_fill(~valid[..., None], 0.0)
    mean = masked.sum(dim=1, keepdim=True) / count
    var = (features - mean).masked_fill(~valid[..., None], 0.0).pow(2).sum(dim=1, keepdim=True) / (
        count - 1.0
    )
    features = ((features - mean) / torch.sqrt(var + 1e-7)).masked_fill(
        ~valid[..., None], PADDING_VALUE
    )

    longest = int(frames.max().item())
    even = longest + (longest % STRIDE)
    features = features[:, :longest]
    valid = valid[:, :longest]
    if even > longest:
        features = torch.nn.functional.pad(features, (0, 0, 0, even - longest), value=PADDING_VALUE)
        valid = torch.nn.functional.pad(valid, (0, even - longest), value=False)

    batch = features.size(0)
    return {
        "input_features": features.reshape(batch, even // STRIDE, NUM_MEL_BINS * STRIDE),
        "attention_mask": valid[:, 1::STRIDE].long(),
    }


__all__ = [
    "FFT_LENGTH",
    "FRAME_LENGTH",
    "HOP_LENGTH",
    "MEL_FLOOR",
    "NUM_MEL_BINS",
    "PADDING_VALUE",
    "SAMPLE_RATE",
    "STRIDE",
    "extract_features",
    "frame_count",
    "load_audio",
    "log_mel",
]
