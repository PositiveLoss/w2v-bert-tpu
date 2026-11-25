"""Tests for SeamlessM4T-compatible feature extraction against Hugging Face."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoFeatureExtractor, SeamlessM4TFeatureExtractor

from w2vbert_tpu.features import extract_features, load_audio

EXAMPLE_WAV = Path("/content/example_1.wav")


def test_extract_features_matches_reference_batch() -> None:
    reference = SeamlessM4TFeatureExtractor(
        feature_size=80,
        num_mel_bins=80,
        sampling_rate=16000,
        padding_value=1.0,
        stride=2,
    )
    rng = np.random.default_rng(0)
    waves = [(0.1 * rng.standard_normal(n)).astype(np.float32) for n in (16000, 12345, 5000, 1000)]

    expected = reference(waves, sampling_rate=16000, return_tensors="pt")
    got = extract_features(
        [torch.from_numpy(w) for w in waves],
        sampling_rate=16000,
        dtype=torch.float64,
    )

    assert tuple(got["input_features"].shape) == tuple(expected["input_features"].shape)
    torch.testing.assert_close(
        got["attention_mask"], expected["attention_mask"].long(), rtol=0, atol=0
    )
    valid = expected["attention_mask"].bool()
    torch.testing.assert_close(
        got["input_features"].float()[valid],
        expected["input_features"][valid],
        rtol=1e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        got["input_features"].float()[~valid],
        expected["input_features"][~valid],
    )


def test_extract_features_padded_2d_tensor_with_lengths() -> None:
    rng = np.random.default_rng(7)
    waves = [(0.2 * rng.standard_normal(n)).astype(np.float32) for n in (16000, 8000)]
    padded = torch.zeros(2, 16000, dtype=torch.float32)
    padded[0] = torch.from_numpy(waves[0])
    padded[1, :8000] = torch.from_numpy(waves[1])

    from_list = extract_features(waves, sampling_rate=16000, dtype=torch.float32)
    from_tensor = extract_features(
        padded,
        lengths=[16000, 8000],
        sampling_rate=16000,
        dtype=torch.float32,
    )

    torch.testing.assert_close(from_tensor["attention_mask"], from_list["attention_mask"])
    torch.testing.assert_close(from_tensor["input_features"], from_list["input_features"])


def test_extract_features_resampling() -> None:
    sr_in = 22050
    t = torch.linspace(0, 1.0, sr_in, dtype=torch.float32)
    wave = 0.25 * torch.sin(2 * np.pi * 440.0 * t)
    out = extract_features(wave, sampling_rate=sr_in)
    assert out["input_features"].ndim == 3
    assert out["input_features"].shape[0] == 1
    assert out["input_features"].shape[2] == 160
    assert out["attention_mask"].shape == out["input_features"].shape[:2]
    assert torch.isfinite(out["input_features"]).all()


def test_extract_features_example_wav_matches_hf() -> None:
    if not EXAMPLE_WAV.exists():
        pytest.skip(f"{EXAMPLE_WAV} not found")
    fe = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0", local_files_only=True)
    audio, sr = load_audio(EXAMPLE_WAV)
    expected = fe(audio.numpy(), sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
    got = extract_features(audio, sampling_rate=sr, dtype=torch.float64)

    assert tuple(got["input_features"].shape) == tuple(expected["input_features"].shape)
    assert torch.equal(got["attention_mask"], expected["attention_mask"].long())
    torch.testing.assert_close(
        got["input_features"].float(),
        expected["input_features"],
        rtol=1e-4,
        atol=1e-4,
    )


def test_extract_features_rejects_short_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="at least 2 mel frames"):
        extract_features(torch.zeros(559), sampling_rate=16000)
    with pytest.raises(ValueError, match="NaN or Inf"):
        extract_features(torch.tensor([0.0, float("nan")] + [0.0] * 1000), sampling_rate=16000)
    with pytest.raises(ValueError, match="sampling_rate"):
        extract_features(torch.zeros(1000), sampling_rate=0)
