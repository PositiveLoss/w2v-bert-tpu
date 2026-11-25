"""Benchmark for SeamlessM4T-compatible audio feature extraction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import SeamlessM4TFeatureExtractor

from w2vbert_tpu.features import SAMPLE_RATE, extract_features, load_audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark feature extraction")
    parser.add_argument("--durations", type=int, nargs="+", default=[2, 5, 10, 30])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--audio", default="/content/example_1.wav")
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    hf_fe = SeamlessM4TFeatureExtractor(
        feature_size=80,
        num_mel_bins=80,
        sampling_rate=SAMPLE_RATE,
        padding_value=1.0,
        stride=2,
    )

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)

    for sec in args.durations:
        waves_np = [
            rng.standard_normal(sec * SAMPLE_RATE).astype(np.float32) for _ in range(args.batch)
        ]
        waves_pt = torch.from_numpy(np.stack(waves_np, axis=0))

        # Warmup
        _ = hf_fe(waves_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        _ = extract_features(waves_pt, sampling_rate=SAMPLE_RATE)

        hf_lats = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            _ = hf_fe(waves_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            hf_lats.append((time.perf_counter() - t0) * 1000.0)

        tpu_fe_lats = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            _ = extract_features(waves_pt, sampling_rate=SAMPLE_RATE)
            tpu_fe_lats.append((time.perf_counter() - t0) * 1000.0)

        row = {
            "kind": "synthetic",
            "batch": args.batch,
            "audio_seconds": sec,
            "hf_p50_ms": round(float(np.percentile(hf_lats, 50)), 3),
            "w2vbert_tpu_p50_ms": round(float(np.percentile(tpu_fe_lats, 50)), 3),
            "speedup": round(float(np.percentile(hf_lats, 50) / np.percentile(tpu_fe_lats, 50)), 2),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    audio_path = Path(args.audio)
    if audio_path.exists():
        audio, sr = load_audio(audio_path)
        feats = extract_features(audio, sampling_rate=sr)  # warm-up; also gives the frame count
        t_list = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            extract_features(audio, sampling_rate=sr)
            t_list.append((time.perf_counter() - t0) * 1000.0)
        row = {
            "kind": "real_audio",
            "audio": str(audio_path),
            "samples": int(audio.numel()),
            "frames": int(feats["input_features"].size(1)),
            "w2vbert_tpu_p50_ms": round(float(np.percentile(t_list, 50)), 3),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench_features.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
