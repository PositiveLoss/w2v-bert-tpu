"""End-to-end TPU inference example for w2v-bert-tpu on real audio."""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch

import w2vbert_tpu


def main() -> None:
    parser = argparse.ArgumentParser(description="Run w2v-bert-tpu inference on an audio file")
    parser.add_argument("--audio", default="/content/example_1.wav")
    parser.add_argument("--model-id", default="facebook/w2v-bert-2.0")
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "pallas", "native", "sdpa", "eager"]
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--bucket", action="store_true", help="Pad input features to static frame bucket"
    )
    args = parser.parse_args()

    audio, sr = w2vbert_tpu.load_audio(args.audio)
    inputs = w2vbert_tpu.extract_features(audio, sampling_rate=sr)

    if args.bucket:
        bucketed = w2vbert_tpu.pad_to_bucket(inputs["input_features"], inputs["attention_mask"])
        model_inputs = {
            "input_features": bucketed.input_features.to("tpu", dtype=torch.bfloat16),
            "attention_mask": bucketed.attention_mask.to("tpu"),
        }
    else:
        bucketed = None
        model_inputs = {
            "input_features": inputs["input_features"].to("tpu", dtype=torch.bfloat16),
            "attention_mask": inputs["attention_mask"].to("tpu"),
        }

    model = w2vbert_tpu.load(
        args.model_id,
        dtype=torch.bfloat16,
        device="tpu",
        compile=not args.no_compile,
        backend=args.backend,
        local_files_only=True,
    )

    with torch.inference_mode():
        t0 = time.perf_counter()
        out = model(**model_inputs).last_hidden_state
        torch.accelerator.synchronize()
        first_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        out = model(**model_inputs).last_hidden_state
        torch.accelerator.synchronize()
        warm_ms = (time.perf_counter() - t1) * 1000.0

    if bucketed is not None:
        out = w2vbert_tpu.unpad_output(out, bucketed.original_batch, bucketed.original_frames)

    out_cpu = out.float().cpu()
    print(
        {
            "audio": args.audio,
            "duration_s": round(audio.numel() / sr, 3),
            "input_shape": list(inputs["input_features"].shape),
            "output_shape": list(out.shape),
            "device": str(out.device),
            "dtype": str(out.dtype),
            "backend": args.backend,
            "compiled": not args.no_compile,
            "first_run_ms": round(first_ms, 2),
            "warm_run_ms": round(warm_ms, 2),
            "finite": bool(torch.isfinite(out_cpu).all().item()),
            "output_l2": round(float(out_cpu.norm().item()), 3),
        }
    )


if __name__ == "__main__":
    main()
