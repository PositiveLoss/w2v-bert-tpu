"""Microbenchmark for Conformer convolution implementations on TPU v5e."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch
from transformers import Wav2Vec2BertConfig
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertConvolutionModule

from w2vbert_tpu.compiler import compile_tpu, configure_compiler, reset_compiler
from w2vbert_tpu.convolution import TPUConvolutionModule


def _time_tpu(fn, *, warmup: int, runs: int) -> dict[str, float]:
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    with torch.inference_mode():
        t0 = time.perf_counter()
        out = fn()
        torch.accelerator.synchronize()
        first_s = time.perf_counter() - t0
        for _ in range(warmup):
            out = fn()
        torch.accelerator.synchronize()
        lats = []
        for _ in range(runs):
            t_s = time.perf_counter()
            out = fn()
            torch.accelerator.synchronize()
            lats.append((time.perf_counter() - t_s) * 1000.0)
    del out
    return {
        "first_run_s": round(first_s, 4),
        "p50_ms": round(float(np.percentile(lats, 50)), 4),
        "p95_ms": round(float(np.percentile(lats, 95)), 4),
        "mean_ms": round(float(statistics.mean(lats)), 4),
        "peak_hbm_gib": round(torch.accelerator.max_memory_allocated() / (1024**3), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Conformer convolution on TPU v5e")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--frames", type=int, nargs="+", default=[250, 500, 1500])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--impls", nargs="+", default=["conv1d", "shift", "unfold"])
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--output-prefix", default="bench_convolution")
    args = parser.parse_args()

    import torch_tpu  # noqa: F401

    from w2vbert_tpu.kernels.causal_conv import register_conv_op

    register_conv_op()

    configure_compiler(recompile_limit=128, fail_on_recompile_limit=True)
    cfg = Wav2Vec2BertConfig(
        hidden_size=1024,
        conv_depthwise_kernel_size=31,
        conformer_conv_dropout=0.0,
        hidden_act="swish",
        layer_norm_eps=1e-5,
    )
    base = Wav2Vec2BertConvolutionModule(cfg).eval().to(device="tpu", dtype=torch.bfloat16)

    variants: dict[str, torch.nn.Module] = {"hf_conv1d": copy.deepcopy(base)}
    for impl in args.impls:
        mod = copy.deepcopy(base)
        mod.__class__ = type(
            "BenchTPUConv", (TPUConvolutionModule, Wav2Vec2BertConvolutionModule), {}
        )
        mod.conv_impl = impl  # type: ignore[attr-defined]
        variants[f"tpu_{impl}"] = mod

    rows: list[dict[str, Any]] = []
    for frames in args.frames:
        x = torch.randn(args.batch, frames, 1024, device="tpu", dtype=torch.bfloat16)
        for name, mod in variants.items():
            reset_compiler(clear_counters=True)
            compiled = compile_tpu(mod, dynamic=False, fullgraph=True)
            row: dict[str, Any] = {"batch": args.batch, "frames": frames, "variant": name}
            try:
                stats = _time_tpu(lambda: compiled(x), warmup=args.warmup, runs=args.runs)
                row.update(status="ok", **stats)
            except Exception as exc:
                row.update(status="error", note=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            print(json.dumps(row), flush=True)
        x = None  # release the input before the next shape
        gc.collect()
        torch.accelerator.empty_cache()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.output_prefix}.json").write_text(json.dumps(rows, indent=2) + "\n")
    fieldnames = [
        "batch",
        "frames",
        "variant",
        "status",
        "first_run_s",
        "p50_ms",
        "p95_ms",
        "mean_ms",
        "peak_hbm_gib",
        "note",
    ]
    with (out_dir / f"{args.output_prefix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


if __name__ == "__main__":
    main()
