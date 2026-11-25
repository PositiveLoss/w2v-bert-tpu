"""Microbenchmark for w2v-BERT 2.0 relative-key attention and QKV projection on TPU v5e."""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F

from w2vbert_tpu._tpu import torch_tpu
from w2vbert_tpu.attention import (
    build_distance_indices,
    build_flat_bucket_indices,
    native_relative_key_attention,
    relative_key_reference,
)
from w2vbert_tpu.compiler import compile_tpu, configure_compiler, reset_compiler
from w2vbert_tpu.kernels.relative_attention import (
    pallas_relative_key_attention,
    register_pallas_ops,
)


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


def bench_qkv_projections(
    batch: int, frames_list: list[int], warmup: int, runs: int
) -> list[dict[str, Any]]:
    hidden_size = 1024
    rows: list[dict[str, Any]] = []
    configure_compiler(recompile_limit=128, fail_on_recompile_limit=True)

    for frames in frames_list:
        rows.extend(_bench_qkv_one(batch, frames, hidden_size, warmup, runs))
        gc.collect()
        torch.accelerator.empty_cache()
    return rows


def _bench_qkv_one(
    batch: int, frames: int, hidden_size: int, warmup: int, runs: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    x = torch.randn(batch, frames, hidden_size, device="tpu", dtype=torch.bfloat16)
    wq = torch.nn.Linear(hidden_size, hidden_size, device="tpu", dtype=torch.bfloat16)
    wk = torch.nn.Linear(hidden_size, hidden_size, device="tpu", dtype=torch.bfloat16)
    wv = torch.nn.Linear(hidden_size, hidden_size, device="tpu", dtype=torch.bfloat16)
    packed_w = torch.cat([wq.weight, wk.weight, wv.weight], dim=0).contiguous()
    packed_b = torch.cat([wq.bias, wk.bias, wv.bias], dim=0).contiguous()

    def _separate(inp):
        return wq(inp), wk(inp), wv(inp)

    def _runtime_cat(inp):
        w = torch.cat([wq.weight, wk.weight, wv.weight], dim=0)
        b = torch.cat([wq.bias, wk.bias, wv.bias], dim=0)
        return F.linear(inp, w, b).view(batch, frames, 3, 16, 64)

    def _prepacked(inp):
        return F.linear(inp, packed_w, packed_b).view(batch, frames, 3, 16, 64)

    for name, fn in (
        ("qkv_separate", _separate),
        ("qkv_runtime_cat", _runtime_cat),
        ("qkv_prepacked", _prepacked),
    ):
        reset_compiler(clear_counters=True)
        compiled_fn = compile_tpu(fn, dynamic=False, fullgraph=True)
        stats = _time_tpu(lambda: compiled_fn(x), warmup=warmup, runs=runs)
        row = {
            "kind": "qkv",
            "batch": batch,
            "frames": frames,
            "variant": name,
            "status": "ok",
            **stats,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def _bench_attention_frames(
    args: argparse.Namespace, frames: int, heads: int, head_dim: int, left: int, right: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    q = torch.randn(args.batch, heads, frames, head_dim, device="tpu", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    emb = torch.randn(left + right + 1, head_dim, device="tpu", dtype=torch.bfloat16)
    e_pad = F.pad(emb, (0, 0, 0, 128 - (left + right + 1))).contiguous()
    dist = build_distance_indices(frames, frames, left, right, q.device)
    flat_idx = build_flat_bucket_indices(frames, frames, left, right, q.device)

    for variant in args.variants:
        reset_compiler(clear_counters=True)
        row: dict[str, Any] = {
            "kind": "attention",
            "batch": args.batch,
            "frames": frames,
            "variant": variant,
        }
        try:
            if variant == "hf_ref_compiled":
                fn = compile_tpu(
                    lambda qq, kk, vv, ee: relative_key_reference(qq, kk, vv, ee, left, right),
                    dynamic=False,
                    fullgraph=True,
                )
                stats = _time_tpu(lambda: fn(q, k, v, emb), warmup=args.warmup, runs=args.runs)
            elif variant.startswith("pallas_"):
                bq, bk = (int(x) for x in variant.split("_")[1].split("x"))
                fn = compile_tpu(
                    lambda qq, kk, vv, ee, ep: pallas_relative_key_attention(
                        qq, kk, vv, ee, left, right, tile_size=(bq, bk), padded_embedding=ep
                    ),
                    dynamic=False,
                    fullgraph=True,
                )
                stats = _time_tpu(
                    lambda: fn(q, k, v, emb, e_pad), warmup=args.warmup, runs=args.runs
                )
            else:
                fn = compile_tpu(
                    lambda qq, kk, vv, ee, dd, fi: native_relative_key_attention(
                        qq,
                        kk,
                        vv,
                        ee,
                        left,
                        right,
                        distance=dd,
                        flat_index=fi,
                        variant=variant,  # type: ignore[arg-type]
                    ),
                    dynamic=False,
                    fullgraph=True,
                )
                stats = _time_tpu(
                    lambda: fn(q, k, v, emb, dist, flat_idx), warmup=args.warmup, runs=args.runs
                )
            row.update(status="ok", **stats)
        except Exception as exc:
            row.update(status="error", note=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark relative-key attention on TPU v5e")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--frames", type=int, nargs="+", default=[250, 256, 500, 512, 1500, 1536])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "hf_ref_compiled",
            "explicit_gather",
            "explicit_index",
            "sdpa_gather",
            "sdpa_index",
            "pallas_128x128",
            "pallas_256x256",
            "pallas_512x256",
            "pallas_512x512",
        ],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--include-qkv", action="store_true")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--output-prefix", default="bench_attention")
    args = parser.parse_args()

    torch_tpu()

    configure_compiler(recompile_limit=128, fail_on_recompile_limit=True)
    register_pallas_ops()
    heads, head_dim, left, right = 16, 64, 64, 8
    rows: list[dict[str, Any]] = []

    for frames in args.frames:
        rows.extend(_bench_attention_frames(args, frames, heads, head_dim, left, right))
        gc.collect()
        torch.accelerator.empty_cache()

    if args.include_qkv:
        rows.extend(bench_qkv_projections(args.batch, args.frames, args.warmup, args.runs))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.output_prefix}.json").write_text(json.dumps(rows, indent=2) + "\n")
    fieldnames = [
        "kind",
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
