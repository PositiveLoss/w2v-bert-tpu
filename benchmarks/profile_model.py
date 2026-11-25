"""Profile compiled w2v-BERT 2.0 inference on TPU v5e and write a compact op-time summary.

Uses ``torch.profiler`` with ``ProfilerActivity.TPU`` (TorchTPU Kineto backend) and summarizes
``key_averages()`` device time by op and by op category. Raw XPlane/Chrome traces are written to
a temporary directory and deleted afterwards (they must never be committed).

Example::

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 JAX_PLATFORMS=cpu \\
        python benchmarks/profile_model.py --variants hf_compiled optimized_auto \\
        --configs 5:1 10:16 30:16
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch

from w2vbert_tpu._tpu import torch_tpu
from w2vbert_tpu.compiler import reset_compiler

FRAMES_PER_SECOND = 50

_HLO_NAME = re.compile(r"^%([A-Za-z_\-]+)")
_PROGRAM = re.compile(r"_fwd\(|^jit_|^SyncTensorsGraph")


def _classify(name: str) -> tuple[str, str]:
    """Return ``(kind, group)`` for a device event.

    ``kind`` is ``"program"`` (whole compiled executable), ``"async"`` (overlapped ``*-start`` /
    ``*-done`` copies whose spans must not be summed), or ``"op"``. ``group`` is the HLO
    instruction prefix (``fusion``, ``convolution`` = MXU matmul/conv, ``custom-call`` = Pallas,
    ``copy``, ...).
    """
    if _PROGRAM.search(name):
        return "program", "program"
    match = _HLO_NAME.match(name)
    group = match.group(1) if match else "other"
    if group.endswith("-start") or group.endswith("-done"):
        return "async", group
    return "op", group


def _device_time_us(evt: Any) -> float:
    for attr in ("self_device_time_total", "device_time_total"):
        value = getattr(evt, attr, None)
        if value:
            return float(value)
    return 0.0


def profile_configuration(
    model: torch.nn.Module,
    config: Any,
    *,
    batch: int,
    seconds: int,
    steps: int,
    warmup: int,
    top: int,
) -> dict[str, Any]:
    frames = seconds * FRAMES_PER_SECOND
    x = torch.randn(
        batch, frames, int(config.feature_projection_input_dim), device="tpu", dtype=torch.bfloat16
    )
    mask = torch.ones(batch, frames, device="tpu", dtype=torch.long)

    with torch.inference_mode():
        for _ in range(warmup):
            model(input_features=x, attention_mask=mask)
        torch.accelerator.synchronize()

        trace_dir = Path(tempfile.mkdtemp(prefix="w2vbert_profile_"))
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.TPU,  # type: ignore[attr-defined]
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
            ) as prof:
                for _ in range(steps):
                    model(input_features=x, attention_mask=mask)
                    torch.accelerator.synchronize()
                    prof.step()
            averages = prof.key_averages()
        finally:
            shutil.rmtree(trace_dir, ignore_errors=True)

    program_us = 0.0
    async_us: dict[str, float] = {}
    ops = []
    for evt in averages:
        dev_us = _device_time_us(evt) / steps
        if dev_us <= 0:
            continue
        kind, group = _classify(evt.key)
        if kind == "program":
            program_us += dev_us
        elif kind == "async":
            async_us[group] = async_us.get(group, 0.0) + dev_us
        else:
            ops.append(
                {
                    "name": evt.key,
                    "group": group,
                    "count": int(evt.count) // steps,
                    "device_us_per_step": round(dev_us, 2),
                }
            )
    ops.sort(key=lambda r: r["device_us_per_step"], reverse=True)
    op_total = sum(r["device_us_per_step"] for r in ops)
    groups: dict[str, float] = {}
    for r in ops:
        groups[r["group"]] = groups.get(r["group"], 0.0) + r["device_us_per_step"]
        r["pct_of_ops"] = round(100.0 * r["device_us_per_step"] / op_total, 2) if op_total else 0.0

    ordered = sorted(groups.items(), key=lambda kv: -kv[1])
    return {
        "audio_seconds": seconds,
        "frames": frames,
        "batch_size": batch,
        "steps": steps,
        "program_ms_per_step": round(program_us / 1000.0, 3),
        "sync_op_ms_per_step": round(op_total / 1000.0, 3),
        "num_sync_ops": len(ops),
        "groups_ms": {k: round(v / 1000.0, 3) for k, v in ordered},
        "groups_pct": {k: round(100.0 * v / op_total, 2) if op_total else 0.0 for k, v in ordered},
        "async_overlapped_ms": {k: round(v / 1000.0, 3) for k, v in async_us.items()},
        "top_ops": ops[:top],
    }


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# TPU v5e profile summary (`torch.profiler`, `ProfilerActivity.TPU`)",
        "",
        "Per forward step, from `key_averages()`. *Program* is the compiled executable's device",
        "span. Synchronous HLO ops are grouped by instruction prefix (`convolution` = MXU",
        "matmuls, `custom-call` = Pallas kernel, `fusion` = XLA elementwise/reduction fusions).",
        "Async `*-start`/`*-done` copies (weight prefetch) overlap compute and are listed apart.",
        "",
    ]
    for r in rows:
        lines += [
            f"## {r['variant']}: {r['audio_seconds']} s (T={r['frames']}), B={r['batch_size']}",
            "",
            f"Program: **{r['program_ms_per_step']} ms/step**; synchronous ops "
            f"{r['sync_op_ms_per_step']} ms ({r['num_sync_ops']} distinct).",
            "",
            "| HLO group | ms/step | % of sync ops |",
            "|---|---:|---:|",
        ]
        for grp, ms in r["groups_ms"].items():
            lines.append(f"| `{grp}` | {ms} | {r['groups_pct'][grp]} |")
        if r["async_overlapped_ms"]:
            overlapped = ", ".join(f"`{k}` {v} ms" for k, v in r["async_overlapped_ms"].items())
            lines += ["", f"Overlapped async spans: {overlapped}."]
        lines += ["", "| Top op | Calls | µs/step | % |", "|---|---:|---:|---:|"]
        for op in r["top_ops"][:10]:
            name = op["name"].split(" = ")[0].replace("|", "\\|")
            detail = op["name"].split(" = ", 1)[-1]
            shape = detail.split("{")[0][:40].replace("|", "\\|")
            lines.append(
                f"| `{name}` {shape} | {op['count']} | {op['device_us_per_step']} | "
                f"{op['pct_of_ops']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile w2v-BERT 2.0 inference on TPU v5e")
    parser.add_argument("--model-id", default="facebook/w2v-bert-2.0")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["hf_compiled", "optimized_auto"],
        choices=["hf_compiled", "native_compiled", "pallas_compiled", "optimized_auto"],
    )
    parser.add_argument("--configs", nargs="+", default=["5:1", "10:16", "30:16"])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--output-prefix", default="profile_summary")
    args = parser.parse_args()

    torch_tpu()
    from bench_model import _build_variant_model  # type: ignore[import-not-found]
    from transformers import Wav2Vec2BertModel

    base = Wav2Vec2BertModel.from_pretrained(args.model_id, local_files_only=True).eval()
    configs = [tuple(int(v) for v in c.split(":")) for c in args.configs]

    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        model, _ = _build_variant_model(base, variant)
        for seconds, batch in configs:
            row = {"variant": variant}
            row.update(
                profile_configuration(
                    model,
                    base.config,
                    batch=batch,
                    seconds=seconds,
                    steps=args.steps,
                    warmup=args.warmup,
                    top=args.top,
                )
            )
            rows.append(row)
            print(
                json.dumps({k: row[k] for k in row if k != "top_ops"}),
                flush=True,
            )
        del model
        gc.collect()
        reset_compiler(clear_counters=True)
        torch.accelerator.empty_cache()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.output_prefix}.json").write_text(json.dumps(rows, indent=2) + "\n")
    _write_markdown(rows, out_dir / f"{args.output_prefix}.md")


if __name__ == "__main__":
    main()
