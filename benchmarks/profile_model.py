"""Profile compiled w2v-BERT 2.0 inference on TPU v5e and write a compact op-time summary.

Uses ``torch.profiler`` with ``ProfilerActivity.TPU`` (TorchTPU Kineto backend) and summarizes
``key_averages()`` device time by op and by op category. Raw XPlane/Chrome traces are written to
a temporary directory and deleted afterwards (they must never be committed).

Works with the base model and with fine-tuned models such as ``Wav2Vec2BertForCTC``: the model
class comes from the checkpoint's ``config.json``. Compiled ops can't be traced back to the
module they came from, so for a model with a head the script also profiles the encoder alone
(adapter and head removed) and reports the difference as the adapter + head share of the
program time. ``--no-encoder-only`` skips that second run.

Examples::

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 JAX_PLATFORMS=cpu \\
        python benchmarks/profile_model.py --variants hf_compiled optimized_auto \\
        --configs 5:1 10:16 30:16

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 JAX_PLATFORMS=cpu \\
        python benchmarks/profile_model.py --model-id speech-uk/w2v-bert-v3 \\
        --variants optimized_auto --configs 5:1 30:16 120:16 --output-prefix profile_ctc
"""

from __future__ import annotations

import argparse
import copy
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
from w2vbert_tpu.model import _model_class_for

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


def _encoder_only(model: torch.nn.Module) -> torch.nn.Module | None:
    """Copy of the base ``Wav2Vec2BertModel`` inside ``model`` without its adapter.

    Returns ``None`` when ``model`` is already a plain encoder model (nothing to subtract).
    """
    inner = getattr(model, "wav2vec2_bert", None)
    if inner is None:
        return None
    encoder_only: Any = copy.deepcopy(inner)
    encoder_only.adapter = None
    encoder_only.intermediate_ffn = None
    return encoder_only


def _head_share_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair full-model and encoder-only rows and compute the adapter + head share."""
    encoder_ms = {
        (r["variant"], r["audio_seconds"], r["batch_size"]): r["program_ms_per_step"]
        for r in rows
        if r["part"] == "encoder_only"
    }
    shares = []
    for r in rows:
        key = (r["variant"], r["audio_seconds"], r["batch_size"])
        if r["part"] != "full" or key not in encoder_ms:
            continue
        full, enc = r["program_ms_per_step"], encoder_ms[key]
        shares.append(
            {
                "variant": r["variant"],
                "audio_seconds": r["audio_seconds"],
                "batch_size": r["batch_size"],
                "full_ms": full,
                "encoder_only_ms": enc,
                "adapter_head_ms": round(full - enc, 3),
                "adapter_head_pct": round(100.0 * (full - enc) / full, 2) if full else 0.0,
            }
        )
    return shares


def _write_markdown(rows: list[dict[str, Any]], path: Path, model_id: str) -> None:
    lines = [
        "# TPU v5e profile summary (`torch.profiler`, `ProfilerActivity.TPU`)",
        "",
        "Per forward step, from `key_averages()`. *Program* is the compiled executable's device",
        "span. Synchronous HLO ops are grouped by instruction prefix (`convolution` = MXU",
        "matmuls, `custom-call` = Pallas kernel, `fusion` = XLA elementwise/reduction fusions).",
        "Async `*-start`/`*-done` copies (weight prefetch) overlap compute and are listed apart.",
        "",
        f"Model: `{model_id}` ({rows[0]['model_class'] if rows else '?'}).",
        "",
    ]
    shares = _head_share_rows(rows)
    if shares:
        lines += [
            "## Adapter + head share",
            "",
            "Program time of the full model minus the same model with the adapter and head",
            "removed (encoder only). Differences of a few percent are within run-to-run noise.",
            "",
            "| Variant | Audio | B | Full ms | Encoder-only ms | Adapter + head ms | % |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for s in shares:
            lines.append(
                f"| {s['variant']} | {s['audio_seconds']} s | {s['batch_size']} | {s['full_ms']} | "
                f"{s['encoder_only_ms']} | {s['adapter_head_ms']} | {s['adapter_head_pct']} |"
            )
        lines.append("")
    for r in rows:
        part = " (encoder only)" if r["part"] == "encoder_only" else ""
        lines += [
            f"## {r['variant']}{part}: {r['audio_seconds']} s (T={r['frames']}), "
            f"B={r['batch_size']}",
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
    parser.add_argument(
        "--no-encoder-only",
        action="store_true",
        help="For models with a head (e.g. CTC), skip the encoder-only comparison run",
    )
    args = parser.parse_args()

    torch_tpu()
    from bench_model import _build_variant_model  # type: ignore[import-not-found]

    model_class: Any = _model_class_for(args.model_id, {"local_files_only": True})
    base = model_class.from_pretrained(args.model_id, local_files_only=True).eval()
    print(f"Model: {args.model_id} ({model_class.__name__})", flush=True)
    configs = [tuple(int(v) for v in c.split(":")) for c in args.configs]

    parts: list[tuple[str, torch.nn.Module]] = [("full", base)]
    encoder_only = None if args.no_encoder_only else _encoder_only(base)
    if encoder_only is not None:
        parts.append(("encoder_only", encoder_only))

    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        for part, source in parts:
            model, _ = _build_variant_model(source, variant)
            for seconds, batch in configs:
                row: dict[str, Any] = {
                    "variant": variant,
                    "part": part,
                    "model_class": model_class.__name__,
                }
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
                print(json.dumps({k: row[k] for k in row if k != "top_ops"}), flush=True)
            del model
            gc.collect()
            reset_compiler(clear_counters=True)
            torch.accelerator.empty_cache()

    for share in _head_share_rows(rows):
        print(json.dumps({"adapter_head_share": share}), flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.output_prefix}.json").write_text(json.dumps(rows, indent=2) + "\n")
    _write_markdown(rows, out_dir / f"{args.output_prefix}.md", args.model_id)


if __name__ == "__main__":
    main()
