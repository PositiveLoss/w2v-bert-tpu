"""Benchmark suite for w2v-BERT 2.0 inference on Google TPU v5e.

Supports running from both the Python CLI and Google Colab notebooks.
Records compilation time, warm p50/p95/p99 latency, throughput (audio hours/hour and files/s),
peak HBM usage, TorchDynamo compilation/graph-break stats, and analytical MFU.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

from w2vbert_tpu._tpu import torch_tpu
from w2vbert_tpu.batching import (
    DEFAULT_FRAME_BUCKETS,
    pad_to_bucket,
    select_frame_bucket,
    unpad_output,
)
from w2vbert_tpu.compiler import (
    compile_tpu,
    configure_compiler,
    get_compile_metrics,
    reset_compiler,
)
from w2vbert_tpu.model import Backend

# Historical reference measurements (TPU v5e, compiled HF Wav2Vec2BertModel, BF16).
HISTORICAL_HF_COMPILED_MS: dict[tuple[int, int], float] = {
    (5, 1): 4.578,
    (5, 4): 12.336,
    (5, 8): 28.114,
    (5, 16): 48.834,
    (10, 1): 7.517,
    (10, 4): 26.870,
    (10, 8): 61.857,
    (10, 16): 135.803,
    (30, 1): 51.034,
    (30, 4): 179.288,
    (30, 8): 360.660,
    (30, 16): 727.246,
}

HISTORICAL_HF_COMPILED_AUDIO_HPH: dict[tuple[int, int], float] = {
    (5, 1): 1092.25,
    (5, 4): 1621.22,
    (5, 8): 1422.79,
    (5, 16): 1638.21,
    (10, 1): 1330.23,
    (10, 4): 1488.64,
    (10, 8): 1293.30,
    (10, 16): 1178.17,
    (30, 1): 587.84,
    (30, 4): 669.32,
    (30, 8): 665.45,
    (30, 16): 660.02,
}

FRAMES_PER_SECOND = 50
OPTIMIZED_HBM_BUDGET_GIB = 14.5
DEFAULT_PEAK_BF16_FLOPS = 197e12


def analytical_model_flops(config: Any, batch: int, frames: int) -> float:
    """Consistent analytical FLOP count (2 FLOPs per MAC) for Wav2Vec2BertModel inference.

    Per token and Conformer layer:
    - 2 feed-forward blocks (Macaron FFN1 + FFN2): ``8 * d * intermediate``
    - Q, K, V, Out projections: ``8 * d * d``
    - Attention Q@K^T and P@V: ``4 * frames * d``
    - Relative-key bucket matmul Q@E^T: ``2 * (L + R + 1) * d``
    - Conformer pointwise1 (d -> 2d) and pointwise2 (d -> d): ``6 * d * d``
    - Conformer causal depthwise conv (kernel K): ``2 * d * K``
    Plus feature projection (``2 * input_dim * d``) per token.
    """
    d = int(config.hidden_size)
    f = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    buckets = int(config.left_max_position_embeddings + config.right_max_position_embeddings + 1)
    k = int(config.conv_depthwise_kernel_size)
    in_dim = int(config.feature_projection_input_dim)
    per_layer = 8 * d * f + 8 * d * d + 4 * frames * d + 2 * buckets * d + 6 * d * d + 2 * d * k
    per_token = per_layer * layers + 2 * in_dim * d
    return float(per_token * batch * frames)


def _pkg_ver(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _tpu_metadata() -> dict[str, Any]:
    tpu = torch_tpu()
    if not torch.accelerator.is_available():
        raise RuntimeError("No TPU accelerator is available")
    props = tpu.get_device_properties(0)
    return {
        "accelerator_name": props.name,
        "accelerator_version": str(props.version),
        "physical_chip_count": int(props.devices_per_chip),
        "hbm_bytes_per_chip": int(props.hbm_bytes_per_chip),
        "peak_bf16_flops": float(props.bf16_flops_per_chip or DEFAULT_PEAK_BF16_FLOPS),
        "frameworks": {
            "torch": torch.__version__,
            "torch_tpu": _pkg_ver("torch_tpu"),
            "libtpu": _pkg_ver("libtpu"),
            "transformers": _pkg_ver("transformers"),
            "jax": _pkg_ver("jax"),
        },
    }


def _estimate_safe_skip(variant: str, batch: int, frames: int, hbm_bytes: int) -> str | None:
    """Estimate whether dense attention tensors are guaranteed to exceed 15.75 GiB HBM."""
    # Dense [B, 16, T, T] scores in fp32 (softmax) + bf16 attn weights/bias + activations
    if variant in ("hf_eager", "hf_compiled", "native_compiled"):
        dense_bytes = batch * 16 * frames * frames * 4  # fp32 softmax scores [B, H, T, T]
        if variant.startswith("hf_"):
            # HF also builds [T, T, 64] positional embedding + multiple [B, 16, T, T] buffers
            dense_bytes = dense_bytes * 3 + frames * frames * 64 * 2
        if dense_bytes > 0.85 * hbm_bytes:
            return (
                f"Estimated dense attention buffer ({dense_bytes / 1024**3:.2f} GiB) "
                "exceeds safe 15.75 GiB HBM limit"
            )
    if variant in ("pallas_compiled", "optimized_auto"):
        # Linear fit of measured peak HBM (TPU v5e, BF16, extended sweep): ~1.3 GiB weights +
        # workspace plus ~0.1 GiB per 1000 batch*frames (120 s B=8: 6.0 GiB); 1.25x margin.
        estimate_gib = 1.3 + 1.25 * 0.1 * batch * frames / 1000.0
        if estimate_gib > OPTIMIZED_HBM_BUDGET_GIB:
            return (
                f"Estimated peak HBM {estimate_gib:.1f} GiB exceeds "
                f"{OPTIMIZED_HBM_BUDGET_GIB} GiB budget"
            )
    return None


def _build_variant_model(
    base_model: torch.nn.Module,
    variant: str,
    *,
    compile_strategy: str = "whole",
    recompile_limit: int = 128,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    configure_compiler(recompile_limit=recompile_limit, fail_on_recompile_limit=True)
    reset_compiler(clear_counters=True)

    model = copy.deepcopy(base_model).eval().to(device="tpu", dtype=torch.bfloat16)
    if variant == "hf_eager":
        return model, {"compiled": False, "strategy": "eager"}
    if variant == "hf_compiled":
        if compile_strategy == "layer":
            encoder: Any = getattr(model, "encoder")
            for layer in encoder.layers:
                layer.compile(backend="tpu", dynamic=False, fullgraph=True)
        else:
            model = compile_tpu(model, dynamic=False, fullgraph=True)
        return model, {"compiled": True, "strategy": compile_strategy}

    import w2vbert_tpu

    backend_map: dict[str, Backend] = {
        "native_compiled": "native",
        "pallas_compiled": "pallas",
        "optimized_auto": "auto",
    }
    if variant not in backend_map:
        raise ValueError(f"Unknown benchmark variant: {variant}")
    w2vbert_tpu.optimize(
        model,
        backend=backend_map[variant],
        compile=(compile_strategy == "layer"),
    )
    if compile_strategy == "whole":
        model = compile_tpu(model, dynamic=False, fullgraph=True)
    return model, {
        "compiled": True,
        "strategy": compile_strategy,
        "backend": backend_map[variant],
    }


def benchmark_configuration(
    model: torch.nn.Module,
    config: Any,
    *,
    variant: str,
    batch_size: int,
    seconds: int,
    frames: int,
    runs: int,
    warmup: int,
    peak_flops: float,
    compile_meta: dict[str, Any],
    use_bucketing: bool = False,
) -> dict[str, Any]:
    """Run timed inference for one (variant, batch_size, frames) configuration."""
    if use_bucketing:
        effective_frames = select_frame_bucket(frames, DEFAULT_FRAME_BUCKETS)
    else:
        effective_frames = frames
    x = torch.randn(
        batch_size,
        frames,
        int(config.feature_projection_input_dim),
        device="tpu",
        dtype=torch.bfloat16,
    )
    mask = torch.ones(batch_size, frames, device="tpu", dtype=torch.long)

    def _run_once():
        if use_bucketing and effective_frames != frames:
            bucketed = pad_to_bucket(x, mask, frame_buckets=DEFAULT_FRAME_BUCKETS)
            out = model(
                input_features=bucketed.input_features,
                attention_mask=bucketed.attention_mask,
            ).last_hidden_state
            return unpad_output(out, batch_size, frames)
        return model(input_features=x, attention_mask=mask).last_hidden_state

    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    counters_before = get_compile_metrics()

    with torch.inference_mode():
        t0 = time.perf_counter()
        out = _run_once()
        torch.accelerator.synchronize()
        first_run_s = time.perf_counter() - t0
        out_shape = list(out.shape)
        out_dtype = str(out.dtype)
        out_device = str(out.device)

        for _ in range(warmup):
            out = _run_once()
        torch.accelerator.synchronize()

        # Per-request synchronized latency
        latencies_s: list[float] = []
        for _ in range(runs):
            t_start = time.perf_counter()
            out = _run_once()
            torch.accelerator.synchronize()
            latencies_s.append(time.perf_counter() - t_start)

        # Queued throughput (async dispatch pipelined, single sync at end)
        t_q0 = time.perf_counter()
        queued_outs = []
        for _ in range(runs):
            queued_outs.append(_run_once())
        torch.accelerator.synchronize()
        queued_total_s = time.perf_counter() - t_q0
        del queued_outs

    counters_after = get_compile_metrics()
    peak_hbm_bytes = int(torch.accelerator.max_memory_allocated())

    lat_ms = [v * 1000.0 for v in latencies_s]
    p50_ms = float(np.percentile(lat_ms, 50))
    p95_ms = float(np.percentile(lat_ms, 95))
    p99_ms = float(np.percentile(lat_ms, 99))
    mean_ms = float(statistics.mean(lat_ms))
    p50_s = p50_ms / 1000.0
    queued_per_step_s = queued_total_s / runs

    flops = analytical_model_flops(config, batch_size, effective_frames)
    mfu = flops / p50_s / peak_flops
    queued_mfu = flops / queued_per_step_s / peak_flops

    hist_ms = HISTORICAL_HF_COMPILED_MS.get((seconds, batch_size))
    return {
        "variant": variant,
        "status": "ok",
        "audio_seconds": seconds,
        "frames": frames,
        "effective_frames": effective_frames,
        "batch_size": batch_size,
        "input_shape": [batch_size, frames, int(config.feature_projection_input_dim)],
        "output_shape": out_shape,
        "dtype": out_dtype,
        "device": out_device,
        "first_run_s": round(first_run_s, 4),
        "p50_ms": round(p50_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "p99_ms": round(p99_ms, 3),
        "mean_ms": round(mean_ms, 3),
        "queued_step_ms": round(queued_per_step_s * 1000.0, 3),
        "audio_hours_per_hour": round((batch_size * seconds) / p50_s, 2),
        "queued_audio_hours_per_hour": round((batch_size * seconds) / queued_per_step_s, 2),
        "files_per_second": round(batch_size / p50_s, 2),
        "peak_hbm_gib": round(peak_hbm_bytes / (1024**3), 3),
        "analytical_tflops_per_call": round(flops / 1e12, 4),
        "mfu_pct": round(mfu * 100.0, 2),
        "queued_mfu_pct": round(queued_mfu * 100.0, 2),
        "compile_count_delta": counters_after["compile_count"] - counters_before["compile_count"],
        "unique_graphs_delta": counters_after["unique_graphs"] - counters_before["unique_graphs"],
        "graph_breaks_delta": counters_after["graph_break_count"]
        - counters_before["graph_break_count"],
        "compile_meta": compile_meta,
        "historical_hf_compiled_ms": hist_ms,
    }


def _write_reports(
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
    out_dir: Path,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": results}
    json_path = out_dir / f"{prefix}.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    csv_path = out_dir / f"{prefix}.csv"
    fieldnames = [
        "variant",
        "status",
        "audio_seconds",
        "frames",
        "effective_frames",
        "batch_size",
        "first_run_s",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "queued_step_ms",
        "audio_hours_per_hour",
        "files_per_second",
        "peak_hbm_gib",
        "mfu_pct",
        "compile_count_delta",
        "graph_breaks_delta",
        "historical_hf_compiled_ms",
        "note",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    md_path = out_dir / f"{prefix}.md"
    lines = [
        f"# Benchmark Report (`{prefix}`)",
        "",
        f"- **Accelerator**: {metadata['accelerator_name']} "
        f"({metadata['physical_chip_count']} chip)",
        f"- **Model**: `{metadata['model_id']}` ({metadata['num_parameters']:,} params, "
        f"{metadata['num_layers']} layers)",
        f"- **Frameworks**: torch `{metadata['frameworks']['torch']}`, "
        f"torch_tpu `{metadata['frameworks']['torch_tpu']}`, "
        f"libtpu `{metadata['frameworks']['libtpu']}`, "
        f"transformers `{metadata['frameworks']['transformers']}`, "
        f"jax `{metadata['frameworks']['jax']}`",
        "",
        "| Variant | Status | Dur (s) | Frames | Batch | 1st Run (s) | p50 (ms) | p95 (ms) "
        "| p99 (ms) | Audio hr/hr | Peak HBM (GiB) | MFU (%) | Compiles | Graph Breaks |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['status']} | {r['audio_seconds']} | {r['frames']} | "
            f"{r['batch_size']} | {r.get('first_run_s', '-')} | {r.get('p50_ms', '-')} | "
            f"{r.get('p95_ms', '-')} | {r.get('p99_ms', '-')} | "
            f"{r.get('audio_hours_per_hour', '-')} | {r.get('peak_hbm_gib', '-')} | "
            f"{r.get('mfu_pct', '-')} | {r.get('compile_count_delta', '-')} | "
            f"{r.get('graph_breaks_delta', '-')} |"
        )
    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark w2v-BERT 2.0 inference on TPU v5e")
    parser.add_argument("--model-id", default="facebook/w2v-bert-2.0")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["hf_compiled"],
        choices=[
            "hf_eager",
            "hf_compiled",
            "native_compiled",
            "pallas_compiled",
            "optimized_auto",
        ],
    )
    parser.add_argument("--durations", type=int, nargs="+", default=[5, 10, 30])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile-strategy", choices=["whole", "layer"], default="whole")
    parser.add_argument("--recompile-limit", type=int, default=128)
    parser.add_argument("--bucketing", action="store_true")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--output-prefix", default="benchmark_model")
    args = parser.parse_args()

    from transformers import Wav2Vec2BertModel

    meta = _tpu_metadata()
    base = Wav2Vec2BertModel.from_pretrained(args.model_id, local_files_only=True)
    base.config.layerdrop = 0.0
    base.config.mask_time_prob = 0.0
    base.config.mask_feature_prob = 0.0
    base.eval()

    meta.update(
        {
            "model_id": args.model_id,
            "num_parameters": int(sum(p.numel() for p in base.parameters())),
            "num_layers": int(len(base.encoder.layers)),
            "hidden_size": int(base.config.hidden_size),
            "num_attention_heads": int(base.config.num_attention_heads),
            "intermediate_size": int(base.config.intermediate_size),
            "runs": args.runs,
            "warmup": args.warmup,
            "compile_strategy": args.compile_strategy,
            "recompile_limit": args.recompile_limit,
        }
    )

    all_results: list[dict[str, Any]] = []
    for variant in args.variants:
        model, compile_meta = _build_variant_model(
            base,
            variant,
            compile_strategy=args.compile_strategy,
            recompile_limit=args.recompile_limit,
        )
        for seconds in args.durations:
            frames = seconds * FRAMES_PER_SECOND
            for batch in args.batches:
                skip_reason = _estimate_safe_skip(
                    variant, batch, frames, meta["hbm_bytes_per_chip"]
                )
                if skip_reason is not None:
                    row = {
                        "variant": variant,
                        "status": "skipped",
                        "audio_seconds": seconds,
                        "frames": frames,
                        "effective_frames": frames,
                        "batch_size": batch,
                        "note": skip_reason,
                    }
                    all_results.append(row)
                    print(json.dumps(row), flush=True)
                    continue
                try:
                    row = benchmark_configuration(
                        model,
                        base.config,
                        variant=variant,
                        batch_size=batch,
                        seconds=seconds,
                        frames=frames,
                        runs=args.runs,
                        warmup=args.warmup,
                        peak_flops=meta["peak_bf16_flops"],
                        compile_meta=compile_meta,
                        use_bucketing=args.bucketing,
                    )
                except Exception as exc:
                    row = {
                        "variant": variant,
                        "status": "error",
                        "audio_seconds": seconds,
                        "frames": frames,
                        "effective_frames": frames,
                        "batch_size": batch,
                        "note": f"{type(exc).__name__}: {exc}",
                    }
                all_results.append(row)
                print(json.dumps(row), flush=True)
        del model
        gc.collect()
        reset_compiler(clear_counters=True)
        torch.accelerator.empty_cache()

    _write_reports(all_results, meta, Path(args.output_dir), args.output_prefix)


if __name__ == "__main__":
    main()
