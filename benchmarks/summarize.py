"""Build ``benchmark_summary.{json,csv,md}`` from the committed benchmark result files.

Compares the optimized default (``optimized_auto``) against the compiled Hugging Face baseline
measured on the same TPU v5e, and folds in the bucketed, compile-strategy and extended sweeps.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent / "results"


def _load(name: str) -> list[dict[str, Any]]:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["results"]


def _key(r: dict[str, Any]) -> tuple[int, int]:
    return int(r["audio_seconds"]), int(r["batch_size"])


def main() -> None:
    hf_rows = _load("baseline_hf") + _load("baseline_hf_2s")
    hf = {_key(r): r for r in hf_rows if r["status"] == "ok"}
    hf_eager = {_key(r): r for r in _load("baseline_hf_eager") if r["status"] == "ok"}
    matrix = _load("benchmark_optimized")
    # The 5 s B>=8 rows were re-measured after the batch-aware auto crossover was introduced.
    auto = {_key(r): r for r in matrix if r["variant"] == "optimized_auto"}
    auto.update({_key(r): r for r in _load("benchmark_optimized_auto_5s")})
    auto.update({_key(r): r for r in _load("benchmark_optimized_2s") if r["status"] == "ok"})
    native = {_key(r): r for r in matrix if r["variant"] == "native_compiled"}
    pallas = {_key(r): r for r in matrix if r["variant"] == "pallas_compiled"}
    bucketed = {_key(r): r for r in _load("benchmark_bucketed") if r["status"] == "ok"}
    layer = {_key(r): r for r in _load("benchmark_layer_compile") if r["status"] == "ok"}
    merged: dict[tuple[int, int], dict[str, Any]] = {}
    for idx in range(1, 6):
        merged.update({_key(r): r for r in _load(f"benchmark_extended_part{idx}")})
    # 50-run re-measurement of 2 s (the 10-run sweep's first configuration was an outlier).
    merged.update({_key(r): r for r in _load("benchmark_optimized_2s") if r["status"] == "ok"})
    # Matrix rows (50 runs) fill the 5/10/30 s cells the reduced sweep did not repeat.
    for key, r in auto.items():
        merged.setdefault(key, r)
    # Shapes only measured in the sweep (e.g. 2 s) join the HF comparison when HF was measured.
    for key, r in merged.items():
        if r.get("status") == "ok":
            auto.setdefault(key, r)
    extended = [merged[k] for k in sorted(merged)]
    bucketed.update({_key(r): r for r in _load("benchmark_bucketed_2s") if r["status"] == "ok"})

    # Current default: Pallas kernel reading the packed QKV projection in place (fused_qkv).
    # Everything above is the pre-fused default, kept as an ablation column.
    pre_fused = auto
    auto = {_key(r): r for r in _load("benchmark_fused_qkv") if r["status"] == "ok"}
    auto.update({_key(r): r for r in _load("benchmark_fused_qkv_short") if r["status"] == "ok"})
    bucketed.update(
        {_key(r): r for r in _load("benchmark_fused_qkv_bucketed") if r["status"] == "ok"}
    )

    rows: list[dict[str, Any]] = []
    for key in sorted(hf):
        a, h = auto.get(key), hf[key]
        if a is None or a.get("status") != "ok":
            continue
        rows.append(
            {
                "audio_seconds": key[0],
                "batch_size": key[1],
                "hf_eager_p50_ms": hf_eager.get(key, {}).get("p50_ms"),
                "hf_compiled_p50_ms": h["p50_ms"],
                "hf_compiled_mfu_pct": h["mfu_pct"],
                "hf_compiled_peak_hbm_gib": h["peak_hbm_gib"],
                "native_p50_ms": native.get(key, {}).get("p50_ms"),
                "pallas_p50_ms": pallas.get(key, {}).get("p50_ms"),
                "optimized_p50_ms": a["p50_ms"],
                "optimized_p95_ms": a["p95_ms"],
                "optimized_mfu_pct": a["mfu_pct"],
                "optimized_peak_hbm_gib": a["peak_hbm_gib"],
                "optimized_audio_hours_per_hour": a["audio_hours_per_hour"],
                "speedup_vs_hf_compiled": round(h["p50_ms"] / a["p50_ms"], 3),
                "pre_fused_p50_ms": pre_fused.get(key, {}).get("p50_ms"),
                "bucketed_p50_ms": bucketed.get(key, {}).get("p50_ms"),
                "layer_compile_p50_ms": layer.get(key, {}).get("p50_ms"),
                "compiles": a["compile_count_delta"],
                "graph_breaks": a["graph_breaks_delta"],
            }
        )

    summary = {"comparison": rows, "extended_sweep": extended}
    (RESULTS / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        with (RESULTS / "benchmark_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Benchmark summary — TPU v5e, BF16, `facebook/w2v-bert-2.0` (24 layers)",
        "",
        "p50 latency per forward call (synchronized), whole-model `torch.compile(backend='tpu',",
        "fullgraph=True)`, 50 runs after 5 warmups. Optimized = `w2vbert_tpu.optimize()` defaults",
        "(`backend='auto'`). MFU uses the analytical FLOP count in `bench_model.py`.",
        "",
        "| Audio | B | HF eager | HF compiled | Optimized | Speedup | MFU HF → opt | "
        "HBM HF → opt (GiB) | Compiles / breaks |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        eager = r["hf_eager_p50_ms"] if r["hf_eager_p50_ms"] is not None else "–"
        lines.append(
            f"| {r['audio_seconds']} s | {r['batch_size']} | {eager} | "
            f"{r['hf_compiled_p50_ms']} | **{r['optimized_p50_ms']}** | "
            f"**{r['speedup_vs_hf_compiled']:.2f}×** | "
            f"{r['hf_compiled_mfu_pct']}% → {r['optimized_mfu_pct']}% | "
            f"{r['hf_compiled_peak_hbm_gib']} → {r['optimized_peak_hbm_gib']} | "
            f"{r['compiles']} / {r['graph_breaks']} |"
        )
    lines += [
        "",
        "## Backend ablation and deployment options (p50 ms)",
        "",
        "Pre-fused columns use the earlier `[B, H, T, D]` kernel path (transposes around it).",
        "",
        "| Audio | B | native only | Pallas only (pre-fused) | pre-fused auto | auto (fused QKV) "
        "| bucketed auto | per-layer compile (pre-fused) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        cells = [
            r["native_p50_ms"],
            r["pallas_p50_ms"],
            r["pre_fused_p50_ms"],
            r["optimized_p50_ms"],
            r["bucketed_p50_ms"],
            r["layer_compile_p50_ms"],
        ]
        cells_txt = " | ".join("–" if c is None else str(c) for c in cells)
        lines.append(f"| {r['audio_seconds']} s | {r['batch_size']} | {cells_txt} |")
    unbucketed_60 = {_key(r): r for r in _load("benchmark_unbucketed_60s") if r["status"] == "ok"}
    for key in (k for k in sorted(bucketed) if k not in hf):
        plain = unbucketed_60.get(key, {}).get("p50_ms", "–")
        lines.append(
            f"| {key[0]} s | {key[1]} | – | – | {plain} | – | {bucketed[key]['p50_ms']} | – |"
        )

    if extended:
        durations = sorted({int(r["audio_seconds"]) for r in extended})
        batches = sorted({int(r["batch_size"]) for r in extended})
        lines += [
            "",
            "## Extended sweep (pre-fused default; p50 ms / audio-hours per hour)",
            "",
            "Measured before the fused-QKV kernel; current-default latencies are lower (above).",
            "",
            "10 runs after 2 warmups; 5/10/30 s cells not repeated in the reduced sweep come",
            "from the 50-run matrix. –: skipped (estimated peak HBM above 14.5 GiB) or not run.",
            "",
            "| Audio | " + " | ".join(f"B={b}" for b in batches) + " |",
            "|---|" + "---:|" * len(batches),
        ]
        by_key = {_key(r): r for r in extended}
        for d in durations:
            cells = []
            for b in batches:
                r = by_key.get((d, b))
                if r is None:
                    cells.append("–")
                elif r["status"] == "ok":
                    cells.append(f"{r['p50_ms']} / {round(r['audio_hours_per_hour'])}")
                else:
                    cells.append(r["status"])
            lines.append(f"| {d} s | " + " | ".join(cells) + " |")
    (RESULTS / "benchmark_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
