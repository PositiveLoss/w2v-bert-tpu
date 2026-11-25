"""TorchDynamo and TorchTPU compilation configuration, cache management, and diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch._dynamo

DEFAULT_RECOMPILE_LIMIT = 128
DEFAULT_ACCUMULATED_RECOMPILE_LIMIT = 1024
DEFAULT_TIER2_CACHE_NAME = "w2vbert_tpu_tier2"
DEFAULT_TIER3_CACHE_DIR = "/tmp/w2vbert_tpu_tier3_cache"


def configure_compiler(
    *,
    recompile_limit: int = DEFAULT_RECOMPILE_LIMIT,
    accumulated_recompile_limit: int = DEFAULT_ACCUMULATED_RECOMPILE_LIMIT,
    fail_on_recompile_limit: bool = True,
    suppress_errors: bool = False,
    enable_cache: bool = True,
    tier2_name: str = DEFAULT_TIER2_CACHE_NAME,
    tier3_dir: str = DEFAULT_TIER3_CACHE_DIR,
) -> dict[str, Any]:
    """Configure TorchDynamo and TorchTPU caching for predictable static-shape compilation.

    Every encoder layer of ``Wav2Vec2BertModel`` shares the same Python forward code object when
    compiled layer-wise, and multiple sequence/batch buckets share the code object when compiled
    whole-model. Raising ``recompile_limit`` and ``accumulated_recompile_limit`` prevents silent
    eager fallbacks after the default 8 recompilations.
    """
    if recompile_limit < 1:
        raise ValueError(f"recompile_limit must be >= 1, got {recompile_limit}")
    if accumulated_recompile_limit < recompile_limit:
        raise ValueError(
            "accumulated_recompile_limit must be >= recompile_limit, got "
            f"{accumulated_recompile_limit} < {recompile_limit}"
        )

    config = torch._dynamo.config
    config.recompile_limit = recompile_limit
    if hasattr(config, "cache_size_limit"):
        config.cache_size_limit = recompile_limit
    if hasattr(config, "accumulated_recompile_limit"):
        config.accumulated_recompile_limit = accumulated_recompile_limit
    if hasattr(config, "suppress_errors"):
        config.suppress_errors = suppress_errors
    if hasattr(config, "fail_on_recompile_limit_hit"):
        config.fail_on_recompile_limit_hit = fail_on_recompile_limit
    if hasattr(torch.compiler.config, "fail_on_recompile_limit_hit"):
        torch.compiler.config.fail_on_recompile_limit_hit = fail_on_recompile_limit

    cache_info: dict[str, str | bool] = {"enabled": enable_cache}
    if enable_cache:
        tier2_path = Path("/dev/shm/torch_tpu_cache") / tier2_name
        tier3_path = Path(tier3_dir)
        tier2_path.mkdir(parents=True, exist_ok=True)
        tier3_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_TPU_TIER2_COMPILATION_CACHE", tier2_name)
        os.environ.setdefault("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", str(tier3_path))
        cache_info["tier2_name"] = os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"]
        cache_info["tier3_root"] = os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"]

    return {
        "recompile_limit": int(config.recompile_limit),
        "accumulated_recompile_limit": int(getattr(config, "accumulated_recompile_limit", -1)),
        "fail_on_recompile_limit_hit": bool(getattr(config, "fail_on_recompile_limit_hit", False)),
        "suppress_errors": bool(getattr(config, "suppress_errors", False)),
        "cache": cache_info,
    }


def reset_compiler(*, clear_counters: bool = True) -> None:
    """Reset TorchDynamo/torch.compile state and optionally clear Dynamo diagnostic counters."""
    torch.compiler.reset()
    torch._dynamo.reset()
    if clear_counters:
        torch._dynamo.utils.counters.clear()


def get_compile_metrics() -> dict[str, Any]:
    """Return a JSON-serializable summary of Dynamo compile, graph-break and fallback stats."""
    counters = torch._dynamo.utils.counters
    frames = dict(counters.get("frames", {}))
    graph_breaks = dict(counters.get("graph_break", {}))
    stats = dict(counters.get("stats", {}))
    unimplemented = dict(counters.get("unimplemented", {}))
    unique_graphs = int(stats.get("unique_graphs", 0))
    frames_total = int(frames.get("total", 0))
    return {
        "frames_total": frames_total,
        "frames_ok": int(frames.get("ok", 0)),
        "unique_graphs": unique_graphs,
        "calls_captured": int(stats.get("calls_captured", 0)),
        "compile_count": max(frames_total, unique_graphs),
        "graph_break_count": int(sum(graph_breaks.values())),
        "graph_breaks": {str(k): int(v) for k, v in graph_breaks.items()},
        "unimplemented": {str(k): int(v) for k, v in unimplemented.items()},
        "stats": {str(k): int(v) for k, v in stats.items()},
    }


def compile_tpu(
    module: Any,
    *,
    dynamic: bool = False,
    fullgraph: bool = True,
    options: dict[str, Any] | None = None,
) -> Any:
    """Compile a module or function with TorchTPU (``backend='tpu'``)."""
    compile_kwargs: dict[str, Any] = {
        "backend": "tpu",
        "dynamic": dynamic,
        "fullgraph": fullgraph,
    }
    if options is not None:
        compile_kwargs["options"] = options
    return torch.compile(module, **compile_kwargs)


__all__ = [
    "DEFAULT_ACCUMULATED_RECOMPILE_LIMIT",
    "DEFAULT_RECOMPILE_LIMIT",
    "compile_tpu",
    "configure_compiler",
    "get_compile_metrics",
    "reset_compiler",
]
