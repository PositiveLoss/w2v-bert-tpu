"""Inspect Python, TorchTPU, JAX/Pallas, system resources, and reference repositories."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

PACKAGES = (
    "torch",
    "torch_tpu",
    "libtpu",
    "transformers",
    "jax",
    "jaxlib",
    "tokamax",
    "xprof",
    "tensorboard",
    "soundfile",
    "torchaudio",
    "librosa",
    "scipy",
    "pytest",
    "ruff",
)

REFERENCE_REPOS = (
    Path("/content/w2v-bert-fast"),
    Path("/content/torch_tpu"),
    Path("/content/jax"),
    Path("/content/tokamax"),
)


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_rev(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except subprocess.SubprocessError:
        return None


def collect_environment() -> dict:
    import psutil
    import torch

    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ld_preload": os.environ.get("LD_PRELOAD", ""),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
        "packages": {pkg: _pkg_version(pkg) for pkg in PACKAGES},
        "system": {
            "cpu_count": os.cpu_count(),
            "ram_total_gib": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_available_gib": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_free_gib": round(shutil.disk_usage("/content").free / (1024**3), 2),
        },
        "reference_revisions": {str(p): _git_rev(p) for p in REFERENCE_REPOS},
    }

    try:
        import torch_tpu  # noqa: F401

        tpu_available = torch.accelerator.is_available()
        device_count = torch.accelerator.device_count() if tpu_available else 0
        tpu_info: dict = {
            "available": tpu_available,
            "current_accelerator": str(torch.accelerator.current_accelerator())
            if tpu_available
            else None,
            "device_count": device_count,
            "pallas_jax_op": hasattr(getattr(torch.tpu, "pallas", None), "jax_op"),
        }
        if tpu_available:
            props = torch.tpu.get_device_properties(0)
            free_bytes, total_bytes = torch.accelerator.get_memory_info()
            tpu_info.update(
                {
                    "name": props.name,
                    "version": str(props.version),
                    "devices_per_chip": props.devices_per_chip,
                    "hbm_bytes_per_chip": props.hbm_bytes_per_chip,
                    "hbm_gib_total_reported": round(total_bytes / (1024**3), 2),
                    "hbm_gib_free_reported": round(free_bytes / (1024**3), 2),
                    "bf16_tflops_per_chip": props.bf16_flops_per_chip / 1e12,
                    "sparse_cores_per_device": props.sparse_cores_per_device,
                }
            )
        info["tpu"] = tpu_info
    except Exception as exc:
        info["tpu"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    return info


def main() -> None:
    print(json.dumps(collect_environment(), indent=2))


if __name__ == "__main__":
    main()
