"""Access to TorchTPU without a static dependency on it."""

from __future__ import annotations

import importlib
from typing import Any

import torch


def torch_tpu() -> Any:
    """Import TorchTPU and return the ``torch.tpu`` namespace it registers.

    TorchTPU is not installable from PyPI and adds ``torch.tpu`` only when imported, so type
    checkers can see neither. Code that needs the TPU goes through this function instead.
    """
    importlib.import_module("torch_tpu")
    return getattr(torch, "tpu")
