"""Pallas TPU custom kernels for w2v-bert-tpu."""

from __future__ import annotations

from w2vbert_tpu.kernels.fused_qkv_attention import fused_qkv_attention, register_fused_ops
from w2vbert_tpu.kernels.relative_attention import (
    DEFAULT_TILE_SIZE,
    pallas_relative_key_attention,
    register_pallas_ops,
)

__all__ = [
    "DEFAULT_TILE_SIZE",
    "fused_qkv_attention",
    "register_fused_ops",
    "pallas_relative_key_attention",
    "register_pallas_ops",
]
