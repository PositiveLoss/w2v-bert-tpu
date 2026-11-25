"""Channels-last causal depthwise convolution (Conformer, kernel 31) as a Pallas TPU kernel.

Port of w2v-bert-fast's channels-last Triton convolution: ``x`` stays ``[B, T, C]`` (no
transposes) and ``out[b, t, c] = sum_k w[c, k] * x[b, t - (K - 1) + k, c]`` with zeros before
``t = 0``.

Grid ``(B, T / block_t, C / block_c)``. Each step reads its ``[block_t, block_c]`` tile and the
32-row halo block immediately before it (a second BlockSpec on the same array, clamped to block 0
and zeroed for the first time tile), then accumulates ``K`` sublane-rolled multiply-adds in FP32.
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from torch import Tensor  # noqa: E402

from w2vbert_tpu._tpu import torch_tpu  # noqa: E402

HALO = 32


def _conv_kernel(x_ref, halo_ref, w_ref, o_ref, *, kernel_size: int):
    block_t = x_ref.shape[1]
    t_idx = pl.program_id(1)
    cur = x_ref[0].astype(jnp.float32)  # [block_t, block_c]
    halo = halo_ref[0].astype(jnp.float32)  # [HALO, block_c], rows t0-32 .. t0-1
    halo = jnp.where(t_idx == 0, jnp.zeros_like(halo), halo)
    window = jnp.concatenate([halo, cur], axis=0)  # [HALO + block_t, block_c]
    rows = HALO + block_t
    w = w_ref[...].astype(jnp.float32)  # [K_pad, block_c]
    acc = jnp.zeros_like(cur)
    for k in range(kernel_size):
        # Output row t needs window row t + HALO - (K - 1) + k.
        start = HALO - (kernel_size - 1) + k
        shifted = window if start == 0 else pltpu.roll(window, rows - start, 0)
        acc = acc + shifted[:block_t] * w[k : k + 1, :]
    o_ref[0] = acc.astype(o_ref.dtype)


def causal_depthwise_conv_jax(
    x: jax.Array,
    w_kc: jax.Array,
    kernel_size: int = 31,
    block_t: int = 256,
    block_c: int = 512,
    interpret: bool = False,
) -> jax.Array:
    """``x``: ``[B, T, C]`` (``T % block_t == 0``); ``w_kc``: ``[K_pad, C]`` weights."""
    batch, frames, channels = x.shape
    if kernel_size - 1 > HALO:
        raise ValueError(f"kernel_size must be <= {HALO + 1}")
    halo_blocks_per_tile = block_t // HALO
    kernel = jax.tree_util.Partial(_conv_kernel, kernel_size=kernel_size)
    return pl.pallas_call(
        kernel,
        grid=(batch, frames // block_t, channels // block_c),
        in_specs=[
            pl.BlockSpec((1, block_t, block_c), lambda b, t, c: (b, t, c)),
            pl.BlockSpec(
                (1, HALO, block_c),
                lambda b, t, c: (b, jnp.maximum(t * halo_blocks_per_tile - 1, 0), c),
            ),
            pl.BlockSpec((w_kc.shape[0], block_c), lambda b, t, c: (0, c)),
        ],
        out_specs=pl.BlockSpec((1, block_t, block_c), lambda b, t, c: (b, t, c)),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel"),
        ),
        interpret=interpret,
    )(x, x, w_kc)


_CONV_OP: Any = None


def register_conv_op() -> Any:
    """Register (once, eagerly, outside Dynamo) the conv kernel with ``torch.tpu.pallas.jax_op``."""
    global _CONV_OP
    if _CONV_OP is not None:
        return _CONV_OP
    if torch.compiler.is_compiling():
        raise RuntimeError("register_conv_op() must run before torch.compile tracing")
    import hashlib
    import importlib.metadata
    from pathlib import Path

    tpu = torch_tpu()
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for dist in ("jax", "jaxlib", "libtpu"):
        try:
            digest.update(importlib.metadata.version(dist).encode())
        except importlib.metadata.PackageNotFoundError:
            digest.update(b"missing")
    _CONV_OP = tpu.pallas.jax_op(
        f"w2vbert_tpu::causal_dwconv_{digest.hexdigest()[:12]}", causal_depthwise_conv_jax
    )
    return _CONV_OP


def conv_weight_kc(weight: Tensor) -> Tensor:
    """HF depthwise weight ``[C, 1, K]`` -> ``[K_pad, C]`` (K padded to a multiple of 8)."""
    channels, _, kernel = weight.shape
    w = weight.reshape(channels, kernel).t()
    pad = (-kernel) % 8
    return (F.pad(w, (0, 0, 0, pad)) if pad else w).contiguous()


def pallas_causal_depthwise_conv(
    x: Tensor, w_kc: Tensor, kernel_size: int, *, block_t: int = 256, block_c: int = 512
) -> Tensor:
    """Causal depthwise conv on ``[B, T, C]`` TPU tensors (pads T to ``block_t`` internally)."""
    frames, channels = x.size(1), x.size(2)
    block_c = block_c if channels % block_c == 0 else channels
    pad_t = (-frames) % block_t
    if pad_t:
        x = F.pad(x, (0, 0, 0, pad_t))  # causal: right padding never affects valid outputs
    op = _CONV_OP if _CONV_OP is not None else register_conv_op()
    out = op(x.contiguous(), w_kc, kernel_size, block_t, block_c)
    return out[:, :frames] if pad_t else out


__all__ = [
    "causal_depthwise_conv_jax",
    "conv_weight_kc",
    "pallas_causal_depthwise_conv",
    "register_conv_op",
]
