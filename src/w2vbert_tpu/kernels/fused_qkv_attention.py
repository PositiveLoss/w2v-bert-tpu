"""Relative-key attention that reads the packed Q/K/V projection in place (Pallas, TPU v5e).

Port of the w2v-bert-fast design ("one projection for q, k and v, whose output the attention
kernel reads in place") to Mosaic. The input is ``qkv = x @ [Wq; Wk; Wv]^T + b`` with layout
``[B, T, 3 * H * D]`` and the output is ``[B, T, H * D]``, ready for ``linear_out``: no head
transposes, ``.contiguous()`` copies or per-tensor pads.

With ``D = 64`` one 128-lane block holds a *pair* of heads. The grid is
``(B, H / 2, T / block_q, T / block_k)``; for head pair ``hp`` the q, k and v tiles are lane
blocks ``hp``, ``H / 2 + hp`` and ``H + hp`` of the same array. Inside the kernel each head is
handled without lane slicing:

* ``q_h . k^T`` uses ``q`` with the other head's 64 lanes zeroed (exact; the v5e MXU contracts
  128 lanes anyway);
* ``q_h . E^T`` uses ``E`` placed in that head's lanes (``e_pair[h]``);
* the two heads' ``p_h @ v`` results and softmax statistics are merged with lane selects.

The relative bias (far-left/far-right constant columns, Transformer-XL band shift) and the online
softmax are the same as in :mod:`w2vbert_tpu.kernels.relative_attention`.
"""

from __future__ import annotations

import math
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
from w2vbert_tpu.kernels.relative_attention import (  # noqa: E402
    _MASK_VALUE,
    _TRANS_B,
    VMEM_LIMIT_BYTES,
    _band_bias,
    _lane_tile,
    select_tile_size,
)

HEAD_DIM = 64
_LANES = 128


def _fused_kernel_body(
    q_ref,
    k_ref,
    v_ref,
    e_ref,
    *rest,
    left: int,
    right: int,
    num_k_steps: int,
    masked: bool,
    scale: float,
):
    if masked:
        len_ref, o_ref, m_ref, l_ref, acc_ref, qb_ref, edge_ref, bias_ref = rest
    else:
        len_ref = None
        o_ref, m_ref, l_ref, acc_ref, qb_ref, edge_ref, bias_ref = rest

    block_q = q_ref.shape[1]
    block_k = k_ref.shape[1]
    q_idx = pl.program_id(2)
    k_idx = pl.program_id(3)
    # D = 64 => scale = 1/8: exact in BF16.
    q = q_ref[0] * jnp.asarray(scale, q_ref.dtype)
    first_head = jax.lax.broadcasted_iota(jnp.int32, (block_q, _LANES), 1) < HEAD_DIM
    head_q = (jnp.where(first_head, q, 0), jnp.where(first_head, 0, q))

    @pl.when(k_idx == 0)
    def _init():
        m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, jnp.float32)
        l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
        for h in range(2):
            e = e_ref[h]  # [E_pad, 128]: E in head h's lanes, zeros elsewhere
            qb_ref[h] = jax.lax.dot_general(q, e, _TRANS_B, preferred_element_type=jnp.float32)
            edge_rows = jnp.concatenate(
                [
                    jnp.broadcast_to(e[0:1, :], (_LANES, _LANES)),
                    jnp.broadcast_to(e[left + right : left + right + 1, :], (_LANES, _LANES)),
                ],
                axis=0,
            )
            edge_ref[h] = jax.lax.dot_general(
                q, edge_rows, _TRANS_B, preferred_element_type=jnp.float32
            )

    k = k_ref[0]
    v = v_ref[0]
    q_start = q_idx * block_q
    k_start = k_idx * block_k
    is_far_left = (k_start + block_k - 1 - q_start) <= -left
    is_far_right = (k_start - (q_start + block_q - 1)) >= right
    is_band = jnp.logical_not(jnp.logical_or(is_far_left, is_far_right))
    col_valid = None
    if len_ref is not None:
        cols = jax.lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1) + k_start
        col_valid = cols < len_ref[0, 0, 0]

    alphas, outs = [], []
    for h in range(2):
        first128 = edge_ref[h, :, :_LANES]
        last128 = edge_ref[h, :, _LANES:]

        @pl.when(jnp.logical_not(is_band))
        def _far(h=h, first128=first128, last128=last128):
            bias_ref[h] = _lane_tile(jnp.where(is_far_left, first128, last128), block_k)

        @pl.when(is_band)
        def _band(h=h, first128=first128, last128=last128):
            bias_ref[h] = _band_bias(
                qb_ref[h],
                first128,
                last128,
                q_start,
                k_start,
                left=left,
                right=right,
                block_k=block_k,
            )

        s = jax.lax.dot_general(head_q[h], k, _TRANS_B, preferred_element_type=jnp.float32)
        s = s + bias_ref[h]
        if col_valid is not None:
            s = jnp.where(col_valid, s, _MASK_VALUE)
        m_prev = m_ref[h]
        m_next = jnp.maximum(m_prev, jnp.max(s, axis=1)[:, None])
        p = jnp.exp(s - _lane_tile(m_next, block_k))
        if col_valid is not None:
            p = jnp.where(col_valid, p, 0.0)
        alpha = jnp.exp(m_prev - m_next)
        l_ref[h] = alpha * l_ref[h] + jnp.sum(p, axis=1)[:, None]
        m_ref[h] = m_next
        alphas.append(alpha)
        outs.append(jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32))

    acc_ref[...] = acc_ref[...] * jnp.where(first_head, alphas[0], alphas[1]) + jnp.where(
        first_head, outs[0], outs[1]
    )

    @pl.when(k_idx == num_k_steps - 1)
    def _store():
        denom = jnp.where(first_head, l_ref[0], l_ref[1])
        inv = jnp.where(denom == 0.0, 0.0, 1.0 / denom)
        o_ref[0] = (acc_ref[...] * inv).astype(o_ref.dtype)


def _pallas_call_fused(
    qkv: jax.Array,
    e_pair: jax.Array,
    len_tiled: jax.Array | None,
    *,
    num_heads: int,
    left: int,
    right: int,
    block_q: int,
    block_k: int,
    interpret: bool = False,
) -> jax.Array:
    batch, frames, width = qkv.shape
    pairs = num_heads // 2
    if width != 3 * num_heads * HEAD_DIM or num_heads % 2:
        raise ValueError(
            f"expected [B, T, 3*H*{HEAD_DIM}] with even H, got {qkv.shape}, H={num_heads}"
        )
    num_q, num_k = frames // block_q, frames // block_k
    e_pad = e_pair.shape[1]
    masked = len_tiled is not None
    kernel = jax.tree_util.Partial(
        _fused_kernel_body,
        left=left,
        right=right,
        num_k_steps=num_k,
        masked=masked,
        scale=1.0 / math.sqrt(HEAD_DIM),
    )
    in_specs = [
        pl.BlockSpec((1, block_q, _LANES), lambda b, hp, i, j: (b, i, hp)),
        pl.BlockSpec((1, block_k, _LANES), lambda b, hp, i, j: (b, j, pairs + hp)),
        pl.BlockSpec((1, block_k, _LANES), lambda b, hp, i, j: (b, j, 2 * pairs + hp)),
        pl.BlockSpec((2, e_pad, _LANES), lambda b, hp, i, j: (0, 0, 0)),
    ]
    args: list[jax.Array] = [qkv, qkv, qkv, e_pair]
    if masked:
        in_specs.append(pl.BlockSpec((1, 8, 128), lambda b, hp, i, j: (b, 0, 0)))
        args.append(len_tiled)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, pairs, num_q, num_k),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((1, block_q, _LANES), lambda b, hp, i, j: (b, i, hp)),
            scratch_shapes=[
                pltpu.VMEM((2, block_q, _LANES), jnp.float32),  # m per head
                pltpu.VMEM((2, block_q, _LANES), jnp.float32),  # l per head
                pltpu.VMEM((block_q, _LANES), jnp.float32),  # acc (both heads' lanes)
                pltpu.VMEM((2, block_q, e_pad), jnp.float32),  # q.E^T per head
                pltpu.VMEM((2, block_q, 2 * _LANES), jnp.float32),  # far-region columns
                pltpu.VMEM((2, block_q, block_k), jnp.float32),  # bias tile per head
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((batch, frames, num_heads * HEAD_DIM), qkv.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
    )(*args)


def fused_qkv_attention_unmasked_jax(
    qkv: jax.Array,
    e_pair: jax.Array,
    num_heads: int = 16,
    left: int = 64,
    right: int = 8,
    block_q: int = 512,
    block_k: int = 512,
) -> jax.Array:
    """Pure JAX entry point (no key-length mask)."""
    return _pallas_call_fused(
        qkv, e_pair, None, num_heads=num_heads, left=left, right=right,
        block_q=block_q, block_k=block_k,
    )  # fmt: skip


def fused_qkv_attention_masked_jax(
    qkv: jax.Array,
    e_pair: jax.Array,
    len_tiled: jax.Array,
    num_heads: int = 16,
    left: int = 64,
    right: int = 8,
    block_q: int = 512,
    block_k: int = 512,
) -> jax.Array:
    """Pure JAX entry point with per-sequence key lengths."""
    return _pallas_call_fused(
        qkv, e_pair, len_tiled, num_heads=num_heads, left=left, right=right,
        block_q=block_q, block_k=block_k,
    )  # fmt: skip


def pair_embedding(embedding: Tensor) -> Tensor:
    """``[L+R+1, 64]`` table -> ``[2, 128, 128]``: rows padded to 128, E in lanes 0:64 / 64:128."""
    rows, dim = embedding.shape
    if dim != HEAD_DIM:
        raise ValueError(f"fused QKV attention requires head_dim={HEAD_DIM}, got {dim}")
    e_rows = F.pad(embedding, (0, 0, 0, _LANES - rows)) if rows < _LANES else embedding
    zeros = torch.zeros_like(e_rows)
    return torch.stack(
        [torch.cat([e_rows, zeros], dim=1), torch.cat([zeros, e_rows], dim=1)]
    ).contiguous()


_FUSED_OPS: tuple[Any, Any] | None = None


def _fingerprint() -> str:
    import hashlib
    import importlib.metadata
    from pathlib import Path

    from w2vbert_tpu.kernels import relative_attention

    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(Path(relative_attention.__file__).read_bytes())
    for dist in ("jax", "jaxlib", "libtpu"):
        try:
            digest.update(importlib.metadata.version(dist).encode())
        except importlib.metadata.PackageNotFoundError:
            digest.update(b"missing")
    return digest.hexdigest()[:12]


def register_fused_ops() -> tuple[Any, Any]:
    """Register (once, eagerly, outside Dynamo) and return ``(unmasked_op, masked_op)``."""
    global _FUSED_OPS
    if _FUSED_OPS is not None:
        return _FUSED_OPS
    if torch.compiler.is_compiling():
        raise RuntimeError(
            "fused QKV Pallas ops are not registered; call "
            "w2vbert_tpu.kernels.register_pallas_ops() before torch.compile tracing"
        )
    tpu = torch_tpu()
    tag = _fingerprint()
    _FUSED_OPS = (
        tpu.pallas.jax_op(
            f"w2vbert_tpu::fused_qkv_attn_unmasked_{tag}", fused_qkv_attention_unmasked_jax
        ),
        tpu.pallas.jax_op(
            f"w2vbert_tpu::fused_qkv_attn_masked_{tag}", fused_qkv_attention_masked_jax
        ),
    )
    return _FUSED_OPS


def fused_qkv_attention(
    qkv: Tensor,
    e_pair: Tensor,
    num_heads: int,
    left: int,
    right: int,
    *,
    lengths: Tensor | None = None,
    tile_size: tuple[int, int] | None = None,
) -> Tensor:
    """Relative-key attention on a packed ``[B, T, 3*H*64]`` projection -> ``[B, T, H*64]``.

    Args:
        qkv: Packed projection (q heads, then k heads, then v heads) on ``tpu``.
        e_pair: :func:`pair_embedding` of the distance table, same dtype as ``qkv``.
        num_heads: Number of heads ``H`` (even).
        left: Left max relative distance.
        right: Right max relative distance.
        lengths: Optional ``[B]`` int32 valid key lengths.
        tile_size: Optional ``(block_q, block_k)`` override.
    """
    if qkv.device.type != "tpu":
        raise ValueError(f"fused_qkv_attention requires TPU tensors, got {qkv.device}")
    batch, frames, _ = qkv.shape
    block_q, block_k = select_tile_size(frames, tile_size)
    tile = math.lcm(block_q, block_k)
    pad_t = (-frames) % tile
    if pad_t:
        # Padded keys are masked by length; padded query rows are sliced off below.
        qkv = F.pad(qkv, (0, 0, 0, pad_t))
    unmasked_op, masked_op = _FUSED_OPS if _FUSED_OPS is not None else register_fused_ops()
    if lengths is None and pad_t == 0:
        out = unmasked_op(qkv, e_pair, num_heads, left, right, block_q, block_k)
    else:
        if lengths is None:
            lengths = torch.full((batch,), frames, device=qkv.device, dtype=torch.int32)
        len_tiled = lengths.to(torch.int32)[:, None, None].expand(batch, 8, 128).contiguous()
        out = masked_op(qkv, e_pair, len_tiled, num_heads, left, right, block_q, block_k)
    return out[:, :frames] if pad_t else out


__all__ = [
    "HEAD_DIM",
    "fused_qkv_attention",
    "fused_qkv_attention_masked_jax",
    "fused_qkv_attention_unmasked_jax",
    "pair_embedding",
    "register_fused_ops",
]
