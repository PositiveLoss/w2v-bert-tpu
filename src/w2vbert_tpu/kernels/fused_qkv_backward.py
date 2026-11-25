"""Backward pass of the packed-Q/K/V relative-key attention kernel (Pallas, TPU v5e).

Follows the FlashAttention-2 backward as w2v-bert-fast does it for Triton, adapted to Mosaic and
to the packed ``[B, T, 3 * H * 64]`` layout (two heads per 128-lane block). With the scaled
query ``q' = q / sqrt(D)``, the bucket scores ``qb = q' E^T`` and the forward's log-sum-exp:

    s_ij = q'_i . k_j + qb[i, clip(j - i, -L, R) + L]      p_ij = exp(s_ij - lse_i)
    dP = dO V^T      Delta_i = dO_i . O_i      dS = P * (dP - Delta)
    dV = P^T dO      dK = dS^T q'      dq = (dS K + dB E) / sqrt(D)      dE = dB^T q'

``dB[i, n]`` sums the score gradients of row ``i`` whose clipped distance is bucket ``n``: one
entry per interior bucket, row sums for the two clipped ones.

Two kernels:

* **dK, dV** (grid ``(B, H/2, T/bk, T/bq)``, query blocks inner) work in transposed orientation
  (``[bk, bq]`` tiles), so every matmul is a plain or right-transposed product and the per-row
  softmax statistics are cheap sublane broadcasts of ``[1, bq]`` rows. Far tiles get their bias
  as a broadcast row; band tiles build it with the forward's relative shift and transpose it.
* **dQ, dB** (grid ``(B, H/2, T/bq, T/bk)``, key blocks inner) work in the forward's
  orientation. Far tiles add row sums to the clipped buckets. Band tiles undo the relative shift
  (the transpose of :func:`~w2vbert_tpu.kernels.relative_attention._band_bias`): a strided lane
  roll with stride ``W - 1`` (rotating row ``i`` by ``bq - 1 - i`` modulo ``W``; Mosaic only takes
  non-negative strides) and one dynamic roll put each interior bucket in its own lane.

``dE`` (``[2, 128, 128]`` per head of a pair) is a small einsum outside the kernels.
"""

from __future__ import annotations

import math
import os
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from torch import Tensor  # noqa: E402

from w2vbert_tpu._tpu import torch_tpu  # noqa: E402
from w2vbert_tpu.kernels.fused_qkv_attention import (  # noqa: E402
    _LANES,
    HEAD_DIM,
    _fingerprint,
    _pallas_call_fused,
)
from w2vbert_tpu.kernels.relative_attention import (  # noqa: E402
    _MASK_VALUE,
    _TRANS_B,
    VMEM_LIMIT_BYTES,
    _band_bias,
    _lane_tile,
)

_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _row_to_column(row: jax.Array) -> jax.Array:
    """``[1, n]`` row -> lane-replicated ``[n, 128]`` column (sublane broadcast + transpose)."""
    return jnp.transpose(jnp.broadcast_to(row, (_LANES, row.shape[1])))


def _edge_rows(e: jax.Array, left: int, right: int, rows: int) -> jax.Array:
    """``[2 * rows, 128]``: ``rows`` copies of ``E[0]``, then of ``E[L + R]`` (head lanes)."""
    return jnp.concatenate(
        [
            jnp.broadcast_to(e[0:1, :], (rows, _LANES)),
            jnp.broadcast_to(e[left + right : left + right + 1, :], (rows, _LANES)),
        ],
        axis=0,
    )


def _inverse_band(
    ds: jax.Array,
    q_start: jax.Array,
    k_start: jax.Array,
    *,
    left: int,
    right: int,
    e_pad: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Bucket gradients of one band tile: the transpose of ``_band_bias``.

    Returns ``(db, left_sum, right_sum)``: ``db`` is ``[bq, e_pad]`` with the interior buckets
    ``1 .. L+R-1`` (other lanes zero); the sums are ``[bq, 1]`` gradients of buckets ``0`` and
    ``L + R`` from this tile.
    """
    block_q, block_k = ds.shape
    width = block_q + block_k
    num_buckets = left + right + 1
    wide = jnp.concatenate([ds, jnp.zeros((block_q, width - block_k), ds.dtype)], axis=1)
    # Undo the skew: g[i, w] = ds[i, w + i - (block_q - 1)]. That is row i rotated by
    # block_q - 1 - i, written with a non-negative stride: -i == (W - 1) * i (mod W).
    g = pltpu.roll(wide, block_q - 1, 1, stride=width - 1, stride_axis=0)
    shift = k_start - q_start + left - (block_q - 1)
    pos = jax.lax.broadcasted_iota(jnp.int32, (block_q, width), 1) + shift
    left_sum = jnp.sum(jnp.where(pos <= 0, g, 0.0), axis=1)[:, None]
    right_sum = jnp.sum(jnp.where(pos >= num_buckets - 1, g, 0.0), axis=1)[:, None]
    # Interior bucket n lives at lane w = n - shift; rotate so that lane n holds it.
    rolled = pltpu.roll(g, jnp.mod(shift, width).astype(jnp.int32), 1)[:, :e_pad]
    n = jax.lax.broadcasted_iota(jnp.int32, (block_q, e_pad), 1)
    w = n - shift
    interior = (n >= 1) & (n <= num_buckets - 2) & (w >= 0) & (w < width)
    return jnp.where(interior, rolled, 0.0), left_sum, right_sum


def _geometry(q_start, k_start, block_q: int, block_k: int, left: int, right: int):
    is_far_left = (k_start + block_k - 1 - q_start) <= -left
    is_far_right = (k_start - (q_start + block_q - 1)) >= right
    return is_far_left, is_far_right, jnp.logical_not(jnp.logical_or(is_far_left, is_far_right))


def _dq_kernel_body(
    q_ref,
    k_ref,
    v_ref,
    e_ref,
    len_ref,
    do_ref,
    lse_ref,
    delta_ref,
    dq_ref,
    db_ref,
    qb_ref,
    edge_ref,
    bias_ref,
    lse_col_ref,
    delta_col_ref,
    dq_acc_ref,
    db_acc_ref,
    dl_ref,
    dr_ref,
    *,
    left: int,
    right: int,
    num_k_steps: int,
):
    block_q = q_ref.shape[1]
    block_k = k_ref.shape[1]
    e_pad = e_ref.shape[1]
    q_idx = pl.program_id(2)
    k_idx = pl.program_id(3)
    q = q_ref[0] * jnp.asarray(_SCALE, q_ref.dtype)
    do = do_ref[0]
    first_head = jax.lax.broadcasted_iota(jnp.int32, (block_q, _LANES), 1) < HEAD_DIM
    head_q = (jnp.where(first_head, q, 0), jnp.where(first_head, 0, q))
    head_do = (jnp.where(first_head, do, 0), jnp.where(first_head, 0, do))

    @pl.when(k_idx == 0)
    def _init():
        dq_acc_ref[...] = jnp.zeros(dq_acc_ref.shape, jnp.float32)
        db_acc_ref[...] = jnp.zeros(db_acc_ref.shape, jnp.float32)
        dl_ref[...] = jnp.zeros(dl_ref.shape, jnp.float32)
        dr_ref[...] = jnp.zeros(dr_ref.shape, jnp.float32)
        for h in range(2):
            e = e_ref[h]
            qb_ref[h] = jax.lax.dot_general(q, e, _TRANS_B, preferred_element_type=jnp.float32)
            edge_ref[h] = jax.lax.dot_general(
                q, _edge_rows(e, left, right, _LANES), _TRANS_B, preferred_element_type=jnp.float32
            )
            lse_col_ref[h] = _row_to_column(lse_ref[0, h])
            delta_col_ref[h] = _row_to_column(delta_ref[0, h])

    k = k_ref[0]
    v = v_ref[0]
    q_start = q_idx * block_q
    k_start = k_idx * block_k
    is_far_left, is_far_right, is_band = _geometry(q_start, k_start, block_q, block_k, left, right)
    cols = jax.lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1) + k_start
    col_valid = cols < len_ref[0, 0, 0]

    for h in range(2):
        first128 = edge_ref[h, :, :_LANES]
        last128 = edge_ref[h, :, _LANES:]

        @pl.when(jnp.logical_not(is_band))
        def _far(h=h, first128=first128, last128=last128):
            bias_ref[h] = _lane_tile(jnp.where(is_far_left, first128, last128), block_k)

        @pl.when(is_band)
        def _band(h=h, first128=first128, last128=last128):
            bias_ref[h] = _band_bias(
                qb_ref[h], first128, last128, q_start, k_start,
                left=left, right=right, block_k=block_k,
            )  # fmt: skip

        s = jax.lax.dot_general(head_q[h], k, _TRANS_B, preferred_element_type=jnp.float32)
        s = jnp.where(col_valid, s + bias_ref[h], _MASK_VALUE)
        p = jnp.where(col_valid, jnp.exp(s - _lane_tile(lse_col_ref[h], block_k)), 0.0)
        dp = jax.lax.dot_general(head_do[h], v, _TRANS_B, preferred_element_type=jnp.float32)
        ds = p * (dp - _lane_tile(delta_col_ref[h], block_k))
        dsk = jax.lax.dot(ds.astype(k.dtype), k, preferred_element_type=jnp.float32)
        head_lanes = first_head if h == 0 else jnp.logical_not(first_head)
        dq_acc_ref[...] += jnp.where(head_lanes, dsk, 0.0)

        row_sum = jnp.sum(ds, axis=1)[:, None]
        dl_ref[h] += jnp.where(is_far_left, row_sum, 0.0)
        dr_ref[h] += jnp.where(is_far_right, row_sum, 0.0)

        @pl.when(is_band)
        def _band_grad(h=h, ds=ds):
            db, left_sum, right_sum = _inverse_band(
                ds, q_start, k_start, left=left, right=right, e_pad=e_pad
            )
            db_acc_ref[h] += db
            dl_ref[h] += left_sum
            dr_ref[h] += right_sum

    @pl.when(k_idx == num_k_steps - 1)
    def _store():
        lane = jax.lax.broadcasted_iota(jnp.int32, (block_q, e_pad), 1)
        dq = dq_acc_ref[...]
        for h in range(2):
            db = (
                db_acc_ref[h]
                + jnp.where(lane == 0, dl_ref[h], 0.0)
                + jnp.where(lane == left + right, dr_ref[h], 0.0)
            )
            db_ref[0, h] = db
            # dB E: E sits in head h's lanes of e_pair[h], so the product lands there too.
            dq = dq + jax.lax.dot(
                db.astype(e_ref.dtype), e_ref[h], preferred_element_type=jnp.float32
            )
        dq_ref[0] = (dq * _SCALE).astype(dq_ref.dtype)


def _dkv_kernel_body(
    q_ref,
    k_ref,
    v_ref,
    e_ref,
    len_ref,
    do_ref,
    lse_ref,
    delta_ref,
    dk_ref,
    dv_ref,
    bias_t_ref,
    dk_acc_ref,
    dv_acc_ref,
    *,
    left: int,
    right: int,
    num_q_steps: int,
):
    block_q = q_ref.shape[1]
    block_k = k_ref.shape[1]
    k_idx = pl.program_id(2)
    q_idx = pl.program_id(3)
    q = q_ref[0] * jnp.asarray(_SCALE, q_ref.dtype)
    do = do_ref[0]
    first_head = jax.lax.broadcasted_iota(jnp.int32, (block_q, _LANES), 1) < HEAD_DIM
    head_q = (jnp.where(first_head, q, 0), jnp.where(first_head, 0, q))
    head_do = (jnp.where(first_head, do, 0), jnp.where(first_head, 0, do))

    @pl.when(q_idx == 0)
    def _init():
        dk_acc_ref[...] = jnp.zeros(dk_acc_ref.shape, jnp.float32)
        dv_acc_ref[...] = jnp.zeros(dv_acc_ref.shape, jnp.float32)

    k = k_ref[0]
    v = v_ref[0]
    q_start = q_idx * block_q
    k_start = k_idx * block_k
    is_far_left, _, is_band = _geometry(q_start, k_start, block_q, block_k, left, right)
    keys = jax.lax.broadcasted_iota(jnp.int32, (block_k, block_q), 0) + k_start
    key_valid = keys < len_ref[0, 0, 0]

    for h in range(2):
        e = e_ref[h]

        @pl.when(jnp.logical_not(is_band))
        def _far(h=h, e=e):
            # Per-query constants q'.E[0] and q'.E[L+R] as [1, bq] rows (8 copies each).
            edge = jax.lax.dot_general(
                _edge_rows(e, left, right, 8), q, _TRANS_B, preferred_element_type=jnp.float32
            )
            row = jnp.where(is_far_left, edge[0:1, :], edge[8:9, :])
            bias_t_ref[h] = jnp.broadcast_to(row, (block_k, block_q))

        @pl.when(is_band)
        def _band(h=h, e=e):
            qb = jax.lax.dot_general(q, e, _TRANS_B, preferred_element_type=jnp.float32)
            edge = jax.lax.dot_general(
                q, _edge_rows(e, left, right, _LANES), _TRANS_B, preferred_element_type=jnp.float32
            )
            bias = _band_bias(
                qb, edge[:, :_LANES], edge[:, _LANES:], q_start, k_start,
                left=left, right=right, block_k=block_k,
            )  # fmt: skip
            bias_t_ref[h] = jnp.transpose(bias)

        s_t = jax.lax.dot_general(k, head_q[h], _TRANS_B, preferred_element_type=jnp.float32)
        s_t = s_t + bias_t_ref[h]
        p_t = jnp.where(key_valid, jnp.exp(s_t - lse_ref[0, h]), 0.0)
        dp_t = jax.lax.dot_general(v, head_do[h], _TRANS_B, preferred_element_type=jnp.float32)
        ds_t = p_t * (dp_t - delta_ref[0, h])
        # head_do / head_q are zero outside head h's lanes, so the products land in its lanes.
        dv_acc_ref[...] += jax.lax.dot(
            p_t.astype(do.dtype), head_do[h], preferred_element_type=jnp.float32
        )
        dk_acc_ref[...] += jax.lax.dot(
            ds_t.astype(q.dtype), head_q[h], preferred_element_type=jnp.float32
        )

    @pl.when(q_idx == num_q_steps - 1)
    def _store():
        dk_ref[0] = dk_acc_ref[...].astype(dk_ref.dtype)
        dv_ref[0] = dv_acc_ref[...].astype(dv_ref.dtype)


def _check(qkv: jax.Array, num_heads: int, block_q: int, block_k: int) -> tuple[int, int, int]:
    batch, frames, width = qkv.shape
    if width != 3 * num_heads * HEAD_DIM or num_heads % 2:
        raise ValueError(
            f"expected [B, T, 3*H*{HEAD_DIM}] with even H, got {qkv.shape}, H={num_heads}"
        )
    if frames % block_q or frames % block_k:
        raise ValueError(f"T={frames} must be a multiple of the tiles ({block_q}, {block_k})")
    return batch, frames, num_heads // 2


def _pallas_call_dq(
    qkv, e_pair, len_tiled, d_out, lse, delta, *, num_heads, left, right, block_q, block_k,
    interpret=False,
):  # fmt: skip
    batch, frames, pairs = _check(qkv, num_heads, block_q, block_k)
    e_pad = e_pair.shape[1]
    kernel = jax.tree_util.Partial(
        _dq_kernel_body, left=left, right=right, num_k_steps=frames // block_k
    )
    row_spec = pl.BlockSpec((1, 2, 1, block_q), lambda b, hp, i, j: (b, hp, 0, i))
    in_specs = [
        pl.BlockSpec((1, block_q, _LANES), lambda b, hp, i, j: (b, i, hp)),
        pl.BlockSpec((1, block_k, _LANES), lambda b, hp, i, j: (b, j, pairs + hp)),
        pl.BlockSpec((1, block_k, _LANES), lambda b, hp, i, j: (b, j, 2 * pairs + hp)),
        pl.BlockSpec((2, e_pad, _LANES), lambda b, hp, i, j: (0, 0, 0)),
        pl.BlockSpec((1, 8, 128), lambda b, hp, i, j: (b, 0, 0)),
        pl.BlockSpec((1, block_q, _LANES), lambda b, hp, i, j: (b, i, hp)),
        row_spec,
        row_spec,
    ]
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, pairs, frames // block_q, frames // block_k),
            in_specs=in_specs,
            out_specs=[
                pl.BlockSpec((1, block_q, _LANES), lambda b, hp, i, j: (b, i, hp)),
                pl.BlockSpec((1, 2, block_q, e_pad), lambda b, hp, i, j: (b, hp, i, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((2, block_q, e_pad), jnp.float32),  # q'.E^T per head
                pltpu.VMEM((2, block_q, 2 * _LANES), jnp.float32),  # far-region columns
                pltpu.VMEM((2, block_q, block_k), jnp.float32),  # bias tile per head
                pltpu.VMEM((2, block_q, _LANES), jnp.float32),  # lse, lane-replicated
                pltpu.VMEM((2, block_q, _LANES), jnp.float32),  # Delta, lane-replicated
                pltpu.VMEM((block_q, _LANES), jnp.float32),  # dq (both heads' lanes)
                pltpu.VMEM((2, block_q, e_pad), jnp.float32),  # interior dB per head
                pltpu.VMEM((2, block_q, _LANES), jnp.float32),  # dB bucket 0 (row sums)
                pltpu.VMEM((2, block_q, _LANES), jnp.float32),  # dB bucket L+R (row sums)
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((batch, frames, num_heads * HEAD_DIM), qkv.dtype),
            jax.ShapeDtypeStruct((batch, num_heads, frames, e_pad), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
    )(qkv, qkv, qkv, e_pair, len_tiled, d_out, lse, delta)


def _pallas_call_dkv(
    qkv, e_pair, len_tiled, d_out, lse, delta, *, num_heads, left, right, block_q, block_k,
    interpret=False,
):  # fmt: skip
    batch, frames, pairs = _check(qkv, num_heads, block_q, block_k)
    e_pad = e_pair.shape[1]
    kernel = jax.tree_util.Partial(
        _dkv_kernel_body, left=left, right=right, num_q_steps=frames // block_q
    )
    row_spec = pl.BlockSpec((1, 2, 1, block_q), lambda b, hp, j, i: (b, hp, 0, i))
    in_specs = [
        pl.BlockSpec((1, block_q, _LANES), lambda b, hp, j, i: (b, i, hp)),
        pl.BlockSpec((1, block_k, _LANES), lambda b, hp, j, i: (b, j, pairs + hp)),
        pl.BlockSpec((1, block_k, _LANES), lambda b, hp, j, i: (b, j, 2 * pairs + hp)),
        pl.BlockSpec((2, e_pad, _LANES), lambda b, hp, j, i: (0, 0, 0)),
        pl.BlockSpec((1, 8, 128), lambda b, hp, j, i: (b, 0, 0)),
        pl.BlockSpec((1, block_q, _LANES), lambda b, hp, j, i: (b, i, hp)),
        row_spec,
        row_spec,
    ]
    kv_spec = pl.BlockSpec((1, block_k, _LANES), lambda b, hp, j, i: (b, j, hp))
    kv_shape = jax.ShapeDtypeStruct((batch, frames, num_heads * HEAD_DIM), qkv.dtype)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, pairs, frames // block_k, frames // block_q),
            in_specs=in_specs,
            out_specs=[kv_spec, kv_spec],
            scratch_shapes=[
                pltpu.VMEM((2, block_k, block_q), jnp.float32),  # transposed bias per head
                pltpu.VMEM((block_k, _LANES), jnp.float32),  # dk (both heads' lanes)
                pltpu.VMEM((block_k, _LANES), jnp.float32),  # dv (both heads' lanes)
            ],
        ),
        out_shape=[kv_shape, kv_shape],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
    )(qkv, qkv, qkv, e_pair, len_tiled, d_out, lse, delta)


# JAX entry points registered with torch.tpu.pallas.jax_op (annotations define the op schema).


def fused_qkv_attention_fwd_jax(
    qkv: jax.Array,
    e_pair: jax.Array,
    len_tiled: jax.Array,
    num_heads: int = 16,
    left: int = 64,
    right: int = 8,
    block_q: int = 512,
    block_k: int = 512,
) -> tuple[jax.Array, jax.Array]:
    """Forward for training: ``(out [B, T, H*64], lse [B, H, 1, T] float32)``."""
    out, lse = _pallas_call_fused(
        qkv, e_pair, len_tiled, num_heads=num_heads, left=left, right=right,
        block_q=block_q, block_k=block_k, with_lse=True,
    )  # fmt: skip
    return out, lse


def fused_qkv_attention_bwd_dq_jax(
    qkv: jax.Array,
    e_pair: jax.Array,
    len_tiled: jax.Array,
    d_out: jax.Array,
    lse: jax.Array,
    delta: jax.Array,
    num_heads: int = 16,
    left: int = 64,
    right: int = 8,
    block_q: int = 512,
    block_k: int = 512,
) -> tuple[jax.Array, jax.Array]:
    """``(dq [B, T, H*64], dB [B, H, T, 128] float32)``."""
    dq, db = _pallas_call_dq(
        qkv, e_pair, len_tiled, d_out, lse, delta, num_heads=num_heads, left=left,
        right=right, block_q=block_q, block_k=block_k,
    )  # fmt: skip
    return dq, db


def fused_qkv_attention_bwd_dkv_jax(
    qkv: jax.Array,
    e_pair: jax.Array,
    len_tiled: jax.Array,
    d_out: jax.Array,
    lse: jax.Array,
    delta: jax.Array,
    num_heads: int = 16,
    left: int = 64,
    right: int = 8,
    block_q: int = 512,
    block_k: int = 512,
) -> tuple[jax.Array, jax.Array]:
    """``(dk [B, T, H*64], dv [B, T, H*64])``."""
    dk, dv = _pallas_call_dkv(
        qkv, e_pair, len_tiled, d_out, lse, delta, num_heads=num_heads, left=left,
        right=right, block_q=block_q, block_k=block_k,
    )  # fmt: skip
    return dk, dv


_TRAIN_OPS: tuple[Any, Any, Any] | None = None


def register_fused_train_ops() -> tuple[Any, Any, Any]:
    """Register (once, eagerly, outside Dynamo) the forward-with-lse and the backward ops."""
    global _TRAIN_OPS
    if _TRAIN_OPS is not None:
        return _TRAIN_OPS
    if torch.compiler.is_compiling():
        raise RuntimeError(
            "fused QKV training ops are not registered; call "
            "w2vbert_tpu.kernels.register_fused_train_ops() before torch.compile tracing"
        )
    import hashlib
    from pathlib import Path

    tpu = torch_tpu()
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(_fingerprint().encode())
    tag = digest.hexdigest()[:12]
    _TRAIN_OPS = (
        tpu.pallas.jax_op(f"w2vbert_tpu::fused_qkv_attn_fwd_{tag}", fused_qkv_attention_fwd_jax),
        tpu.pallas.jax_op(
            f"w2vbert_tpu::fused_qkv_attn_bwd_dq_{tag}", fused_qkv_attention_bwd_dq_jax
        ),
        tpu.pallas.jax_op(
            f"w2vbert_tpu::fused_qkv_attn_bwd_dkv_{tag}", fused_qkv_attention_bwd_dkv_jax
        ),
    )
    return _TRAIN_OPS


def _train_ops() -> tuple[Any, Any, Any]:
    """The three ops; tests replace this to run the kernels in interpret mode on CPU."""
    return _TRAIN_OPS if _TRAIN_OPS is not None else register_fused_train_ops()


class _FusedQKVAttention(torch.autograd.Function):
    """Packed-Q/K/V relative-key attention with gradients for ``qkv`` and ``e_pair``."""

    @staticmethod
    def forward(ctx, qkv, e_pair, len_tiled, num_heads, left, right, block_q, block_k):  # type: ignore[override]
        fwd, _, _ = _train_ops()
        out, lse = fwd(qkv, e_pair, len_tiled, num_heads, left, right, block_q, block_k)
        ctx.save_for_backward(qkv, e_pair, len_tiled, out, lse)
        ctx.dims = (num_heads, left, right, block_q, block_k)
        return out

    @staticmethod
    def backward(ctx, d_out):  # type: ignore[override]
        qkv, e_pair, len_tiled, out, lse = ctx.saved_tensors
        num_heads, left, right, block_q, block_k = ctx.dims
        _, bwd_dq, bwd_dkv = _train_ops()
        batch, frames, _ = d_out.shape
        d_out = d_out.contiguous()
        # Delta = rowsum(dO * O) per head, as a [B, H, 1, T] row like lse.
        delta = (d_out.float() * out.float()).view(batch, frames, num_heads, HEAD_DIM).sum(-1)
        delta = delta.transpose(1, 2).unsqueeze(2).contiguous()
        dims = (num_heads, left, right, block_q, block_k)
        dq, db = bwd_dq(qkv, e_pair, len_tiled, d_out, lse, delta, *dims)
        dk, dv = bwd_dkv(qkv, e_pair, len_tiled, d_out, lse, delta, *dims)
        d_qkv = torch.cat([dq, dk, dv], dim=-1)
        # dE: bias of head h in pair p is q'_pair . e_pair[h] (E sits in head h's lanes), so
        # d e_pair[h][n, l] = sum over b, p, t of dB[b, 2p+h, t, n] * q'[b, t, p, l].
        q_scaled = qkv[..., : num_heads * HEAD_DIM].float() * _SCALE
        q_pairs = q_scaled.view(batch, frames, num_heads // 2, 2 * HEAD_DIM)
        db_pairs = db.view(batch, num_heads // 2, 2, frames, db.shape[-1])
        d_e_pair = torch.einsum("bphtn,btpl->hnl", db_pairs, q_pairs).to(e_pair.dtype)
        return d_qkv, d_e_pair, None, None, None, None, None, None


def fused_qkv_attention_train(
    qkv: Tensor,
    e_pair: Tensor,
    len_tiled: Tensor,
    num_heads: int,
    left: int,
    right: int,
    block_q: int,
    block_k: int,
) -> Tensor:
    """Differentiable fused attention on a tile-aligned packed projection (see the forward op)."""
    out = _FusedQKVAttention.apply(qkv, e_pair, len_tiled, num_heads, left, right, block_q, block_k)
    return out  # type: ignore[return-value]


__all__ = [
    "fused_qkv_attention_bwd_dkv_jax",
    "fused_qkv_attention_bwd_dq_jax",
    "fused_qkv_attention_fwd_jax",
    "fused_qkv_attention_train",
    "register_fused_train_ops",
]
