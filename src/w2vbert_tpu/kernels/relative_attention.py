"""Fused 3-region Pallas TPU forward kernel for w2v-BERT 2.0 relative-key attention.

Exploits the mathematical structure of clipped relative-key positional attention:

    score(i, j) = (q_i . k_j + q_i . E[clip(j - i, -L, R) + L]) / sqrt(D)

For each query block ``[block_q, D]``:
1. ``qb = q_scaled @ E^T`` (``[block_q, E_pad]``) is computed once at ``k_idx == 0`` and kept
   in VMEM.
2. Key blocks wholly to the left of the diagonal band (``j - i <= -L``) reuse constant ``qb[:, 0]``.
3. Key blocks wholly to the right of the diagonal band (``j - i >= R``) reuse constant
   ``qb[:, L + R]``.
4. Key blocks overlapping ``-L < j - i < R`` gather ``qb[i, clip(j - i, -L, R) + L]`` via
   ``jnp.take_along_axis``.
5. Online softmax in FP32 and ``@ v`` accumulation run entirely inside VMEM/MXU without
   materializing any ``[B, H, T, T]`` or ``[T, T, D]`` HBM tensor.
"""

from __future__ import annotations

import math
import os
from typing import Any

# Ensure host-side JAX tracing/lowering never opens /dev/vfio/0 while TorchTPU owns the TPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax._src.cloud_tpu_init as _cloud_tpu_init  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from torch import Tensor  # noqa: E402

# Prevent JAX 0.11/0.12 Cloud TPU libtpu version probes when lowering on host CPU.
_cloud_tpu_init.running_in_cloud_tpu_vm = False

DEFAULT_TILE_SIZE: tuple[int, int] = (512, 512)
# TPU v5e has 128 MiB VMEM per core; the 16 MiB scoped default is too small for 512-wide tiles.
VMEM_LIMIT_BYTES = 64 * 1024 * 1024
_TRANS_B = (((1,), (1,)), ((), ()))
_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def _broadcast_row_stat(stat: jax.Array, head_dim: int) -> jax.Array:
    """Adapt a ``[block_q, 128]`` row-stat tensor to ``[block_q, head_dim]``."""
    if head_dim <= 128:
        return stat[:, :head_dim]
    repeats, rem = divmod(head_dim, 128)
    if rem != 0:
        raise ValueError(f"head_dim > 128 must be divisible by 128, got {head_dim}")
    return jnp.tile(stat, (1, repeats))


def _lane_tile(column128: jax.Array, width: int) -> jax.Array:
    """Tile a lane-replicated ``[rows, 128]`` column to ``[rows, width]`` (whole-vreg copies).

    Broadcasting a single-lane ``[rows, 1]`` slice across lanes is very expensive in Mosaic;
    every per-row quantity in this kernel is therefore kept replicated across 128 lanes.
    """
    repeats, rem = divmod(width, 128)
    if rem:
        raise ValueError(f"width must be a multiple of 128, got {width}")
    return jnp.tile(column128, (1, repeats))


def _band_bias(
    qb: jax.Array,
    first128: jax.Array,
    last128: jax.Array,
    q_start: jax.Array,
    k_start: jax.Array,
    *,
    left: int,
    right: int,
    block_k: int,
) -> jax.Array:
    """Relative-key bias ``qb[i, clip(k - q, -L, R) + L]`` for one band tile via lane rotations.

    A ``take_along_axis`` gather needs a ``block_q x block_k x E_pad`` VMEM temporary in Mosaic
    (32 MiB for 256x256 tiles), so the band tile is built with the Transformer-XL "relative
    shift" instead: with ``W = block_q + block_k`` and ``s = k_start - q_start + L - block_q + 1``

    1. ``H[i, w] = qb[i, clip(w + s, 0, L + R)]``: one dynamic lane roll of ``qb`` (zero padded
       to ``W`` lanes) plus two ``where`` selects for the clipped regions.
    2. ``bias[i, c] = H[i, c - i + block_q - 1]``: one strided lane roll (row ``i`` rotated by
       ``block_k + 1 + i``), keeping the first ``block_k`` lanes. Indices stay in
       ``[0, W - 2]``, so nothing wraps.
    """
    block_q, e_pad = qb.shape
    width = block_q + block_k
    num_buckets = left + right + 1
    if e_pad < num_buckets or width < e_pad:
        raise ValueError(f"invalid band geometry: E_pad={e_pad}, W={width}")
    shift = k_start - q_start + left - (block_q - 1)
    qb_wide = jnp.concatenate([qb, jnp.zeros((block_q, width - e_pad), qb.dtype)], axis=1)
    # roll(x, r)[w] == x[(w - r) mod W]; r = -shift mod W gives x[(w + shift) mod W].
    rolled = pltpu.roll(qb_wide, jnp.mod(-shift, width).astype(jnp.int32), 1)
    pos = jax.lax.broadcasted_iota(jnp.int32, (block_q, width), 1) + shift
    first = _lane_tile(first128, width)
    last = _lane_tile(last128, width)
    table = jnp.where(pos <= 0, first, jnp.where(pos >= num_buckets - 1, last, rolled))
    skewed = pltpu.roll(table, block_k + 1, 1, stride=1, stride_axis=0)
    return skewed[:, :block_k]


def _rel_attn_kernel_body(
    q_ref,
    k_ref,
    v_ref,
    e_ref,
    *rest,
    left: int,
    right: int,
    num_k_steps: int,
    masked: bool,
):
    if masked:
        (
            len_ref,
            o_ref,
            m_scratch_ref,
            l_scratch_ref,
            acc_scratch_ref,
            qb_scratch_ref,
            edge_scratch_ref,
            bias_scratch_ref,
        ) = rest
    else:
        (
            o_ref,
            m_scratch_ref,
            l_scratch_ref,
            acc_scratch_ref,
            qb_scratch_ref,
            edge_scratch_ref,
            bias_scratch_ref,
        ) = rest

    block_q = q_ref.shape[2]
    block_k = k_ref.shape[2]
    head_dim = q_ref.shape[3]
    q_idx = pl.program_id(2)
    k_idx = pl.program_id(3)
    q = q_ref[0, 0]

    @pl.when(k_idx == 0)
    def _init():
        m_scratch_ref[0, 0] = jnp.full((block_q, 128), -jnp.inf, jnp.float32)
        l_scratch_ref[0, 0] = jnp.zeros((block_q, 128), jnp.float32)
        acc_scratch_ref[0, 0] = jnp.zeros((block_q, head_dim), jnp.float32)
        e = e_ref[...]
        qb_scratch_ref[0, 0] = jax.lax.dot_general(
            q, e, _TRANS_B, preferred_element_type=jnp.float32
        )
        # Lane-replicated q.E[0] and q.E[L+R] columns via the MXU: rows of E are broadcast along
        # sublanes (cheap), so every output lane holds the same far-region bias value.
        edge_rows = jnp.concatenate(
            [
                jnp.broadcast_to(e[0:1, :], (128, head_dim)),
                jnp.broadcast_to(e[left + right : left + right + 1, :], (128, head_dim)),
            ],
            axis=0,
        )
        edge_scratch_ref[...] = jax.lax.dot_general(
            q, edge_rows, _TRANS_B, preferred_element_type=jnp.float32
        )

    k = k_ref[0, 0]
    v = v_ref[0, 0]
    qb = qb_scratch_ref[0, 0]
    first128 = edge_scratch_ref[:, :128]
    last128 = edge_scratch_ref[:, 128:]

    s = jax.lax.dot_general(q, k, _TRANS_B, preferred_element_type=jnp.float32)

    q_start = q_idx * block_q
    k_start = k_idx * block_k
    is_far_left = (k_start + block_k - 1 - q_start) <= -left
    is_far_right = (k_start - (q_start + block_q - 1)) >= right

    is_band = jnp.logical_not(jnp.logical_or(is_far_left, is_far_right))

    # Select the bias tile with pl.when into VMEM scratch. A value-returning lax.cond over a
    # [block_q, block_k] tile makes Mosaic reserve ~29 MiB of scoped VMEM at 256x256 tiles.
    @pl.when(jnp.logical_not(is_band))
    def _far():
        bias_scratch_ref[...] = _lane_tile(jnp.where(is_far_left, first128, last128), block_k)

    @pl.when(is_band)
    def _band():
        bias_scratch_ref[...] = _band_bias(
            qb, first128, last128, q_start, k_start, left=left, right=right, block_k=block_k
        )

    s = s + bias_scratch_ref[...]

    if masked:
        cols = jax.lax.broadcasted_iota(jnp.int32, (block_q, block_k), 1) + k_start
        valid_len = len_ref[0, 0, 0]
        col_valid = cols < valid_len
        s = jnp.where(col_valid, s, _MASK_VALUE)

    m_prev = m_scratch_ref[0, 0]
    l_prev = l_scratch_ref[0, 0]
    m_curr = jnp.max(s, axis=1)[:, None]
    m_next = jnp.maximum(m_prev, m_curr)

    p = jnp.exp(s - _lane_tile(m_next, block_k))
    if masked:
        p = jnp.where(col_valid, p, 0.0)
    alpha = jnp.exp(m_prev - m_next)
    l_corr = alpha * l_prev
    l_next = jnp.sum(p, axis=1)[:, None] + l_corr

    l_scratch_ref[0, 0] = l_next
    m_scratch_ref[0, 0] = m_next

    l_next_inv = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
    scale_prev = _broadcast_row_stat(l_corr * l_next_inv, head_dim)
    scale_curr = _broadcast_row_stat(l_next_inv, head_dim)

    o_curr = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
    acc_scratch_ref[0, 0] = acc_scratch_ref[0, 0] * scale_prev + o_curr * scale_curr

    @pl.when(k_idx == num_k_steps - 1)
    def _store():
        o_ref[0, 0] = acc_scratch_ref[0, 0].astype(o_ref.dtype)


def _pallas_call_relative_attention(
    q_scaled: jax.Array,
    k: jax.Array,
    v: jax.Array,
    e_padded: jax.Array,
    len_tiled: jax.Array | None,
    *,
    left: int,
    right: int,
    block_q: int,
    block_k: int,
    interpret: bool = False,
) -> jax.Array:
    batch, heads, frames, head_dim = q_scaled.shape
    e_pad = e_padded.shape[0]
    num_q = frames // block_q
    num_k = frames // block_k
    masked = len_tiled is not None

    kernel = jax.tree_util.Partial(
        _rel_attn_kernel_body,
        left=left,
        right=right,
        num_k_steps=num_k,
        masked=masked,
    )
    in_specs = [
        pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, mq, mk: (b, h, mq, 0)),
        pl.BlockSpec((1, 1, block_k, head_dim), lambda b, h, mq, mk: (b, h, mk, 0)),
        pl.BlockSpec((1, 1, block_k, head_dim), lambda b, h, mq, mk: (b, h, mk, 0)),
        pl.BlockSpec((e_pad, head_dim), lambda b, h, mq, mk: (0, 0)),
    ]
    call_args: list[jax.Array] = [q_scaled, k, v, e_padded]
    if len_tiled is not None:
        in_specs.append(pl.BlockSpec((1, 8, 128), lambda b, h, mq, mk: (b, 0, 0)))
        call_args.append(len_tiled)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, heads, num_q, num_k),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((1, 1, block_q, head_dim), lambda b, h, mq, mk: (b, h, mq, 0)),
            scratch_shapes=[
                pltpu.VMEM((1, 1, block_q, 128), jnp.float32),
                pltpu.VMEM((1, 1, block_q, 128), jnp.float32),
                pltpu.VMEM((1, 1, block_q, head_dim), jnp.float32),
                pltpu.VMEM((1, 1, block_q, e_pad), jnp.float32),
                pltpu.VMEM((block_q, 256), jnp.float32),
                pltpu.VMEM((block_q, block_k), jnp.float32),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(q_scaled.shape, q_scaled.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
    )(*call_args)


def pallas_relative_key_attention_unmasked_jax(
    q_scaled: jax.Array,
    k: jax.Array,
    v: jax.Array,
    e_padded: jax.Array,
    left: int = 64,
    right: int = 8,
    block_q: int = 256,
    block_k: int = 256,
) -> jax.Array:
    """Pure JAX entry point for unmasked 3-region relative-key attention."""
    return _pallas_call_relative_attention(
        q_scaled,
        k,
        v,
        e_padded,
        None,
        left=left,
        right=right,
        block_q=block_q,
        block_k=block_k,
    )


def pallas_relative_key_attention_masked_jax(
    q_scaled: jax.Array,
    k: jax.Array,
    v: jax.Array,
    e_padded: jax.Array,
    len_tiled: jax.Array,
    left: int = 64,
    right: int = 8,
    block_q: int = 256,
    block_k: int = 256,
) -> jax.Array:
    """Pure JAX entry point for masked 3-region relative-key attention."""
    return _pallas_call_relative_attention(
        q_scaled,
        k,
        v,
        e_padded,
        len_tiled,
        left=left,
        right=right,
        block_q=block_q,
        block_k=block_k,
    )


_PALLAS_OPS: tuple[Any, Any] | None = None


def _kernel_fingerprint() -> str:
    """Short hash of this module's source and the JAX toolchain versions.

    PyTorch's AOTAutograd/FX graph caches key compiled programs on the FX graph, which refers to
    a custom op only by name. Without a content-derived name, editing the Pallas kernel silently
    reuses previously compiled (stale) Mosaic kernels from ``/tmp/torchinductor_*``.
    """
    import hashlib
    import importlib.metadata
    from pathlib import Path

    digest = hashlib.sha256(Path(__file__).read_bytes())
    for dist in ("jax", "jaxlib", "libtpu"):
        try:
            digest.update(importlib.metadata.version(dist).encode())
        except importlib.metadata.PackageNotFoundError:
            digest.update(b"missing")
    return digest.hexdigest()[:12]


def register_pallas_ops() -> tuple[Any, Any]:
    """Register (once) and return ``(unmasked_op, masked_op)`` via ``torch.tpu.pallas.jax_op``.

    Must run outside ``torch.compile`` tracing: registration inspects the JAX function signature,
    which Dynamo cannot trace. :func:`w2vbert_tpu.optimize` calls it eagerly for Pallas backends.
    """
    global _PALLAS_OPS
    if _PALLAS_OPS is not None:
        return _PALLAS_OPS
    if torch.compiler.is_compiling():
        raise RuntimeError(
            "Pallas ops are not registered; call "
            "w2vbert_tpu.kernels.register_pallas_ops() before torch.compile tracing"
        )
    import torch_tpu  # noqa: F401

    if not hasattr(torch, "tpu") or not hasattr(torch.tpu, "pallas"):
        raise RuntimeError("torch.tpu.pallas is not available in this environment")

    tag = _kernel_fingerprint()
    unmasked_op = torch.tpu.pallas.jax_op(
        f"w2vbert_tpu::rel_key_attn_unmasked_{tag}",
        pallas_relative_key_attention_unmasked_jax,
    )
    masked_op = torch.tpu.pallas.jax_op(
        f"w2vbert_tpu::rel_key_attn_masked_{tag}",
        pallas_relative_key_attention_masked_jax,
    )
    _PALLAS_OPS = (unmasked_op, masked_op)
    return _PALLAS_OPS


def _get_tpu_pallas_ops() -> tuple[Any, Any]:
    if _PALLAS_OPS is not None:
        return _PALLAS_OPS
    return register_pallas_ops()


def select_tile_size(
    frames: int,
    tile_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Choose TPU v5e ``(block_q, block_k)`` aligned to 128-lane VMEM/MXU tiles."""
    if tile_size is not None:
        bq, bk = tile_size
        if bq % 128 != 0 or bk % 128 != 0 or bq < 128 or bk < 128:
            raise ValueError(
                f"tile_size dimensions must be positive multiples of 128, got {tile_size}"
            )
        return bq, bk
    # TPU v5e, B=16 (benchmarks/results/bench_attention.md): one 256x256 tile is fastest up to
    # T=256 and 512x512 beyond; larger tiles waste padding, smaller ones add grid steps.
    if frames <= 128:
        return (128, 128)
    if frames <= 256:
        return (256, 256)
    return (512, 512)


def pallas_relative_key_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embedding: Tensor,
    left: int,
    right: int,
    *,
    lengths: Tensor | None = None,
    tile_size: tuple[int, int] | None = None,
    padded_embedding: Tensor | None = None,
) -> Tensor:
    """Compute exact relative-key attention via the fused Pallas Mosaic TPU kernel.

    Args:
        query: ``[B, H, T, D]`` on ``tpu``.
        key: ``[B, H, T, D]`` on ``tpu``.
        value: ``[B, H, T, D]`` on ``tpu``.
        embedding: ``[L + R + 1, D]`` on ``tpu``.
        left: Left max relative distance.
        right: Right max relative distance.
        lengths: Optional ``[B]`` int32 valid key lengths on ``tpu``.
        tile_size: Optional ``(block_q, block_k)`` tile dimensions (multiples of 128).
        padded_embedding: Optional pre-padded ``[E_pad, D]`` embedding tensor.
    """
    if query.device.type != "tpu":
        raise ValueError(f"pallas_relative_key_attention requires TPU tensors, got {query.device}")
    if query.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported Pallas dtype {query.dtype}; expected bfloat16 or float32")

    batch, heads, frames, head_dim = query.shape
    num_buckets = left + right + 1
    if embedding.size(0) != num_buckets:
        raise ValueError(f"Expected {num_buckets} embedding rows, got {embedding.size(0)}")
    if head_dim > 128 and head_dim % 128 != 0:
        raise ValueError(f"head_dim > 128 must be divisible by 128, got {head_dim}")

    block_q, block_k = select_tile_size(frames, tile_size)
    tile_lcm = math.lcm(block_q, block_k)
    pad_t = ((frames + tile_lcm - 1) // tile_lcm) * tile_lcm - frames

    q_scaled = query * (1.0 / math.sqrt(head_dim))
    k_in = key
    v_in = value
    if pad_t > 0:
        q_scaled = F.pad(q_scaled, (0, 0, 0, pad_t))
        k_in = F.pad(k_in, (0, 0, 0, pad_t))
        v_in = F.pad(v_in, (0, 0, 0, pad_t))

    if padded_embedding is not None:
        e_padded = padded_embedding
    else:
        e_cast = embedding.to(query.dtype)
        e_pad = ((num_buckets + 127) // 128) * 128 - num_buckets
        e_padded = F.pad(e_cast, (0, 0, 0, e_pad)) if e_pad > 0 else e_cast

    unmasked_op, masked_op = _get_tpu_pallas_ops()
    if lengths is None and pad_t == 0:
        out = unmasked_op(
            q_scaled.contiguous(),
            k_in.contiguous(),
            v_in.contiguous(),
            e_padded.contiguous(),
            left,
            right,
            block_q,
            block_k,
        )
    else:
        if lengths is None:
            effective_lengths = torch.full((batch,), frames, device=query.device, dtype=torch.int32)
        else:
            effective_lengths = lengths.to(device=query.device, dtype=torch.int32)
        len_tiled = effective_lengths[:, None, None].expand(batch, 8, 128).contiguous()
        out = masked_op(
            q_scaled.contiguous(),
            k_in.contiguous(),
            v_in.contiguous(),
            e_padded.contiguous(),
            len_tiled,
            left,
            right,
            block_q,
            block_k,
        )

    if pad_t > 0:
        out = out[:, :, :frames, :]
    return out


__all__ = [
    "DEFAULT_TILE_SIZE",
    "register_pallas_ops",
    "pallas_relative_key_attention",
    "pallas_relative_key_attention_masked_jax",
    "pallas_relative_key_attention_unmasked_jax",
    "select_tile_size",
]
