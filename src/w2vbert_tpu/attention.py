"""TPU-native relative-key attention implementations and device-side masking helpers.

Computes exact w2v-BERT 2.0 relative-key self-attention:

    score(i, j) = (q_i . k_j + q_i . E[clip(j - i, -L, R) + L]) / sqrt(D)

without materializing the ``[T, T, D]`` intermediate embedding tensor used by Hugging Face
and without any host-device synchronization (no ``.item()``, ``.tolist()``, or ``nonzero()``).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

NativeAttentionVariant = Literal[
    "auto", "sdpa_gather", "sdpa_index", "explicit_gather", "explicit_index"
]

# XLA's TPU lowering of the [B, H, T, T] bucket gather is fast up to T=512 but falls off a cliff
# at long sequences (8.5 s vs 22 ms for index_select at B=16, T=1500 on TPU v5e), so "auto"
# switches to flat index_select above this length.
GATHER_MAX_FRAMES = 512


def resolve_native_variant(variant: NativeAttentionVariant, frames: int) -> NativeAttentionVariant:
    """Resolve ``"auto"`` to a concrete native variant for a static sequence length.

    TPU v5e, B=16 microbenchmarks (``benchmarks/results/bench_attention.md``): explicit gather is
    fastest at T<=256, SDPA + gather at 256<T<=512, explicit index_select at T>512.
    """
    if variant != "auto":
        return variant
    if frames <= 256:
        return "explicit_gather"
    if frames <= GATHER_MAX_FRAMES:
        return "sdpa_gather"
    return "explicit_index"


def relative_key_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embedding: Tensor,
    left: int,
    right: int,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Exact Hugging Face relative-key attention reference (``[B, H, T, D]`` inputs).

    Args:
        query: Query tensor ``[B, H, Tq, D]``.
        key: Key tensor ``[B, H, Tk, D]``.
        value: Value tensor ``[B, H, Tk, D]``.
        embedding: Distance embedding table ``[L + R + 1, D]``.
        left: ``left_max_position_embeddings`` (64 for w2v-BERT 2.0).
        right: ``right_max_position_embeddings`` (8 for w2v-BERT 2.0).
        attention_mask: Optional additive mask broadcastable to ``[B, H, Tq, Tk]``.
    """
    frames_q = query.size(-2)
    frames_k = key.size(-2)
    pos_q = torch.arange(frames_q, dtype=torch.long, device=query.device).view(-1, 1)
    pos_k = torch.arange(frames_k, dtype=torch.long, device=query.device).view(1, -1)
    distance = (pos_k - pos_q).clamp(-left, right) + left
    pos_emb = embedding[distance].to(query.dtype)
    scale = 1.0 / math.sqrt(query.size(-1))
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    scores = scores + torch.einsum("bhld,lrd->bhlr", query, pos_emb) * scale
    if attention_mask is not None:
        scores = scores + attention_mask.to(scores.dtype)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, value)


def extract_key_lengths(
    attention_mask: Tensor | None,
    frames: int,
    conv_attention_mask: Tensor | None = None,
) -> Tensor | None:
    """Extract per-sequence valid key lengths ``[B]`` (int32) purely on device.

    Supports either the 2-D ``[B, T]`` ``conv_attention_mask`` (1/True for valid positions) or
    the 4-D ``[B, 1, T, T]`` additive mask created by Hugging Face (0 for valid keys, minimum
    float value for padded keys). Never triggers host synchronization.
    """
    if conv_attention_mask is not None:
        if conv_attention_mask.ndim != 2 or conv_attention_mask.size(-1) != frames:
            raise ValueError(
                f"Expected conv_attention_mask shape (B, {frames}), "
                f"got {tuple(conv_attention_mask.shape)}"
            )
        return conv_attention_mask.to(torch.int32).sum(dim=-1, dtype=torch.int32)
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2 and attention_mask.size(-1) == frames:
        return attention_mask.to(torch.int32).sum(dim=-1, dtype=torch.int32)
    if (
        attention_mask.ndim == 4
        and attention_mask.size(1) == 1
        and attention_mask.size(-1) == frames
    ):
        row = attention_mask[:, 0, 0, :]
        if attention_mask.dtype == torch.bool:
            return row.sum(dim=-1, dtype=torch.int32)
        if attention_mask.dtype.is_floating_point:
            return (row == 0).sum(dim=-1, dtype=torch.int32)
    raise ValueError(
        "Unsupported attention_mask shape/dtype: "
        f"{tuple(attention_mask.shape)} {attention_mask.dtype}"
    )


def make_additive_key_mask(
    lengths: Tensor,
    frames: int,
    dtype: torch.dtype,
) -> Tensor:
    """Build a ``[B, 1, 1, T]`` additive mask from per-sequence ``lengths`` ``[B]``."""
    positions = torch.arange(frames, device=lengths.device, dtype=lengths.dtype)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)  # [B, T]
    mask = torch.zeros((lengths.size(0), 1, 1, frames), device=lengths.device, dtype=dtype)
    return mask.masked_fill(~valid[:, None, None, :], torch.finfo(dtype).min)


def build_distance_indices(
    frames_q: int,
    frames_k: int,
    left: int,
    right: int,
    device: torch.device,
) -> Tensor:
    """Compute clipped relative-position bucket indices ``[Tq, Tk]`` in ``[0, L + R]``."""
    pos_q = torch.arange(frames_q, dtype=torch.long, device=device).view(-1, 1)
    pos_k = torch.arange(frames_k, dtype=torch.long, device=device).view(1, -1)
    return (pos_k - pos_q).clamp(-left, right) + left


def build_flat_bucket_indices(
    frames_q: int,
    frames_k: int,
    left: int,
    right: int,
    device: torch.device,
) -> Tensor:
    """Compute flat ``[Tq * Tk]`` indices into a ``[B * H, Tq * (L + R + 1)]`` bucket tensor."""
    num_buckets = left + right + 1
    distance = build_distance_indices(frames_q, frames_k, left, right, device)
    row_offsets = (torch.arange(frames_q, dtype=torch.long, device=device) * num_buckets).view(
        -1, 1
    )
    return (row_offsets + distance).reshape(-1)


def compute_relative_bias(
    q_scaled: Tensor,
    embedding: Tensor,
    left: int,
    right: int,
    *,
    distance: Tensor | None = None,
    flat_index: Tensor | None = None,
    mode: Literal["gather", "index"] = "gather",
) -> Tensor:
    """Compute ``[B, H, Tq, Tk]`` relative-key attention bias from pre-scaled queries."""
    batch, heads, frames_q, _ = q_scaled.shape
    frames_k = frames_q if distance is None else distance.size(-1)
    buckets = torch.matmul(q_scaled, embedding.to(q_scaled.dtype).T)  # [B, H, Tq, L+R+1]
    if mode == "index":
        num_buckets = left + right + 1
        if flat_index is None:
            flat_index = build_flat_bucket_indices(frames_q, frames_k, left, right, q_scaled.device)
        flat_buckets = buckets.reshape(batch * heads, frames_q * num_buckets)
        return flat_buckets.index_select(-1, flat_index).view(batch, heads, frames_q, frames_k)
    if distance is None:
        distance = build_distance_indices(frames_q, frames_k, left, right, q_scaled.device)
    return buckets.gather(-1, distance.expand(batch, heads, frames_q, frames_k))


def native_relative_key_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embedding: Tensor,
    left: int,
    right: int,
    attention_mask: Tensor | None = None,
    *,
    lengths: Tensor | None = None,
    distance: Tensor | None = None,
    flat_index: Tensor | None = None,
    variant: NativeAttentionVariant = "auto",
) -> Tensor:
    """Compute exact relative-key attention using native TPU-friendly PyTorch ops.

    Eliminates Hugging Face's ``[T, T, D]`` embedding expansion by multiplying queries with the
    ``[L + R + 1, D]`` bucket table first and gathering/indexing into ``[B, H, T, T]``.
    """
    frames_k = key.size(-2)
    variant = resolve_native_variant(variant, frames_k)
    scale = 1.0 / math.sqrt(query.size(-1))
    q_scaled = query * scale

    gather_mode: Literal["gather", "index"] = "index" if variant.endswith("_index") else "gather"
    bias = compute_relative_bias(
        q_scaled,
        embedding,
        left,
        right,
        distance=distance,
        flat_index=flat_index,
        mode=gather_mode,
    )

    effective_mask = attention_mask
    if effective_mask is None and lengths is not None:
        effective_mask = make_additive_key_mask(lengths, frames_k, bias.dtype)
    if effective_mask is not None:
        if effective_mask.dtype == torch.bool:
            bias = bias.masked_fill(~effective_mask, torch.finfo(bias.dtype).min)
        else:
            bias = bias + effective_mask.to(bias.dtype)

    if variant.startswith("sdpa_"):
        # Queries are already scaled by 1/sqrt(D), so pass scale=1.0 to SDPA.
        return F.scaled_dot_product_attention(
            query=q_scaled, key=key, value=value, attn_mask=bias, scale=1.0
        )

    scores = torch.matmul(q_scaled, key.transpose(-2, -1)) + bias
    softmax_dtype = torch.promote_types(scores.dtype, torch.float32)
    probs = torch.softmax(scores, dim=-1, dtype=softmax_dtype).to(value.dtype)
    return torch.matmul(probs, value)


__all__ = [
    "GATHER_MAX_FRAMES",
    "NativeAttentionVariant",
    "build_distance_indices",
    "build_flat_bucket_indices",
    "compute_relative_bias",
    "extract_key_lengths",
    "make_additive_key_mask",
    "native_relative_key_attention",
    "relative_key_reference",
    "resolve_native_variant",
]
