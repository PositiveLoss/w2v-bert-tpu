"""Static shape bucketing and prewarming utilities for TPU inference."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

DEFAULT_FRAME_BUCKETS: tuple[int, ...] = (128, 256, 512, 768, 1024, 1536, 2048, 3072, 6144)
DEFAULT_BATCH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class BucketedBatch:
    """Right-padded static-shape batch paired with original valid dimensions."""

    input_features: Tensor
    attention_mask: Tensor
    original_batch: int
    original_frames: int
    bucket_batch: int
    bucket_frames: int


def select_frame_bucket(
    frames: int,
    buckets: Sequence[int] = DEFAULT_FRAME_BUCKETS,
) -> int:
    """Return the smallest configured frame bucket ``>= frames``."""
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    ordered = sorted(int(b) for b in buckets)
    for bucket in ordered:
        if frames <= bucket:
            return bucket
    raise ValueError(
        f"frames={frames} exceeds maximum configured frame bucket {ordered[-1]} "
        f"(available buckets: {ordered})"
    )


def select_batch_bucket(
    batch_size: int,
    buckets: Sequence[int] = DEFAULT_BATCH_BUCKETS,
) -> int:
    """Return the smallest configured batch bucket ``>= batch_size``."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    ordered = sorted(int(b) for b in buckets)
    for bucket in ordered:
        if batch_size <= bucket:
            return bucket
    raise ValueError(
        f"batch_size={batch_size} exceeds maximum configured batch bucket {ordered[-1]} "
        f"(available buckets: {ordered})"
    )


def pad_to_bucket(
    input_features: Tensor,
    attention_mask: Tensor | None = None,
    *,
    frame_buckets: Sequence[int] = DEFAULT_FRAME_BUCKETS,
    batch_buckets: Sequence[int] | None = None,
    padding_value: float = 1.0,
) -> BucketedBatch:
    """Right-pad ``input_features`` and ``attention_mask`` to static bucket shapes."""
    if input_features.ndim != 3:
        raise ValueError(
            f"Expected input_features of shape [B, T, F], got {tuple(input_features.shape)}"
        )
    orig_batch, orig_frames, feat_dim = input_features.shape
    bucket_frames = select_frame_bucket(orig_frames, frame_buckets)
    bucket_batch = (
        select_batch_bucket(orig_batch, batch_buckets) if batch_buckets is not None else orig_batch
    )

    pad_t = bucket_frames - orig_frames
    pad_b = bucket_batch - orig_batch

    if attention_mask is None:
        if pad_t > 0 or pad_b > 0:
            padded_mask = torch.zeros(
                (bucket_batch, bucket_frames),
                device=input_features.device,
                dtype=torch.long,
            )
            padded_mask[:orig_batch, :orig_frames] = 1
        else:
            padded_mask = torch.ones(
                (bucket_batch, bucket_frames),
                device=input_features.device,
                dtype=torch.long,
            )
    else:
        if attention_mask.shape != (orig_batch, orig_frames):
            raise ValueError(
                f"Expected attention_mask shape {(orig_batch, orig_frames)}, "
                f"got {tuple(attention_mask.shape)}"
            )
        padded_mask = attention_mask
        if pad_t > 0 or pad_b > 0:
            padded_mask = torch.nn.functional.pad(padded_mask, (0, pad_t, 0, pad_b), value=0)

    padded_features = input_features
    if pad_t > 0 or pad_b > 0:
        padded_features = torch.nn.functional.pad(
            padded_features, (0, 0, 0, pad_t, 0, pad_b), value=padding_value
        )

    return BucketedBatch(
        input_features=padded_features,
        attention_mask=padded_mask,
        original_batch=orig_batch,
        original_frames=orig_frames,
        bucket_batch=bucket_batch,
        bucket_frames=bucket_frames,
    )


def unpad_output(
    output: Tensor,
    original_batch: int,
    original_frames: int,
) -> Tensor:
    """Slice bucket-padded model output back to ``[original_batch, original_frames, ...]``."""
    return output[:original_batch, :original_frames]


def prewarm(
    model: torch.nn.Module,
    *,
    frame_buckets: Sequence[int] = (128, 256, 512, 1024, 1536),
    batch_buckets: Sequence[int] = (1, 4, 16),
    feature_dim: int = 160,
    device: torch.device | str = "tpu",
    dtype: torch.dtype = torch.bfloat16,
) -> list[dict[str, Any]]:
    """Precompile and warm static shape buckets, returning per-bucket compilation timings."""
    records: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for frames in frame_buckets:
            for batch in batch_buckets:
                try:
                    x = torch.randn(batch, frames, feature_dim, device=device, dtype=dtype)
                    mask = torch.ones(batch, frames, device=device, dtype=torch.long)
                    if torch.device(device).type == "tpu":
                        torch.accelerator.synchronize()
                    t0 = time.perf_counter()
                    _ = model(input_features=x, attention_mask=mask).last_hidden_state
                    if torch.device(device).type == "tpu":
                        torch.accelerator.synchronize()
                    elapsed_s = time.perf_counter() - t0
                    records.append(
                        {
                            "batch_size": int(batch),
                            "frames": int(frames),
                            "status": "ok",
                            "compile_and_first_run_s": round(elapsed_s, 4),
                        }
                    )
                except Exception as exc:
                    records.append(
                        {
                            "batch_size": int(batch),
                            "frames": int(frames),
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    return records


__all__ = [
    "BucketedBatch",
    "DEFAULT_BATCH_BUCKETS",
    "DEFAULT_FRAME_BUCKETS",
    "pad_to_bucket",
    "prewarm",
    "select_batch_bucket",
    "select_frame_bucket",
    "unpad_output",
]
