"""Helpers for fine-tuned CTC speech recognizers (``Wav2Vec2BertForCTC``)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor


def ctc_output_lengths(config: Any, frame_lengths: Tensor) -> Tensor:
    """Number of valid logit frames for each input of ``frame_lengths`` encoder frames.

    Models with an adapter (``config.add_adapter``) shorten the sequence: each adapter layer is a
    strided convolution (stride 2 by default), so there are about half as many logits as input
    frames. Without an adapter the lengths are unchanged.

    Args:
        config: The model's ``Wav2Vec2BertConfig`` (``model.config``).
        frame_lengths: ``[B]`` valid input frames per row, e.g. ``attention_mask.sum(-1)``.
    """
    lengths = frame_lengths.to(torch.long)
    if getattr(config, "add_adapter", False):
        kernel = int(config.adapter_kernel_size)
        stride = int(config.adapter_stride)
        padding = kernel // 2
        for _ in range(int(config.num_adapter_layers)):
            lengths = torch.div(lengths + 2 * padding - kernel, stride, rounding_mode="floor") + 1
    return lengths


def ctc_greedy_decode(
    logits: Tensor,
    lengths: Tensor | None = None,
    *,
    blank_id: int = 0,
    ignore_ids: Iterable[int] = (),
) -> list[list[int]]:
    """Greedy CTC decoding: best token per frame, merge repeats, drop blanks.

    Args:
        logits: ``[B, T, vocab]`` model output. Rows beyond ``len(lengths)`` (batch padding) are
            ignored when ``lengths`` is given.
        lengths: Optional ``[B]`` valid frames per row (see ``ctc_output_lengths``). Frames past
            a row's length come from padding and are skipped.
        blank_id: The CTC blank token, usually ``config.pad_token_id``.
        ignore_ids: Extra token ids to drop after merging, e.g. ``<unk>``.

    Returns:
        One list of token ids per row, ready for the tokenizer's decode.
    """
    best = logits.argmax(dim=-1).cpu()
    if lengths is None:
        lengths = torch.full((best.size(0),), best.size(1), dtype=torch.long)
    lengths = lengths.cpu()
    drop = {int(blank_id), *(int(i) for i in ignore_ids)}
    results: list[list[int]] = []
    for row, length in zip(best, lengths.tolist()):
        merged = torch.unique_consecutive(row[: int(length)]).tolist()
        results.append([t for t in merged if t not in drop])
    return results


__all__ = ["ctc_greedy_decode", "ctc_output_lengths"]
