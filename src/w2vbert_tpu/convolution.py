"""TPU-optimized Conformer convolution module in channels-last ``[B, T, C]`` layout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor

ConvImpl = Literal["conv1d", "unfold", "shift", "pallas"]


def causal_depthwise_conv_conv1d(x: Tensor, weight: Tensor) -> Tensor:
    """Causal depthwise 1-D convolution via left-padded ``F.conv1d`` (``x`` is ``[B, T, C]``)."""
    kernel = weight.size(-1)
    channels = x.size(-1)
    x_bct = x.transpose(1, 2)
    padded = F.pad(x_bct, (kernel - 1, 0))
    return F.conv1d(padded, weight, groups=channels).transpose(1, 2)


def causal_depthwise_conv_unfold(x: Tensor, weight_ck: Tensor) -> Tensor:
    """Causal depthwise 1-D convolution directly in ``[B, T, C]`` layout without transposes.

    Args:
        x: Input tensor ``[B, T, C]``.
        weight_ck: Depthwise weights reshaped to ``[C, K]``.
    """
    kernel = weight_ck.size(-1)
    padded = F.pad(x, (0, 0, kernel - 1, 0))
    windows = padded.unfold(1, kernel, 1)  # [B, T, C, K]
    return (windows * weight_ck).sum(dim=-1)


def causal_depthwise_conv_shift(x: Tensor, weight_ck: Tensor) -> Tensor:
    """Causal depthwise 1-D convolution as ``K`` shifted multiply-adds in ``[B, T, C]`` layout.

    Avoids both transposes and the ``[B, T, C, K]`` window tensor of the unfold variant; XLA fuses
    the ``K`` slices into a single elementwise loop over the left-padded input.

    Args:
        x: Input tensor ``[B, T, C]``.
        weight_ck: Depthwise weights reshaped to ``[C, K]``.
    """
    kernel = weight_ck.size(-1)
    frames = x.size(1)
    padded = F.pad(x, (0, 0, kernel - 1, 0))
    out = padded[:, 0:frames, :] * weight_ck[:, 0]
    for k in range(1, kernel):
        out = out + padded[:, k : k + frames, :] * weight_ck[:, k]
    return out


class TPUConvolutionModule(torch.nn.Module):
    """Drop-in replacement for ``Wav2Vec2BertConvolutionModule`` via class swap.

    Replaces 1x1 ``Conv1d`` layers with ``F.linear`` in ``[B, T, C]`` layout. Parameters are used
    as-is, so ``state_dict()`` keys and values stay identical to Hugging Face.
    """

    conv_impl: ConvImpl = "conv1d"

    if TYPE_CHECKING:
        # Provided by Wav2Vec2BertConvolutionModule once optimize() swaps the class in.
        layer_norm: torch.nn.LayerNorm
        pointwise_conv1: torch.nn.Conv1d
        depthwise_conv: torch.nn.Conv1d
        depthwise_layer_norm: torch.nn.LayerNorm
        activation: torch.nn.Module
        pointwise_conv2: torch.nn.Conv1d
        dropout: torch.nn.Dropout

    def _invalidate_conv_cache(self) -> None:
        """Kept for API compatibility; weight views are now taken inside the traced graph."""
        self.__dict__.pop("_conv_cache", None)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if not (
            type(self.pointwise_conv1)
            is type(self.pointwise_conv2)
            is type(self.depthwise_conv)
            is torch.nn.Conv1d
        ):
            from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
                Wav2Vec2BertConvolutionModule,
            )

            return Wav2Vec2BertConvolutionModule.forward(
                cast(Wav2Vec2BertConvolutionModule, self), hidden_states, attention_mask
            )

        hidden = self.layer_norm(hidden_states)
        if attention_mask is not None:
            hidden = hidden.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

        # 1x1 Conv1d weights [O, I, 1] -> [O, I] are free views; no cache (and therefore no
        # data-dependent cache-validity guard) is needed inside the compiled graph.
        hidden = F.linear(hidden, self.pointwise_conv1.weight.squeeze(-1))
        value, gate = hidden.chunk(2, dim=-1)
        hidden = value * torch.sigmoid(gate)

        if self.conv_impl == "unfold":
            hidden = causal_depthwise_conv_unfold(hidden, self.depthwise_conv.weight.squeeze(1))
        elif self.conv_impl == "pallas":
            from w2vbert_tpu.kernels.causal_conv import conv_weight_kc, pallas_causal_depthwise_conv

            weight = self.depthwise_conv.weight
            hidden = pallas_causal_depthwise_conv(
                hidden, conv_weight_kc(weight).to(hidden.dtype), weight.size(-1)
            )
        elif self.conv_impl == "shift":
            hidden = causal_depthwise_conv_shift(hidden, self.depthwise_conv.weight.squeeze(1))
        else:
            hidden = causal_depthwise_conv_conv1d(hidden, self.depthwise_conv.weight)

        hidden = self.activation(self.depthwise_layer_norm(hidden))
        hidden = F.linear(hidden, self.pointwise_conv2.weight.squeeze(-1))
        return self.dropout(hidden)


__all__ = [
    "ConvImpl",
    "TPUConvolutionModule",
    "causal_depthwise_conv_conv1d",
    "causal_depthwise_conv_shift",
    "causal_depthwise_conv_unfold",
]
