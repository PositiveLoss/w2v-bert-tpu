"""Tests for channels-last Conformer convolution module against Hugging Face."""

from __future__ import annotations

import copy
import os

import pytest
import torch
from transformers import Wav2Vec2BertConfig
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertConvolutionModule

from w2vbert_tpu.convolution import (
    TPUConvolutionModule,
    causal_depthwise_conv_conv1d,
    causal_depthwise_conv_shift,
    causal_depthwise_conv_unfold,
)


@pytest.mark.parametrize(
    "batch,frames,channels,kernel",
    [(2, 1, 16, 31), (3, 37, 64, 31), (2, 250, 128, 31), (1, 64, 32, 5)],
)
def test_causal_depthwise_conv_impls_match_fp64(
    batch: int, frames: int, channels: int, kernel: int
) -> None:
    torch.manual_seed(frames + channels)
    x = torch.randn(batch, frames, channels, dtype=torch.float64)
    w = 0.2 * torch.randn(channels, 1, kernel, dtype=torch.float64)
    y_conv1d = causal_depthwise_conv_conv1d(x, w)
    y_unfold = causal_depthwise_conv_unfold(x, w.squeeze(1))
    torch.testing.assert_close(y_unfold, y_conv1d, rtol=1e-12, atol=1e-12)
    y_shift = causal_depthwise_conv_shift(x, w.squeeze(1))
    torch.testing.assert_close(y_shift, y_conv1d, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("conv_impl", ["conv1d", "unfold", "shift"])
@pytest.mark.parametrize("use_mask", [False, True])
def test_tpu_convolution_module_matches_hf_fp64(conv_impl: str, use_mask: bool) -> None:
    config = Wav2Vec2BertConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=128,
        feature_projection_input_dim=32,
        conv_depthwise_kernel_size=31,
        conformer_conv_dropout=0.0,
    )
    torch.manual_seed(0)
    ref_mod = Wav2Vec2BertConvolutionModule(config).double().eval()
    opt_mod = copy.deepcopy(ref_mod)
    opt_mod.__class__ = type(
        "FastTPUConvolutionModule", (TPUConvolutionModule, Wav2Vec2BertConvolutionModule), {}
    )
    opt_mod.conv_impl = conv_impl  # type: ignore[attr-defined]

    x = torch.randn(3, 85, 64, dtype=torch.float64)
    mask = None
    if use_mask:
        mask = torch.zeros(3, 85, dtype=torch.long)
        mask[0, :85] = 1
        mask[1, :51] = 1
        mask[2, :17] = 1

    with torch.no_grad():
        ref_out = ref_mod(x, attention_mask=mask)
        opt_out = opt_mod(x, attention_mask=mask)
    torch.testing.assert_close(opt_out, ref_out, rtol=1e-11, atol=1e-11)


def test_tpu_convolution_module_device_parity() -> None:
    if os.environ.get("W2VBERT_RUN_TPU_TESTS", "0") != "1":
        pytest.skip("Set W2VBERT_RUN_TPU_TESTS=1 to run TPU convolution tests")
    import torch_tpu  # noqa: F401

    if not torch.accelerator.is_available():
        pytest.skip("No TPU accelerator available")

    config = Wav2Vec2BertConfig(
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=512,
        feature_projection_input_dim=160,
        conv_depthwise_kernel_size=31,
        conformer_conv_dropout=0.0,
    )
    torch.manual_seed(11)
    ref_mod = Wav2Vec2BertConvolutionModule(config).eval()
    x32 = torch.randn(4, 250, 256, dtype=torch.float32)
    mask = torch.ones(4, 250, dtype=torch.long)
    mask[1, 180:] = 0
    mask[2, 95:] = 0

    with torch.no_grad():
        ref_out = ref_mod(x32, attention_mask=mask)

    for conv_impl in ("conv1d", "shift"):
        tpu_mod = copy.deepcopy(ref_mod).to(device="tpu", dtype=torch.bfloat16)
        tpu_mod.__class__ = type(
            "FastTPUConvolutionModule", (TPUConvolutionModule, Wav2Vec2BertConvolutionModule), {}
        )
        tpu_mod.conv_impl = conv_impl  # type: ignore[attr-defined]
        with torch.inference_mode():
            out = tpu_mod(
                x32.to(device="tpu", dtype=torch.bfloat16),
                attention_mask=mask.to(device="tpu"),
            )
            torch.accelerator.synchronize()
        rel_l2 = float(((out.float().cpu() - ref_out).norm() / ref_out.norm()).item())
        assert rel_l2 < 1.5e-2, f"{conv_impl} rel_l2={rel_l2}"


@pytest.mark.parametrize(("frames", "block_t"), [(256, 256), (512, 256), (768, 256), (512, 128)])
def test_pallas_causal_conv_interpret_matches_conv1d(frames: int, block_t: int) -> None:
    import jax.numpy as jnp
    import numpy as np

    from w2vbert_tpu.convolution import causal_depthwise_conv_conv1d
    from w2vbert_tpu.kernels.causal_conv import causal_depthwise_conv_jax, conv_weight_kc

    torch.manual_seed(frames)
    x = torch.randn(2, frames, 256)
    w = torch.randn(256, 1, 31)
    ref = causal_depthwise_conv_conv1d(x, w)
    out = causal_depthwise_conv_jax(
        jnp.asarray(x.numpy()),
        jnp.asarray(conv_weight_kc(w).numpy()),
        31,
        block_t,
        128,
        interpret=True,
    )
    torch.testing.assert_close(torch.from_numpy(np.array(out)), ref, rtol=1e-4, atol=1e-4)
