"""Model-level tests for w2vbert_tpu.optimize, state_dict compatibility, and backends."""

from __future__ import annotations

import copy
import io
from typing import Any

import pytest
import torch
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

from w2vbert_tpu import optimize


def _tiny_model(left: int = 6, right: int = 3, layers: int = 3) -> Wav2Vec2BertModel:
    config = Wav2Vec2BertConfig(
        hidden_size=32,
        num_hidden_layers=layers,
        num_attention_heads=4,
        intermediate_size=64,
        feature_projection_input_dim=16,
        conv_depthwise_kernel_size=5,
        position_embeddings_type="relative_key",
        left_max_position_embeddings=left,
        right_max_position_embeddings=right,
        layerdrop=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        conformer_conv_dropout=0.0,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
    )
    torch.manual_seed(0)
    return Wav2Vec2BertModel(config).double().eval()


def test_state_dict_keys_values_and_roundtrip_preserved() -> None:
    original = _tiny_model()
    fast = optimize(copy.deepcopy(original), backend="native")

    sd_orig = original.state_dict()
    sd_fast = fast.state_dict()
    assert list(sd_orig.keys()) == list(sd_fast.keys())
    for key in sd_orig:
        torch.testing.assert_close(sd_orig[key], sd_fast[key], rtol=0, atol=0)

    # Checkpoint serialization round-trip back into both original and optimized models
    buf = io.BytesIO()
    torch.save(sd_fast, buf)
    buf.seek(0)
    reloaded = torch.load(buf, weights_only=True)
    fresh_orig = _tiny_model()
    fresh_orig.load_state_dict(reloaded)
    fast.load_state_dict(reloaded)

    x = torch.randn(2, 32, 16, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(
            fresh_orig(x).last_hidden_state,
            fast(x).last_hidden_state,
            rtol=1e-10,
            atol=1e-10,
        )


def test_outputs_match_unpadded_and_padded_all_layers() -> None:
    original = _tiny_model()
    fast = optimize(copy.deepcopy(original), backend="native")

    torch.manual_seed(1)
    features = torch.randn(3, 40, 16, dtype=torch.float64)
    mask = torch.ones(3, 40, dtype=torch.long)
    mask[1, 31:] = 0
    mask[2, 9:] = 0

    with torch.no_grad():
        ref_unpadded = original(features, output_hidden_states=True).hidden_states
        opt_unpadded = fast(features, output_hidden_states=True).hidden_states
        for a, b in zip(ref_unpadded, opt_unpadded):
            torch.testing.assert_close(b, a, rtol=1e-9, atol=1e-10)

        ref_padded = original(
            features, attention_mask=mask, output_hidden_states=True
        ).hidden_states
        opt_padded = fast(features, attention_mask=mask, output_hidden_states=True).hidden_states
        valid = mask[..., None].to(torch.float64)
        for a, b in zip(ref_padded, opt_padded):
            torch.testing.assert_close(b * valid, a * valid, rtol=1e-9, atol=1e-10)


def test_consecutive_batches_different_padding_and_qkv_invalidation() -> None:
    original = _tiny_model()
    fast = optimize(copy.deepcopy(original), backend="native", prepack_qkv=True)
    features = torch.randn(3, 40, 16, dtype=torch.float64)

    for lengths in ((40, 31, 9), (12, 40, 25), (40, 40, 40), (7, 7, 39)):
        mask = torch.zeros(3, 40, dtype=torch.long)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
        with torch.no_grad():
            a = original(features, attention_mask=mask).last_hidden_state * mask[..., None]
            b = fast(features, attention_mask=mask).last_hidden_state * mask[..., None]
        torch.testing.assert_close(b, a, rtol=1e-9, atol=1e-10)

    # Mutate weight after optimize() via load_state_dict and verify prepack rebuilds automatically
    sd = {k: v * 0.5 for k, v in original.state_dict().items()}
    original.load_state_dict(sd)
    fast.load_state_dict(sd)
    with torch.no_grad():
        torch.testing.assert_close(
            fast(features).last_hidden_state,
            original(features).last_hidden_state,
            rtol=1e-9,
            atol=1e-10,
        )


def test_layer_truncation_and_output_attentions_fallback() -> None:
    original = _tiny_model(layers=4)
    # Keep the transformers default ("sdpa"): optimize() must still return attention weights.
    assert original.config._attn_implementation == "sdpa"
    short: Any = optimize(copy.deepcopy(original), num_layers=2, backend="native")
    reference = copy.deepcopy(original)
    reference.set_attn_implementation("eager")
    features = torch.randn(2, 24, 16, dtype=torch.float64)
    mask = torch.ones(2, 24, dtype=torch.long)
    mask[1, 15:] = 0

    with torch.no_grad():
        full_states = original(
            features, attention_mask=mask, output_hidden_states=True
        ).hidden_states
        short_out = short(features, attention_mask=mask).last_hidden_state
        attn_out = short(features, attention_mask=mask, output_attentions=True)

    assert len(short.encoder.layers) == 2
    valid = mask[..., None].to(torch.float64)
    torch.testing.assert_close(short_out * valid, full_states[2] * valid, rtol=1e-9, atol=1e-10)
    assert short.config._attn_implementation == "eager"
    assert len(attn_out.attentions) == 2
    with torch.no_grad():
        ref_attn = reference(features, attention_mask=mask, output_attentions=True).attentions
    for got, want in zip(attn_out.attentions, ref_attn[:2], strict=True):
        assert got is not None
        torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-10)


def test_pallas_backend_rejects_cpu_tensors_explicitly() -> None:
    model = optimize(_tiny_model(), backend="pallas")
    x = torch.randn(1, 16, 16, dtype=torch.float64)
    with pytest.raises(ValueError, match="requires TPU tensors"):
        model(x)


@pytest.mark.parametrize(("prepack_qkv", "convolution"), [(True, True), (False, False)])
def test_fullgraph_compile_recompiles_across_shapes(prepack_qkv: bool, convolution: bool) -> None:
    """Per-shape recompiles must not trip data-dependent cache guards (``Tensor._version``)."""
    import torch._dynamo

    original = _tiny_model(layers=2)
    fast = optimize(
        copy.deepcopy(original),
        backend="native",
        prepack_qkv=prepack_qkv,
        convolution=convolution,
    )
    torch._dynamo.reset()
    compiled = torch.compile(fast, backend="eager", dynamic=False, fullgraph=True)
    with torch.no_grad():
        for frames in (16, 24, 40):
            features = torch.randn(2, frames, 16, dtype=torch.float64)
            mask = torch.ones(2, frames, dtype=torch.long)
            mask[1, frames // 2 :] = 0
            got = compiled(features, attention_mask=mask).last_hidden_state
            want = original(features, attention_mask=mask).last_hidden_state
            valid = mask[..., None].to(torch.float64)
            torch.testing.assert_close(got * valid, want * valid, rtol=1e-9, atol=1e-10)
    torch._dynamo.reset()


@pytest.mark.parametrize(
    ("batch", "frames", "pallas"),
    [(1, 250, False), (4, 250, False), (8, 250, True), (16, 250, True), (16, 128, False)]
    + [(1, 257, True), (1, 1500, True)],
)
def test_auto_backend_crossover(batch: int, frames: int, pallas: bool) -> None:
    from w2vbert_tpu.model import _auto_prefers_pallas

    assert _auto_prefers_pallas(batch, frames) is pallas
