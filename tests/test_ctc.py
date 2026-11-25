"""Fine-tuned CTC models (``Wav2Vec2BertForCTC`` with an adapter) and the CTC helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertForCTC, Wav2Vec2BertModel

from w2vbert_tpu import ctc_greedy_decode, ctc_output_lengths, optimize
from w2vbert_tpu.model import TPUEncoder, TPUSelfAttention, _model_class_for


def _config(**overrides: Any) -> Wav2Vec2BertConfig:
    values: dict[str, Any] = dict(
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=64,
        feature_projection_input_dim=16,
        conv_depthwise_kernel_size=5,
        position_embeddings_type="relative_key",
        left_max_position_embeddings=6,
        right_max_position_embeddings=3,
        add_adapter=True,
        num_adapter_layers=1,
        output_hidden_size=32,
        vocab_size=12,
        layerdrop=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        conformer_conv_dropout=0.0,
        final_dropout=0.0,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
    )
    values.update(overrides)
    return Wav2Vec2BertConfig(**values)


def _tiny_ctc(**overrides: Any) -> Wav2Vec2BertForCTC:
    torch.manual_seed(0)
    return Wav2Vec2BertForCTC(_config(**overrides)).double().eval()


def _inputs(batch: int = 3, frames: int = 41) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    features = torch.randn(batch, frames, 16, dtype=torch.float64)
    mask = torch.ones(batch, frames, dtype=torch.long)
    mask[1, 30:] = 0
    mask[2, 7:] = 0
    return features, mask


@pytest.mark.parametrize("add_adapter", [True, False])
def test_ctc_model_matches_hf(add_adapter: bool) -> None:
    ref = _tiny_ctc(add_adapter=add_adapter)
    fast: Any = optimize(copy.deepcopy(ref), backend="native")
    info: Any = getattr(fast, "_w2vbert_tpu")
    assert info["attention_modules"] == 3  # encoder layers only; the adapter keeps HF attention
    assert isinstance(fast.wav2vec2_bert.encoder, TPUEncoder)
    if add_adapter:
        assert not isinstance(fast.wav2vec2_bert.adapter.layers[0].self_attn, TPUSelfAttention)

    features, mask = _inputs()
    with torch.no_grad():
        expected = ref(features, attention_mask=mask).logits
        got = fast(features, attention_mask=mask).logits
    lengths = ctc_output_lengths(ref.config, mask.sum(-1))
    for row, length in enumerate(lengths.tolist()):
        torch.testing.assert_close(got[row, :length], expected[row, :length], rtol=1e-9, atol=1e-10)


def test_ctc_num_layers_truncates_inner_encoder() -> None:
    fast: Any = optimize(_tiny_ctc(), num_layers=2, backend="native")
    assert len(fast.wav2vec2_bert.encoder.layers) == 2
    assert fast.config.num_hidden_layers == 2


def test_ctc_fullgraph_compile_with_adapter_and_padding() -> None:
    ref = _tiny_ctc()
    fast = optimize(copy.deepcopy(ref), backend="native")
    compiled = torch.compile(fast, backend="eager", dynamic=False, fullgraph=True)
    for frames in (24, 41):
        features, mask = _inputs(frames=frames)
        with torch.no_grad():
            got = compiled(features, attention_mask=mask).logits
            expected = ref(features, attention_mask=mask).logits
        lengths = ctc_output_lengths(ref.config, mask.sum(-1))
        for row, length in enumerate(lengths.tolist()):
            torch.testing.assert_close(got[row, :length], expected[row, :length])


@pytest.mark.parametrize("kernel,stride,layers", [(3, 2, 1), (3, 2, 2), (5, 3, 1)])
def test_ctc_output_lengths_match_hf(kernel: int, stride: int, layers: int) -> None:
    model: Any = Wav2Vec2BertForCTC(
        _config(adapter_kernel_size=kernel, adapter_stride=stride, num_adapter_layers=layers)
    )
    lengths = torch.arange(1, 200)
    expected = model._get_feat_extract_output_lengths(lengths)
    assert torch.equal(ctc_output_lengths(model.config, lengths), expected)
    no_adapter = _config(add_adapter=False)
    assert torch.equal(ctc_output_lengths(no_adapter, lengths), lengths)


def test_ctc_greedy_decode() -> None:
    # Frames: 3 3 0 3 5 5 1 0 | padding
    ids = torch.tensor([[3, 3, 0, 3, 5, 5, 1, 0, 7, 7], [0, 0, 2, 2, 2, 0, 0, 0, 0, 0]])
    logits = torch.nn.functional.one_hot(ids, num_classes=8).float()
    assert ctc_greedy_decode(logits) == [[3, 3, 5, 1, 7], [2]]
    assert ctc_greedy_decode(logits, torch.tensor([8, 10]), ignore_ids=[1]) == [[3, 3, 5], [2]]
    # Extra (batch-padding) rows beyond len(lengths) are dropped.
    assert ctc_greedy_decode(logits, torch.tensor([2])) == [[3]]


def test_load_picks_class_from_config(tmp_path: Path) -> None:
    ctc_dir = tmp_path / "ctc"
    base_dir = tmp_path / "base"
    _config(architectures=["Wav2Vec2BertForCTC"]).save_pretrained(ctc_dir)
    _config(architectures=["Wav2Vec2BertModel"]).save_pretrained(base_dir)
    _config().save_pretrained(tmp_path)
    assert _model_class_for(str(ctc_dir), {}) is Wav2Vec2BertForCTC
    assert _model_class_for(str(base_dir), {}) is Wav2Vec2BertModel
    assert _model_class_for(str(tmp_path), {"local_files_only": True}) is Wav2Vec2BertModel


# --- Adapter attention through the Pallas kernel -------------------------------------------


def _kernel_on_cpu(
    qkv: torch.Tensor,
    e_pair: torch.Tensor,
    num_heads: int,
    left: int,
    right: int,
    *,
    lengths: torch.Tensor | None = None,
    tile_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Stand-in for ``fused_qkv_attention`` that runs the same Pallas kernel in interpret mode."""
    import math

    import jax.numpy as jnp
    import numpy as np

    from w2vbert_tpu.kernels.fused_qkv_attention import _pallas_call_fused
    from w2vbert_tpu.kernels.relative_attention import select_tile_size

    batch, frames, _ = qkv.shape
    block_q, block_k = select_tile_size(frames, tile_size)
    pad = (-frames) % math.lcm(block_q, block_k)
    if lengths is None:
        lengths = torch.full((batch,), frames, dtype=torch.int32)
    len_tiled = lengths.to(torch.int32)[:, None, None].expand(batch, 8, 128).contiguous()
    out = _pallas_call_fused(
        jnp.asarray(torch.nn.functional.pad(qkv, (0, 0, 0, pad)).float().numpy()),
        jnp.asarray(e_pair.float().numpy()),
        jnp.asarray(len_tiled.numpy()),
        num_heads=num_heads,
        left=left,
        right=right,
        block_q=block_q,
        block_k=block_k,
        interpret=True,
    )
    return torch.from_numpy(np.array(out))[:, :frames].to(qkv.dtype)


def _ctc_head64(**overrides: Any) -> Wav2Vec2BertForCTC:
    """Tiny CTC model with 64-wide heads (what the kernel supports), in FP32."""
    values: dict[str, Any] = dict(
        hidden_size=128, output_hidden_size=128, num_attention_heads=2, num_hidden_layers=1
    )
    values.update(overrides)
    torch.manual_seed(0)
    return Wav2Vec2BertForCTC(_config(**values)).float().eval()


def test_zero_position_kernel_is_plain_attention() -> None:
    from w2vbert_tpu.attention import make_additive_key_mask
    from w2vbert_tpu.kernels.fused_qkv_attention import pair_embedding

    torch.manual_seed(2)
    batch, heads, frames = 2, 4, 300
    q, k, v = (torch.randn(batch, heads, frames, 64) for _ in range(3))
    lengths = torch.tensor([frames, 111], dtype=torch.int32)
    mask = make_additive_key_mask(lengths, frames, torch.float32)
    expected = torch.softmax(q @ k.transpose(-1, -2) / 8.0 + mask, dim=-1) @ v
    expected = expected.transpose(1, 2).reshape(batch, frames, heads * 64)

    def pack(t: torch.Tensor) -> torch.Tensor:
        return t.transpose(1, 2).reshape(batch, frames, heads * 64)

    zeros = pair_embedding(torch.zeros(64 + 8 + 1, 64))
    got = _kernel_on_cpu(
        torch.cat([pack(q), pack(k), pack(v)], dim=-1), zeros, heads, 64, 8, lengths=lengths
    )
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)


def test_adapter_attention_is_swapped_and_matches_hf_on_cpu() -> None:
    ref = _tiny_ctc()
    fast: Any = optimize(copy.deepcopy(ref), backend="auto")
    attn = fast.wav2vec2_bert.adapter.layers[0].self_attn
    assert isinstance(attn, TPUSelfAttention) and attn.adapter_mode
    assert fast._w2vbert_tpu["adapter_attention_modules"] == 1
    # Not on TPU, so the adapter uses Hugging Face attention: identical results.
    features, mask = _inputs()
    with torch.no_grad():
        torch.testing.assert_close(
            fast(features, attention_mask=mask).logits, ref(features, attention_mask=mask).logits
        )


@pytest.mark.parametrize("kwargs", [{"adapter": False}, {"backend": "native"}])
def test_adapter_attention_left_alone(kwargs: dict[str, Any]) -> None:
    fast: Any = optimize(_tiny_ctc(), **kwargs)
    assert not isinstance(fast.wav2vec2_bert.adapter.layers[0].self_attn, TPUSelfAttention)
    assert fast._w2vbert_tpu["adapter_attention_modules"] == 0


def test_adapter_attention_can_be_turned_off_again() -> None:
    ref = _tiny_ctc()
    fast: Any = optimize(copy.deepcopy(ref), backend="auto")
    optimize(fast, backend="auto", adapter=False)
    attn = fast.wav2vec2_bert.adapter.layers[0].self_attn
    assert not attn.adapter_mode
    features, mask = _inputs()
    with torch.no_grad():
        torch.testing.assert_close(
            fast(features, attention_mask=mask).logits, ref(features, attention_mask=mask).logits
        )


@pytest.mark.parametrize("padded", [False, True])
def test_adapter_kernel_path_matches_hf(monkeypatch: pytest.MonkeyPatch, padded: bool) -> None:
    """Force the adapter onto the kernel path (interpret mode on CPU) and compare logits."""
    import importlib

    # Not `import ... as`: the kernels package re-exports a function of the same name.
    fused_module = importlib.import_module("w2vbert_tpu.kernels.fused_qkv_attention")

    ref = _ctc_head64()
    fast: Any = optimize(copy.deepcopy(ref), backend="auto")
    calls: list[tuple[int, ...]] = []

    def fake_kernel(qkv: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(tuple(qkv.shape))
        return _kernel_on_cpu(qkv, *args, **kwargs)

    original = TPUSelfAttention._resolve_backend

    def resolve(self: TPUSelfAttention, hidden_states: torch.Tensor) -> Any:
        return "pallas" if self.adapter_mode else original(self, hidden_states)

    monkeypatch.setattr(fused_module, "fused_qkv_attention", fake_kernel)
    monkeypatch.setattr(TPUSelfAttention, "_resolve_backend", resolve)

    torch.manual_seed(3)
    features = torch.randn(2, 61, 16)
    mask = torch.ones(2, 61, dtype=torch.long)
    if padded:
        mask[1, 37:] = 0
    with torch.no_grad():
        expected = ref(features, attention_mask=mask).logits
        got = fast(features, attention_mask=mask).logits
    assert calls == [(2, 31, 3 * 128)]  # the adapter's attention ran on the kernel path
    lengths = ctc_output_lengths(ref.config, mask.sum(-1))
    for row, length in enumerate(lengths.tolist()):
        torch.testing.assert_close(got[row, :length], expected[row, :length], rtol=1e-4, atol=1e-4)
