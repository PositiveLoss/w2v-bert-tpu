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
