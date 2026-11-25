"""Training: gradients of the optimized model (native path and Pallas backward) vs Hugging Face."""

from __future__ import annotations

import copy
import importlib
from typing import Any

import pytest
import torch
from _interpret import interpret_train_ops
from test_ctc import _config, _inputs
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertModel

from w2vbert_tpu import optimize
from w2vbert_tpu.model import TPUSelfAttention


def _grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _ctc_loss(model: Any, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(7)
    labels = torch.randint(2, model.config.vocab_size, (features.size(0), 5))
    labels[2, 3:] = -100  # padded label positions
    return model(features, attention_mask=mask, labels=labels).loss


@pytest.mark.parametrize("ctc", [False, True])
def test_native_training_gradients_match_hf(ctc: bool) -> None:
    torch.manual_seed(0)
    cls = Wav2Vec2BertForCTC if ctc else Wav2Vec2BertModel
    ref: Any = cls(_config(ctc_loss_reduction="sum", add_adapter=ctc)).double().train()
    fast: Any = optimize(copy.deepcopy(ref), backend="auto").train()
    features, mask = _inputs()
    if ctc:
        _ctc_loss(ref, features, mask).backward()
        _ctc_loss(fast, features, mask).backward()
    else:
        weight = torch.randn(3, 41, 32, dtype=torch.float64)
        valid = mask[..., None].to(torch.float64)
        (ref(features, attention_mask=mask).last_hidden_state * weight * valid).sum().backward()
        (fast(features, attention_mask=mask).last_hidden_state * weight * valid).sum().backward()
    expected, got = _grads(ref), _grads(fast)
    assert got.keys() == expected.keys()
    assert any("distance_embedding" in name for name in got)
    for name in expected:
        torch.testing.assert_close(got[name], expected[name], rtol=1e-9, atol=1e-10, msg=name)


def test_pallas_backward_training_gradients_match_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Encoder and adapter forced onto the Pallas kernels (interpret mode on CPU), FP32."""
    backward_module = importlib.import_module("w2vbert_tpu.kernels.fused_qkv_backward")
    calls = {"fwd": 0, "dq": 0, "dkv": 0}
    fwd, dq, dkv = interpret_train_ops()

    def counted(name: str, fn: Any) -> Any:
        def run(*args: Any) -> Any:
            calls[name] += 1
            return fn(*args)

        return run

    monkeypatch.setattr(
        backward_module,
        "_train_ops",
        lambda: (counted("fwd", fwd), counted("dq", dq), counted("dkv", dkv)),
    )
    monkeypatch.setattr(
        TPUSelfAttention, "_resolve_backend_for_inference", lambda self, hidden: "pallas"
    )

    torch.manual_seed(0)
    config = _config(
        hidden_size=128,
        output_hidden_size=128,
        num_attention_heads=2,
        num_hidden_layers=2,
        ctc_loss_reduction="sum",
    )
    ref: Any = Wav2Vec2BertForCTC(config).float().train()
    fast: Any = optimize(copy.deepcopy(ref), backend="auto").train()
    torch.manual_seed(3)
    features = torch.randn(3, 61, 16)
    mask = torch.ones(3, 61, dtype=torch.long)
    mask[1, 40:] = 0
    mask[2, 9:] = 0
    _ctc_loss(ref, features, mask).backward()
    _ctc_loss(fast, features, mask).backward()
    # 2 encoder layers + 1 adapter layer, one forward and one backward each.
    assert calls == {"fwd": 3, "dq": 3, "dkv": 3}
    expected, got = _grads(ref), _grads(fast)
    assert got.keys() == expected.keys()
    for name in expected:
        # Relative to each parameter's largest gradient; the absolute floor covers gradients
        # that are zero in exact arithmetic (linear_k.bias: softmax ignores a per-row constant).
        diff = (got[name] - expected[name]).abs().max().item()
        bound = 1e-4 * expected[name].abs().max().item() + 1e-8
        assert diff <= bound, f"{name}: max diff {diff:.2e} > {bound:.2e}"


def test_eval_refreshes_caches_in_place() -> None:
    torch.manual_seed(0)
    model: Any = Wav2Vec2BertForCTC(
        _config(hidden_size=128, output_hidden_size=128, num_attention_heads=2)
    ).eval()
    optimize(model, backend="auto")
    attn = model.wav2vec2_bert.encoder.layers[0].self_attn
    cached = attn.__dict__["_qkv_cache"]["weight"]
    pair = attn.__dict__["_emb_cache"]["pair"]
    with torch.no_grad():  # an optimizer step: in-place updates
        attn.linear_k.weight.add_(1.0)
        attn.distance_embedding.weight.mul_(2.0)
    model.train()
    model.eval()
    assert attn.__dict__["_qkv_cache"]["weight"] is cached
    assert attn.__dict__["_emb_cache"]["pair"] is pair
    expected = torch.cat([attn.linear_q.weight, attn.linear_k.weight, attn.linear_v.weight])
    torch.testing.assert_close(cached, expected)


def test_gradient_path_ignores_stale_prepacked_cache() -> None:
    torch.manual_seed(0)
    ref: Any = Wav2Vec2BertModel(_config(add_adapter=False)).double()
    fast: Any = optimize(copy.deepcopy(ref), backend="native", prepack_qkv=True)
    for model in (ref, fast):
        with torch.no_grad():
            model.encoder.layers[0].self_attn.linear_v.weight.add_(0.5)
    features, mask = _inputs()
    out_ref = ref(features, attention_mask=mask).last_hidden_state
    out_fast = fast(features, attention_mask=mask).last_hidden_state  # grad enabled
    valid = mask[..., None].to(torch.float64)
    torch.testing.assert_close(out_fast * valid, out_ref * valid, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize(
    ("frames", "tile", "lengths"),
    [
        (250, (128, 128), [250, 90]),
        (384, (128, 256), [384, 200]),
        (512, (256, 128), [512, 1]),
        (600, (256, 256), [600, 333]),
    ],
)
def test_pallas_backward_matches_reference(
    monkeypatch: pytest.MonkeyPatch, frames: int, tile: tuple[int, int], lengths: list[int]
) -> None:
    """Kernel level (interpret mode): dq, dk, dv and dE against autograd of the reference."""
    import math

    import torch.nn.functional as F

    from w2vbert_tpu.attention import make_additive_key_mask, relative_key_reference
    from w2vbert_tpu.kernels.fused_qkv_attention import pair_embedding

    backward_module = importlib.import_module("w2vbert_tpu.kernels.fused_qkv_backward")
    monkeypatch.setattr(backward_module, "_train_ops", interpret_train_ops)
    torch.manual_seed(frames)
    batch, heads, left, right = len(lengths), 4, 64, 8
    q, k, v = (torch.randn(batch, heads, frames, 64, requires_grad=True) for _ in range(3))
    emb = (0.5 * torch.randn(left + right + 1, 64)).requires_grad_()
    weight = torch.randn(batch, heads, frames, 64)
    lens = torch.tensor(lengths, dtype=torch.int32)
    mask = make_additive_key_mask(lens, frames, torch.float32)
    expected = relative_key_reference(q, k, v, emb, left, right, attention_mask=mask)
    (expected * weight).sum().backward()
    expected_grads = [t.grad.clone() for t in (q, k, v, emb)]  # type: ignore[union-attr]
    for t in (q, k, v, emb):
        t.grad = None

    def pack(t: torch.Tensor) -> torch.Tensor:
        return t.transpose(1, 2).reshape(batch, frames, heads * 64)

    qkv = F.pad(torch.cat([pack(q), pack(k), pack(v)], -1), (0, 0, 0, (-frames) % math.lcm(*tile)))
    len_tiled = lens[:, None, None].expand(batch, 8, 128).contiguous()
    out = backward_module.fused_qkv_attention_train(
        qkv, pair_embedding(emb), len_tiled, heads, left, right, *tile
    )[:, :frames]
    out = out.view(batch, frames, heads, 64).transpose(1, 2)
    (out * weight).sum().backward()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    for name, tensor, grad in zip(("dq", "dk", "dv", "dE"), (q, k, v, emb), expected_grads):
        error = ((tensor.grad - grad).abs().max() / grad.abs().max()).item()  # type: ignore[operator]
        assert error < 1e-4, f"{name}: relative error {error:.2e}"
