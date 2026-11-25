"""TPU runtime, compiler, Pallas verifier contract, and error-handling tests."""

from __future__ import annotations

import os

import pytest
import torch
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

from w2vbert_tpu import optimize
from w2vbert_tpu.compiler import configure_compiler, get_compile_metrics, reset_compiler
from w2vbert_tpu.features import extract_features


def test_configure_compiler_limits() -> None:
    info = configure_compiler(recompile_limit=128, accumulated_recompile_limit=1024)
    assert info["recompile_limit"] >= 128
    assert info["accumulated_recompile_limit"] >= 1024
    assert info["fail_on_recompile_limit_hit"] is True
    assert info["suppress_errors"] is False


def test_pallas_jax_op_verifier_contracts() -> None:
    import torch_tpu  # noqa: F401
    from torch.tpu.pallas import verifier

    from w2vbert_tpu.kernels.relative_attention import _get_tpu_pallas_ops

    unmasked_op, masked_op = _get_tpu_pallas_ops()
    q = torch.randn(2, 4, 256, 64, dtype=torch.bfloat16)
    e = torch.randn(128, 64, dtype=torch.bfloat16)
    lengths = torch.full((2, 8, 128), 200, dtype=torch.int32)

    opts = verifier.VerificationOptions(tpu_version=torch.tpu.TpuVersion.V5E)
    rep_u = verifier.verify(
        unmasked_op,
        (q, q, q, e, 64, 8, 128, 128),
        mode=verifier.VerificationMode.STATIC,
        options=opts,
    )
    assert rep_u.passed, rep_u.summary()

    rep_m = verifier.verify(
        masked_op,
        (q, q, q, e, lengths, 64, 8, 128, 128),
        mode=verifier.VerificationMode.STATIC,
        options=opts,
    )
    assert rep_m.passed, rep_m.summary()


def test_error_handling_for_unsupported_inputs() -> None:
    with pytest.raises(ValueError, match="waves sequence must not be empty"):
        extract_features([])
    with pytest.raises(ValueError, match="Unsupported backend"):
        cfg = Wav2Vec2BertConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            feature_projection_input_dim=16,
            conv_depthwise_kernel_size=5,
        )
        optimize(Wav2Vec2BertModel(cfg), backend="triton")  # type: ignore[arg-type]


def test_tpu_device_execution_and_no_graph_breaks() -> None:
    if os.environ.get("W2VBERT_RUN_TPU_TESTS", "0") != "1":
        pytest.skip("Set W2VBERT_RUN_TPU_TESTS=1 to run live TPU integration tests")
    import torch_tpu  # noqa: F401

    assert torch.accelerator.is_available()
    assert torch.accelerator.current_accelerator().type == "tpu"

    cfg = Wav2Vec2BertConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        feature_projection_input_dim=160,
        conv_depthwise_kernel_size=31,
        position_embeddings_type="relative_key",
        left_max_position_embeddings=64,
        right_max_position_embeddings=8,
        layerdrop=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        conformer_conv_dropout=0.0,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
    )
    configure_compiler(recompile_limit=128, fail_on_recompile_limit=True)
    reset_compiler(clear_counters=True)
    model = optimize(
        Wav2Vec2BertModel(cfg).eval().to(device="tpu", dtype=torch.bfloat16),
        backend="auto",
        compile=True,
    )
    for frames in (128, 256):
        x = torch.randn(2, frames, 160, device="tpu", dtype=torch.bfloat16)
        mask = torch.ones(2, frames, device="tpu", dtype=torch.long)
        with torch.inference_mode():
            out = model(input_features=x, attention_mask=mask).last_hidden_state
            torch.accelerator.synchronize()
        assert out.device.type == "tpu"
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float().cpu()).all()

    metrics = get_compile_metrics()
    assert metrics["graph_break_count"] == 0
    assert metrics["unique_graphs"] >= 2
