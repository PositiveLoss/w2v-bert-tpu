"""Pretrained facebook/w2v-bert-2.0 checkpoint compatibility and TPU numerical parity tests."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2BertModel

from w2vbert_tpu import optimize
from w2vbert_tpu.compiler import configure_compiler, get_compile_metrics, reset_compiler
from w2vbert_tpu.features import extract_features, load_audio

EXAMPLE_WAV = Path("/content/example_1.wav")


def _rel_l2(actual: torch.Tensor, ref: torch.Tensor) -> float:
    a = actual.float().cpu()
    b = ref.float().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def test_pretrained_config_and_state_dict_parity() -> None:
    try:
        base = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", local_files_only=True)
    except OSError:
        pytest.skip("facebook/w2v-bert-2.0 is not in local Hugging Face cache")

    assert len(base.encoder.layers) == 24
    assert sum(p.numel() for p in base.parameters()) == 580_493_120
    assert base.config.hidden_size == 1024
    assert base.config.num_attention_heads == 16
    assert base.config.left_max_position_embeddings == 64
    assert base.config.right_max_position_embeddings == 8
    assert base.config.conv_depthwise_kernel_size == 31

    opt = optimize(copy.deepcopy(base), backend="pallas", prepack_qkv=True)
    sd_base = base.state_dict()
    sd_opt = opt.state_dict()
    assert list(sd_base.keys()) == list(sd_opt.keys())
    for key in sd_base:
        torch.testing.assert_close(sd_opt[key], sd_base[key], rtol=0, atol=0)


def test_pretrained_real_audio_and_layers_tpu() -> None:
    if os.environ.get("W2VBERT_RUN_TPU_TESTS", "0") != "1":
        pytest.skip("Set W2VBERT_RUN_TPU_TESTS=1 to run TPU checkpoint tests")
    if not EXAMPLE_WAV.exists():
        pytest.skip(f"{EXAMPLE_WAV} not found")
    import torch_tpu  # noqa: F401

    if not torch.accelerator.is_available():
        pytest.skip("No TPU accelerator available")

    try:
        base = Wav2Vec2BertModel.from_pretrained(
            "facebook/w2v-bert-2.0", local_files_only=True
        ).eval()
    except OSError:
        pytest.skip("facebook/w2v-bert-2.0 is not in local Hugging Face cache")

    base.config.layerdrop = 0.0
    base.config.mask_time_prob = 0.0
    base.config.mask_feature_prob = 0.0

    audio, sr = load_audio(EXAMPLE_WAV)
    feats = extract_features(audio, sampling_rate=sr, dtype=torch.float32)

    with torch.no_grad():
        ref_states = base(
            input_features=feats["input_features"],
            attention_mask=feats["attention_mask"],
            output_hidden_states=True,
        ).hidden_states

    x_tpu = feats["input_features"].to(device="tpu", dtype=torch.bfloat16)
    m_tpu = feats["attention_mask"].to(device="tpu")

    configure_compiler(recompile_limit=128, fail_on_recompile_limit=True)
    # backend="eager" (unmodified HF, BF16) runs first and sets the per-layer BF16 error floor.
    hf_bf16_layer_err: list[float] = []
    hf_bf16_layer_cos: list[float] = []
    for backend in ("eager", "native", "pallas", "auto"):
        reset_compiler(clear_counters=True)
        model = optimize(
            copy.deepcopy(base).to(device="tpu", dtype=torch.bfloat16),
            backend=backend,  # type: ignore[arg-type]
            compile=(backend != "eager"),
        )
        with torch.inference_mode():
            out = model(
                input_features=x_tpu,
                attention_mask=m_tpu,
                output_hidden_states=True,
            )
            torch.accelerator.synchronize()
        assert out.last_hidden_state.device.type == "tpu"
        assert torch.isfinite(out.last_hidden_state.float().cpu()).all()
        if backend != "eager":
            metrics = get_compile_metrics()
            assert metrics["graph_break_count"] == 0

        # Acceptance criteria on the final 24-layer output vs CPU FP32 Hugging Face.
        got = out.last_hidden_state.float().cpu()
        ref = ref_states[-1].float()
        max_abs = float((got - ref).abs().max())
        final_cos = float(F.cosine_similarity(got.flatten(), ref.flatten(), dim=0))
        print(
            f"real-audio backend={backend}: max_abs={max_abs:.4e} "
            f"rel_l2={_rel_l2(got, ref):.4e} cos={final_cos:.6f}"
        )
        assert final_cos >= 0.999, f"backend={backend} cos={final_cos:.6f}"
        if backend != "eager":
            # backend="eager" is the unmodified HF BF16 path (measured max_abs ~0.165 on this
            # clip); the 0.15 bound is the acceptance criterion for the optimized backends.
            assert max_abs <= 1.5e-1, f"backend={backend} max_abs={max_abs:.4e}"

        worst_err, worst_cos = 0.0, 1.0
        for idx, (got_h, ref_h) in enumerate(zip(out.hidden_states, ref_states)):
            err = _rel_l2(got_h, ref_h)
            cos = float(
                F.cosine_similarity(
                    got_h.float().cpu().flatten(), ref_h.float().flatten(), dim=0
                ).item()
            )
            worst_err, worst_cos = max(worst_err, err), min(worst_cos, cos)
            if backend == "eager":
                hf_bf16_layer_err.append(err)
                hf_bf16_layer_cos.append(cos)
                continue
            # No worse than Hugging Face's own BF16 computation of the same layer.
            err_bound = max(2.5e-2, 1.25 * hf_bf16_layer_err[idx])
            cos_bound = min(0.999, hf_bf16_layer_cos[idx] - 1e-3)
            assert err <= err_bound, f"backend={backend} layer={idx} rel_l2={err:.4e}"
            assert cos >= cos_bound, f"backend={backend} layer={idx} cos={cos:.6f}"
        print(
            f"real-audio backend={backend}: worst layer rel_l2={worst_err:.4e} cos={worst_cos:.6f}"
        )


def test_pretrained_mixed_length_padded_batch_tpu() -> None:
    if os.environ.get("W2VBERT_RUN_TPU_TESTS", "0") != "1":
        pytest.skip("Set W2VBERT_RUN_TPU_TESTS=1 to run TPU checkpoint tests")
    import torch_tpu  # noqa: F401

    if not torch.accelerator.is_available():
        pytest.skip("No TPU accelerator available")

    try:
        base = Wav2Vec2BertModel.from_pretrained(
            "facebook/w2v-bert-2.0", local_files_only=True
        ).eval()
    except OSError:
        pytest.skip("facebook/w2v-bert-2.0 is not in local Hugging Face cache")

    base.config.layerdrop = 0.0
    base.config.mask_time_prob = 0.0
    base.config.mask_feature_prob = 0.0

    torch.manual_seed(0)
    frames = 320
    x = torch.randn(4, frames, 160, dtype=torch.float32)
    mask = torch.zeros(4, frames, dtype=torch.long)
    for row, length in enumerate((320, 250, 73, 1)):
        mask[row, :length] = 1

    with torch.no_grad():
        ref_out = base(input_features=x, attention_mask=mask).last_hidden_state

    x_tpu = x.to(device="tpu", dtype=torch.bfloat16)
    m_tpu = mask.to(device="tpu")
    valid = mask.bool().unsqueeze(-1).expand_as(ref_out)

    configure_compiler(recompile_limit=128, fail_on_recompile_limit=True)
    # Unmodified HF in BF16 on the TPU sets the error floor for this input.
    floor_model = copy.deepcopy(base).to(device="tpu", dtype=torch.bfloat16)
    with torch.inference_mode():
        floor_out = floor_model(input_features=x_tpu, attention_mask=m_tpu).last_hidden_state
    hf_bf16_rel = _rel_l2(floor_out.float().cpu()[valid], ref_out[valid])
    del floor_model
    bound = max(2.5e-2, 1.25 * hf_bf16_rel)
    for backend in ("native", "pallas"):
        reset_compiler(clear_counters=True)
        model = optimize(
            copy.deepcopy(base).to(device="tpu", dtype=torch.bfloat16),
            backend=backend,  # type: ignore[arg-type]
            compile=True,
        )
        with torch.inference_mode():
            out = model(input_features=x_tpu, attention_mask=m_tpu).last_hidden_state
            torch.accelerator.synchronize()
        assert torch.isfinite(out.float().cpu()).all()
        rel = _rel_l2(out.float().cpu()[valid], ref_out[valid])
        print(f"mixed-length backend={backend}: rel_l2={rel:.4e} (HF BF16 {hf_bf16_rel:.4e})")
        assert rel <= bound, f"backend={backend} mixed-length rel_l2={rel:.4e} > {bound:.4e}"
