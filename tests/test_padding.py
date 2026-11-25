"""Tests for static shape bucketing and padding invariance."""

from __future__ import annotations

import copy
import os

import pytest
import torch
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

from w2vbert_tpu import optimize
from w2vbert_tpu.batching import (
    DEFAULT_BATCH_BUCKETS,
    DEFAULT_FRAME_BUCKETS,
    pad_to_bucket,
    select_batch_bucket,
    select_frame_bucket,
    unpad_output,
)


def test_select_buckets_exact_and_boundaries() -> None:
    assert select_frame_bucket(1) == 128
    assert select_frame_bucket(128) == 128
    assert select_frame_bucket(129) == 256
    assert select_frame_bucket(250) == 256
    assert select_frame_bucket(500) == 512
    assert select_frame_bucket(1000) == 1024
    assert select_frame_bucket(1500) == 1536
    assert select_frame_bucket(3000) == 3072
    assert select_frame_bucket(6144) == 6144

    assert select_batch_bucket(1) == 1
    assert select_batch_bucket(3) == 4
    assert select_batch_bucket(16) == 16
    assert select_batch_bucket(32) == 32


def test_select_buckets_rejects_unsupported_shapes() -> None:
    with pytest.raises(ValueError, match="frames must be >= 1"):
        select_frame_bucket(0)
    with pytest.raises(ValueError, match="exceeds maximum configured frame bucket"):
        select_frame_bucket(DEFAULT_FRAME_BUCKETS[-1] + 1)
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        select_batch_bucket(0)
    with pytest.raises(ValueError, match="exceeds maximum configured batch bucket"):
        select_batch_bucket(DEFAULT_BATCH_BUCKETS[-1] + 1)


@pytest.mark.parametrize(
    "frames,expected_bucket",
    [(250, 256), (500, 512), (1000, 1024)],
)
def test_padding_invariance_on_valid_positions_cpu(frames: int, expected_bucket: int) -> None:
    config = Wav2Vec2BertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        feature_projection_input_dim=16,
        conv_depthwise_kernel_size=5,
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
    torch.manual_seed(0)
    model = optimize(Wav2Vec2BertModel(config).double().eval(), backend="native")
    x = torch.randn(2, frames, 16, dtype=torch.float64)
    mask = torch.ones(2, frames, dtype=torch.long)
    mask[1, frames - 30 :] = 0

    bucketed = pad_to_bucket(x, mask, batch_buckets=DEFAULT_BATCH_BUCKETS)
    assert bucketed.bucket_frames == expected_bucket
    assert bucketed.bucket_batch == 2

    with torch.no_grad():
        unpadded_out = model(x, attention_mask=mask).last_hidden_state
        bucket_out = model(
            bucketed.input_features,
            attention_mask=bucketed.attention_mask,
        ).last_hidden_state
    sliced_out = unpad_output(bucket_out, bucketed.original_batch, bucketed.original_frames)
    valid = mask.bool().unsqueeze(-1).expand_as(unpadded_out)
    torch.testing.assert_close(sliced_out[valid], unpadded_out[valid], rtol=1e-10, atol=1e-10)


def test_padding_invariance_tpu_bf16() -> None:
    if os.environ.get("W2VBERT_RUN_TPU_TESTS", "0") != "1":
        pytest.skip("Set W2VBERT_RUN_TPU_TESTS=1 to run TPU padding invariance tests")
    import torch_tpu  # noqa: F401

    from w2vbert_tpu import load

    if not torch.accelerator.is_available():
        pytest.skip("No TPU accelerator available")

    from transformers import Wav2Vec2BertModel

    model = load("facebook/w2v-bert-2.0", local_files_only=True, backend="pallas", compile=True)
    torch.manual_seed(0)
    x = torch.randn(2, 250, 160).to(device="tpu", dtype=torch.bfloat16)
    mask = torch.ones(2, 250, device="tpu", dtype=torch.long)
    mask[1, 180:] = 0

    bucketed = pad_to_bucket(copy.deepcopy(x), copy.deepcopy(mask))
    assert bucketed.bucket_frames == 256
    with torch.inference_mode():
        direct = model(x, attention_mask=mask).last_hidden_state
        padded = model(
            bucketed.input_features, attention_mask=bucketed.attention_mask
        ).last_hidden_state
        torch.accelerator.synchronize()

    # FP32 CPU reference on the same (BF16-rounded) inputs. T=250 and T=256 are different
    # compiled programs, so their BF16 outputs differ by BF16 noise (~2e-2 even in rows with no
    # padding); the invariant is that bucketing does not degrade accuracy.
    base = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", local_files_only=True).eval()
    with torch.no_grad():
        ref = base(x.float().cpu(), attention_mask=mask.cpu()).last_hidden_state

    sliced = unpad_output(padded, 2, 250).float().cpu()
    valid = mask.bool().cpu().unsqueeze(-1).expand_as(ref)

    def rel(a: torch.Tensor) -> float:
        return float((a[valid] - ref[valid]).norm() / ref[valid].norm())

    direct_err, bucket_err = rel(direct.float().cpu()), rel(sliced)
    assert torch.isfinite(sliced).all()
    assert direct_err < 5e-2, f"direct rel_l2 vs FP32 {direct_err:.4e}"
    assert bucket_err <= 1.25 * direct_err, (
        f"bucketing degraded accuracy: {bucket_err:.4e} vs direct {direct_err:.4e}"
    )
