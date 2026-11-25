"""Numerical correctness tests for relative-key attention variants on CPU and TPU."""

from __future__ import annotations

import itertools
import os

import pytest
import torch
import torch.nn.functional as F

from w2vbert_tpu.attention import (
    extract_key_lengths,
    make_additive_key_mask,
    native_relative_key_attention,
    relative_key_reference,
)

BOUNDARY_LENGTHS = (
    1,
    7,
    8,
    9,
    63,
    64,
    65,
    72,
    73,
    74,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
)

NATIVE_VARIANTS = ("sdpa_gather", "sdpa_index", "explicit_gather", "explicit_index")


def _make_inputs(
    batch: int,
    heads: int,
    frames: int,
    dim: int,
    left: int,
    right: int,
    *,
    dtype: torch.dtype,
    device: str | torch.device = "cpu",
    seed: int = 0,
):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, frames, dim, generator=gen, dtype=torch.float64).to(
        device=device, dtype=dtype
    )
    k = torch.randn(batch, heads, frames, dim, generator=gen, dtype=torch.float64).to(
        device=device, dtype=dtype
    )
    v = torch.randn(batch, heads, frames, dim, generator=gen, dtype=torch.float64).to(
        device=device, dtype=dtype
    )
    emb = torch.randn(left + right + 1, dim, generator=gen, dtype=torch.float64).to(
        device=device, dtype=dtype
    )
    return q, k, v, emb


def _error_metrics(actual: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    a = actual.float().cpu()
    b = ref.float().cpu()
    diff = a - b
    return {
        "max_abs": float(diff.abs().max().item()),
        "mae": float(diff.abs().mean().item()),
        "rmse": float(diff.pow(2).mean().sqrt().item()),
        "rel_l2": float((diff.norm() / b.norm().clamp_min(1e-12)).item()),
        "cosine": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()),
        "nonfinite": int((~torch.isfinite(a)).sum().item()),
    }


@pytest.mark.parametrize("variant", NATIVE_VARIANTS)
@pytest.mark.parametrize("frames", BOUNDARY_LENGTHS)
def test_native_attention_matches_reference_boundary_lengths_fp64(
    variant: str, frames: int
) -> None:
    q, k, v, emb = _make_inputs(2, 2, frames, 16, 64, 8, dtype=torch.float64, seed=frames)
    ref = relative_key_reference(q, k, v, emb, 64, 8)
    out = native_relative_key_attention(q, k, v, emb, 64, 8, variant=variant)  # type: ignore[arg-type]
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("variant", NATIVE_VARIANTS)
@pytest.mark.parametrize(
    "frames,clip",
    list(itertools.product((1, 4, 9, 64, 65, 129), ((1, 1), (2, 1), (3, 2), (1, 4)))),
)
def test_native_attention_small_clipping_bounds(
    variant: str, frames: int, clip: tuple[int, int]
) -> None:
    left, right = clip
    q, k, v, emb = _make_inputs(
        1, 2, frames, 8, left, right, dtype=torch.float64, seed=frames + left
    )
    ref = relative_key_reference(q, k, v, emb, left, right)
    out = native_relative_key_attention(q, k, v, emb, left, right, variant=variant)  # type: ignore[arg-type]
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("variant", NATIVE_VARIANTS)
def test_native_attention_with_key_padding_lengths(variant: str) -> None:
    frames = 160
    q, k, v, emb = _make_inputs(3, 4, frames, 16, 64, 8, dtype=torch.float64, seed=123)
    lengths = torch.tensor([160, 73, 19], dtype=torch.int32)
    mask = make_additive_key_mask(lengths, frames, torch.float64)
    ref = relative_key_reference(q, k, v, emb, 64, 8, attention_mask=mask)
    out = native_relative_key_attention(q, k, v, emb, 64, 8, lengths=lengths, variant=variant)  # type: ignore[arg-type]
    torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-11)


def test_extract_key_lengths_supports_2d_and_4d_masks() -> None:
    frames = 64
    conv_mask = torch.zeros(3, frames, dtype=torch.long)
    conv_mask[0, :64] = 1
    conv_mask[1, :37] = 1
    conv_mask[2, :5] = 1
    lengths_2d = extract_key_lengths(None, frames, conv_attention_mask=conv_mask)
    assert lengths_2d is not None
    assert lengths_2d.dtype == torch.int32
    assert torch.equal(lengths_2d, torch.tensor([64, 37, 5], dtype=torch.int32))

    add_mask = make_additive_key_mask(lengths_2d, frames, torch.bfloat16).expand(
        3, 1, frames, frames
    )
    lengths_4d = extract_key_lengths(add_mask, frames)
    assert lengths_4d is not None
    assert torch.equal(lengths_4d, lengths_2d)


def test_pallas_tpu_v5e_static_lowering() -> None:
    import jax
    import jax.numpy as jnp
    import torch_tpu  # noqa: F401
    from torch_tpu._internal.pallas.verify import findings, fx_verifier

    from w2vbert_tpu.kernels.relative_attention import (
        pallas_relative_key_attention_masked_jax,
        pallas_relative_key_attention_unmasked_jax,
    )

    for bq, bk in ((128, 128), (256, 256)):
        seq_len = bq * 2
        q_spec = jax.ShapeDtypeStruct((2, 16, seq_len, 64), jnp.bfloat16)
        e_spec = jax.ShapeDtypeStruct((128, 64), jnp.bfloat16)
        len_spec = jax.ShapeDtypeStruct((2, 8, 128), jnp.int32)

        rep_unmasked = fx_verifier.verify_tpu_lowering(
            jax.jit(
                lambda q, k, v, e: pallas_relative_key_attention_unmasked_jax(
                    q, k, v, e, left=64, right=8, block_q=bq, block_k=bk
                )
            ),
            (q_spec, q_spec, q_spec, e_spec),
            tpu_version=findings.TpuVersion.V5E,
        )
        assert rep_unmasked.passed, rep_unmasked.summary()

        rep_masked = fx_verifier.verify_tpu_lowering(
            jax.jit(
                lambda q, k, v, e, lens: pallas_relative_key_attention_masked_jax(
                    q, k, v, e, lens, left=64, right=8, block_q=bq, block_k=bk
                )
            ),
            (q_spec, q_spec, q_spec, e_spec, len_spec),
            tpu_version=findings.TpuVersion.V5E,
        )
        assert rep_masked.passed, rep_masked.summary()


@pytest.mark.parametrize(
    ("frames", "tile"),
    [(f, (128, 128)) for f in (1, 64, 65, 72, 73, 128, 129, 250, 257)]
    + [
        (257, (256, 256)),
        (600, (256, 256)),
        (600, (256, 128)),
        (600, (128, 256)),
        (1100, (512, 256)),
    ],
)
def test_pallas_cpu_interpret_matches_reference(frames: int, tile: tuple[int, int]) -> None:
    import math

    import jax.numpy as jnp
    import numpy as np

    from w2vbert_tpu.kernels.relative_attention import _pallas_call_relative_attention

    q, k, v, emb = _make_inputs(2, 2, frames, 64, 64, 8, dtype=torch.float32, seed=frames)
    lengths = torch.tensor([frames, max(1, frames // 2)], dtype=torch.int32)
    mask = make_additive_key_mask(lengths, frames, torch.float32)
    ref = relative_key_reference(q, k, v, emb, 64, 8, attention_mask=mask)

    block_q, block_k = tile
    tile_lcm = math.lcm(block_q, block_k)
    pad_t = ((frames + tile_lcm - 1) // tile_lcm) * tile_lcm - frames
    q_scaled = F.pad(q / math.sqrt(64), (0, 0, 0, pad_t))
    k_pad = F.pad(k, (0, 0, 0, pad_t))
    v_pad = F.pad(v, (0, 0, 0, pad_t))
    emb_pad = F.pad(emb, (0, 0, 0, 128 - 73))
    len_tiled = lengths[:, None, None].expand(2, 8, 128).contiguous()

    out_jax = _pallas_call_relative_attention(
        jnp.asarray(q_scaled.numpy()),
        jnp.asarray(k_pad.numpy()),
        jnp.asarray(v_pad.numpy()),
        jnp.asarray(emb_pad.numpy()),
        jnp.asarray(len_tiled.numpy()),
        left=64,
        right=8,
        block_q=block_q,
        block_k=block_k,
        interpret=True,
    )
    out = torch.from_numpy(np.array(out_jax))[:, :, :frames, :]
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_pallas_cpu_interpret_fully_masked_row_no_nan() -> None:
    import math

    import jax.numpy as jnp
    import numpy as np

    from w2vbert_tpu.kernels.relative_attention import _pallas_call_relative_attention

    frames = 128
    q, k, v, emb = _make_inputs(2, 2, frames, 64, 64, 8, dtype=torch.float32, seed=99)
    lengths = torch.tensor([128, 0], dtype=torch.int32)
    emb_pad = F.pad(emb, (0, 0, 0, 128 - 73))
    len_tiled = lengths[:, None, None].expand(2, 8, 128).contiguous()

    out_jax = _pallas_call_relative_attention(
        jnp.asarray((q / math.sqrt(64)).numpy()),
        jnp.asarray(k.numpy()),
        jnp.asarray(v.numpy()),
        jnp.asarray(emb_pad.numpy()),
        jnp.asarray(len_tiled.numpy()),
        left=64,
        right=8,
        block_q=128,
        block_k=128,
        interpret=True,
    )
    out = torch.from_numpy(np.array(out_jax))
    assert torch.isfinite(out).all()
    assert torch.equal(out[1], torch.zeros_like(out[1]))


def test_tpu_native_and_pallas_attention_parity() -> None:
    if os.environ.get("W2VBERT_RUN_TPU_TESTS", "0") != "1":
        pytest.skip("Set W2VBERT_RUN_TPU_TESTS=1 to run TPU attention tests")
    import torch_tpu  # noqa: F401

    from w2vbert_tpu.kernels.relative_attention import pallas_relative_key_attention

    if not torch.accelerator.is_available():
        pytest.skip("No TPU accelerator available")

    for frames in (
        1,
        7,
        8,
        9,
        63,
        64,
        65,
        72,
        73,
        74,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        1500,
        1536,
    ):
        q32, k32, v32, e32 = _make_inputs(
            2, 16, frames, 64, 64, 8, dtype=torch.float32, seed=frames
        )
        lengths_cpu = torch.tensor([frames, max(1, frames // 2)], dtype=torch.int32)
        mask32 = make_additive_key_mask(lengths_cpu, frames, torch.float32)
        ref = relative_key_reference(q32, k32, v32, e32, 64, 8, attention_mask=mask32)

        q_tpu = q32.to(device="tpu", dtype=torch.bfloat16)
        k_tpu = k32.to(device="tpu", dtype=torch.bfloat16)
        v_tpu = v32.to(device="tpu", dtype=torch.bfloat16)
        e_tpu = e32.to(device="tpu", dtype=torch.bfloat16)
        len_tpu = lengths_cpu.to(device="tpu")

        for variant in NATIVE_VARIANTS:
            if frames > 513 and variant != "explicit_gather":
                continue
            out = native_relative_key_attention(
                q_tpu,
                k_tpu,
                v_tpu,
                e_tpu,
                64,
                8,
                lengths=len_tpu,
                variant=variant,  # type: ignore[arg-type]
            )
            torch.accelerator.synchronize()
            m = _error_metrics(out, ref)
            assert m["nonfinite"] == 0
            assert m["rel_l2"] < 2e-2, f"{variant} T={frames} rel_l2={m['rel_l2']}"
            assert m["cosine"] > 0.9995, f"{variant} T={frames} cos={m['cosine']}"

        out_pallas = pallas_relative_key_attention(
            q_tpu, k_tpu, v_tpu, e_tpu, 64, 8, lengths=len_tpu
        )
        torch.accelerator.synchronize()
        mp = _error_metrics(out_pallas, ref)
        assert mp["nonfinite"] == 0
        assert mp["rel_l2"] < 2e-2, f"pallas T={frames} rel_l2={mp['rel_l2']}"
        assert mp["cosine"] > 0.9995, f"pallas T={frames} cos={mp['cosine']}"


def test_pallas_kernel_fingerprint_is_stable_and_identifier_safe() -> None:
    from w2vbert_tpu.kernels.relative_attention import _kernel_fingerprint

    tag = _kernel_fingerprint()
    assert tag == _kernel_fingerprint()
    assert len(tag) == 12 and tag.isalnum()


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(1, "explicit_gather"), (256, "explicit_gather"), (257, "sdpa_gather"), (512, "sdpa_gather")]
    + [(513, "explicit_index"), (1500, "explicit_index"), (6144, "explicit_index")],
)
def test_auto_native_variant_avoids_long_sequence_gather(frames: int, expected: str) -> None:
    from w2vbert_tpu.attention import resolve_native_variant

    assert resolve_native_variant("auto", frames) == expected
    assert resolve_native_variant("sdpa_index", frames) == "sdpa_index"


@pytest.mark.parametrize(
    ("frames", "tile"),
    [(1, (128, 128)), (128, (128, 128)), (129, (256, 256)), (256, (256, 256))]
    + [(257, (512, 512)), (1500, (512, 512)), (6144, (512, 512))],
)
def test_select_tile_size_policy(frames: int, tile: tuple[int, int]) -> None:
    from w2vbert_tpu.kernels.relative_attention import select_tile_size

    assert select_tile_size(frames) == tile
    assert select_tile_size(frames, (256, 128)) == (256, 128)


@pytest.mark.parametrize(
    ("frames", "tile", "masked"),
    [
        (128, (128, 128), False),
        (256, (128, 128), True),
        (250, (128, 128), True),
        (600, (256, 256), True),
        (512, (256, 128), False),
        (1024, (512, 512), True),
    ],
)
def test_fused_qkv_interpret_matches_reference(
    frames: int, tile: tuple[int, int], masked: bool
) -> None:
    import math

    import jax.numpy as jnp
    import numpy as np

    from w2vbert_tpu.kernels.fused_qkv_attention import _pallas_call_fused, pair_embedding

    heads = 4
    q, k, v, emb = _make_inputs(2, heads, frames, 64, 64, 8, dtype=torch.float32, seed=frames)
    lengths = torch.tensor([frames, max(1, frames // 3)], dtype=torch.int32)
    if not masked:
        lengths = torch.tensor([frames, frames], dtype=torch.int32)
    mask = make_additive_key_mask(lengths, frames, torch.float32)
    ref = relative_key_reference(q, k, v, emb, 64, 8, attention_mask=mask)
    ref = ref.transpose(1, 2).reshape(2, frames, heads * 64)

    def pack(t: torch.Tensor) -> torch.Tensor:
        return t.transpose(1, 2).reshape(2, frames, heads * 64)

    qkv = torch.cat([pack(q), pack(k), pack(v)], dim=-1)
    pad_t = (-frames) % math.lcm(*tile)
    qkv = F.pad(qkv, (0, 0, 0, pad_t))
    len_tiled = lengths[:, None, None].expand(2, 8, 128).contiguous()
    out = _pallas_call_fused(
        jnp.asarray(qkv.numpy()),
        jnp.asarray(pair_embedding(emb).numpy()),
        jnp.asarray(len_tiled.numpy()) if (masked or pad_t) else None,
        num_heads=heads,
        left=64,
        right=8,
        block_q=tile[0],
        block_k=tile[1],
        interpret=True,
    )
    got = torch.from_numpy(np.array(out))[:, :frames]
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
