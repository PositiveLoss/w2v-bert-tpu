"""Run the fused attention training ops in Pallas interpret mode on CPU (for tests)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _j(t: torch.Tensor) -> Any:
    import jax.numpy as jnp

    if t.dtype in (torch.int32, torch.int64):
        return jnp.asarray(t.to(torch.int32).numpy())
    return jnp.asarray(t.detach().float().numpy())


def _t(a: Any, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.array(a)).to(dtype)


def interpret_train_ops() -> tuple[Any, Any, Any]:
    """Drop-in replacement for ``fused_qkv_backward._train_ops()`` on CPU tensors."""
    from w2vbert_tpu.kernels.fused_qkv_attention import _pallas_call_fused
    from w2vbert_tpu.kernels.fused_qkv_backward import _pallas_call_dkv, _pallas_call_dq

    def fwd(qkv, e_pair, len_tiled, num_heads, left, right, block_q, block_k):
        out, lse = _pallas_call_fused(
            _j(qkv), _j(e_pair), _j(len_tiled), num_heads=num_heads, left=left, right=right,
            block_q=block_q, block_k=block_k, interpret=True, with_lse=True,
        )  # fmt: skip
        return _t(out, qkv.dtype), _t(lse, torch.float32)

    def bwd(call):
        def run(qkv, e_pair, len_tiled, d_out, lse, delta, num_heads, left, right, bq, bk):
            a, b = call(
                _j(qkv), _j(e_pair), _j(len_tiled), _j(d_out), _j(lse), _j(delta),
                num_heads=num_heads, left=left, right=right, block_q=bq, block_k=bk,
                interpret=True,
            )  # fmt: skip
            return _t(a, qkv.dtype), _t(b, qkv.dtype if call is _pallas_call_dkv else torch.float32)

        return run

    return fwd, bwd(_pallas_call_dq), bwd(_pallas_call_dkv)
