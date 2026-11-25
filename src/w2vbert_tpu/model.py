"""TPU-native model integration for Hugging Face ``Wav2Vec2BertModel`` (w2v-BERT 2.0).

Uses in-place class swapping on ``Wav2Vec2BertSelfAttention``, ``Wav2Vec2BertConvolutionModule``,
and ``Wav2Vec2BertEncoder`` so parameter names, ``state_dict()`` keys/values, and ``isinstance``
relationships remain identical to Hugging Face.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from w2vbert_tpu.attention import (
    GATHER_MAX_FRAMES,
    NativeAttentionVariant,
    build_distance_indices,
    build_flat_bucket_indices,
    extract_key_lengths,
    make_additive_key_mask,
    native_relative_key_attention,
    resolve_native_variant,
)
from w2vbert_tpu.compiler import configure_compiler
from w2vbert_tpu.convolution import ConvImpl, TPUConvolutionModule

Backend = Literal["auto", "pallas", "native", "sdpa", "eager"]
CompileStrategy = Literal["whole", "layer"]

_FAST_CLASSES: dict[tuple[type, type], type] = {}

# backend="auto" crossover (TPU v5e, full 24-layer model, benchmark_optimized.md): native
# attention wins for short sequences at small batch; the fused Pallas kernel wins above
# AUTO_PALLAS_MIN_FRAMES, and at T=250 already from batch AUTO_PALLAS_MIN_BATCH.
AUTO_PALLAS_MIN_FRAMES = 256
AUTO_PALLAS_MIN_BATCH = 8


def _auto_prefers_pallas(batch: int, frames: int) -> bool:
    if frames > AUTO_PALLAS_MIN_FRAMES:
        return True
    return frames > 128 and batch >= AUTO_PALLAS_MIN_BATCH


def _fast_class(base: type, fast: type) -> type:
    key = (base, fast)
    if key not in _FAST_CLASSES:
        _FAST_CLASSES[key] = type(f"TPU{base.__name__}", (fast, base), {})
    return _FAST_CLASSES[key]


class TPUSelfAttention(torch.nn.Module):
    """TPU-native relative-key self-attention (class-swapped onto ``Wav2Vec2BertSelfAttention``).

    Preserves all Hugging Face parameter names and ``state_dict()`` values while optionally
    prepacking immutable Q/K/V weights and biases outside PyTorch parameter registration.
    """

    tpu_backend: Backend = "auto"
    native_variant: NativeAttentionVariant = "auto"
    prepack_qkv: bool = False
    fused_qkv: bool = True
    pallas_tile_size: tuple[int, int] | None = None

    def _invalidate_caches(self) -> None:
        self.__dict__.pop("_qkv_cache", None)
        self.__dict__.pop("_emb_cache", None)

    def _apply(self, fn, recurse: bool = True):
        self._invalidate_caches()
        res = super()._apply(fn, recurse=recurse)
        self.prepack()
        return res

    def _load_from_state_dict(self, *args, **kwargs):
        self._invalidate_caches()
        res = super()._load_from_state_dict(*args, **kwargs)
        self.prepack()
        return res

    def _qkv_signature(self) -> tuple[Any, ...] | None:
        projections = (self.linear_q, self.linear_k, self.linear_v)
        if not all(type(m) is torch.nn.Linear for m in projections):
            return None
        has_bias = all(m.bias is not None for m in projections)
        if any(m.bias is not None for m in projections) and not has_bias:
            return None
        sig: list[Any] = [has_bias]
        for m in projections:
            w = m.weight
            sig.extend((id(w), w._version, w.device, w.dtype))
            if has_bias:
                b = m.bias
                sig.extend((id(b), b._version, b.device, b.dtype))
        return tuple(sig)

    def _emb_signature(self) -> tuple[Any, ...]:
        e = self.distance_embedding.weight
        return (id(e), e._version, e.device, e.dtype)

    def prepack(self) -> None:
        """Pack Q/K/V projection weights and padded distance embeddings off the forward path."""
        if not (self.prepack_qkv or self._fused_qkv_supported()):
            self.__dict__.pop("_qkv_cache", None)
        else:
            sig = self._qkv_signature()
            if sig is None:
                self.__dict__.pop("_qkv_cache", None)
            else:
                projections = (self.linear_q, self.linear_k, self.linear_v)
                packed_w = torch.cat([m.weight for m in projections], dim=0).contiguous()
                packed_b = (
                    torch.cat([m.bias for m in projections], dim=0).contiguous() if sig[0] else None
                )
                self.__dict__["_qkv_cache"] = {
                    "sig": sig,
                    "weight": packed_w,
                    "bias": packed_b,
                }

        if getattr(self, "position_embeddings_type", None) == "relative_key":
            emb = self.distance_embedding.weight
            num_buckets = emb.size(0)
            e_pad = ((num_buckets + 127) // 128) * 128 - num_buckets
            padded_emb = F.pad(emb, (0, 0, 0, e_pad)) if e_pad > 0 else emb
            cache = {"sig": self._emb_signature(), "padded": padded_emb.contiguous()}
            if self._fused_qkv_supported():
                from w2vbert_tpu.kernels.fused_qkv_attention import pair_embedding

                cache["pair"] = pair_embedding(emb)
            self.__dict__["_emb_cache"] = cache

    def _fused_qkv_supported(self) -> bool:
        return (
            self.fused_qkv
            and getattr(self, "position_embeddings_type", None) == "relative_key"
            and self.head_size == 64
            and self.num_heads % 2 == 0
        )

    def _fused_qkv_tensors(self) -> tuple[Tensor, Tensor | None, Tensor] | None:
        """Packed ``[3d, d]`` weight/bias and paired embedding for the fused kernel.

        Validated (and rebuilt if stale) only in eager mode; under Dynamo the caches are read
        directly and guarded by tensor identity (rebuilt by ``_apply``/``_load_from_state_dict``).
        """
        qkv_cache = self.__dict__.get("_qkv_cache")
        emb_cache = self.__dict__.get("_emb_cache")
        if not torch.compiler.is_compiling() and (
            qkv_cache is None
            or qkv_cache["sig"] != self._qkv_signature()
            or emb_cache is None
            or emb_cache["sig"] != self._emb_signature()
        ):
            self.prepack()
            qkv_cache = self.__dict__.get("_qkv_cache")
            emb_cache = self.__dict__.get("_emb_cache")
        if qkv_cache is None or emb_cache is None or "pair" not in emb_cache:
            return None
        return qkv_cache["weight"], qkv_cache["bias"], emb_cache["pair"]

    def _eager(self, *args, **kwargs):
        from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
            Wav2Vec2BertSelfAttention,
        )

        return Wav2Vec2BertSelfAttention.forward(self, *args, **kwargs)

    def _native_variant_for(self, resolved: str, frames: int) -> NativeAttentionVariant:
        if resolved == "sdpa":
            return "sdpa_gather" if frames <= GATHER_MAX_FRAMES else "sdpa_index"
        return resolve_native_variant(self.native_variant, frames)

    def _resolve_backend(self, hidden_states: Tensor) -> Literal["pallas", "native", "sdpa"]:
        backend = self.tpu_backend
        if backend == "pallas":
            if hidden_states.device.type != "tpu":
                raise ValueError(
                    f"backend='pallas' requires TPU tensors, got {hidden_states.device}"
                )
            if hidden_states.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError(
                    f"backend='pallas' supports bfloat16/float32, got {hidden_states.dtype}"
                )
            if self.head_size > 128 and self.head_size % 128 != 0:
                raise ValueError(
                    "backend='pallas' requires head_size <= 128 or a multiple of 128, "
                    f"got {self.head_size}"
                )
            return "pallas"
        if backend in ("native", "sdpa"):
            return backend
        if backend == "auto":
            if (
                hidden_states.device.type == "tpu"
                and hidden_states.dtype in (torch.bfloat16, torch.float32)
                and (self.head_size <= 128 or self.head_size % 128 == 0)
                and _auto_prefers_pallas(hidden_states.size(0), hidden_states.size(1))
            ):
                return "pallas"
            return "native"
        raise ValueError(f"Unsupported backend: {backend}")

    def _project_qkv(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, frames, _ = hidden_states.shape
        cache = self.__dict__.get("_qkv_cache") if self.prepack_qkv else None
        if cache is not None and not torch.compiler.is_compiling():
            # Eager only: repack if weights were replaced or modified in place. Under Dynamo the
            # check is skipped because ``Tensor._version`` becomes a data-dependent symbol; the
            # cache is rebuilt by ``_apply``/``_load_from_state_dict`` and guarded by identity.
            if cache["sig"] != self._qkv_signature():
                self.prepack()
                cache = self.__dict__.get("_qkv_cache")
        if cache is not None:
            qkv = F.linear(hidden_states, cache["weight"], cache["bias"])
        elif self._qkv_signature() is not None:
            projections = (self.linear_q, self.linear_k, self.linear_v)
            weight = torch.cat([m.weight for m in projections], dim=0)
            has_bias = projections[0].bias is not None
            bias = torch.cat([m.bias for m in projections], dim=0) if has_bias else None
            qkv = F.linear(hidden_states, weight, bias)
        else:
            projections = (self.linear_q, self.linear_k, self.linear_v)
            qkv = torch.stack([m(hidden_states) for m in projections], dim=2)
        qkv = qkv.view(batch, frames, 3, self.num_heads, self.head_size)
        return tuple(t.transpose(1, 2) for t in qkv.unbind(2))  # type: ignore[return-value]

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        relative_position_embeddings: Tensor | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor | None]:
        key_lengths = kwargs.pop("_tpu_key_lengths", None)
        rel_distance = kwargs.pop("_tpu_rel_distance", None)
        rel_flat_index = kwargs.pop("_tpu_rel_flat_index", None)
        tpu_additive_mask = kwargs.pop("_tpu_additive_mask", None)
        conv_attention_mask = kwargs.pop("conv_attention_mask", None)

        if (
            self.tpu_backend == "eager"
            or output_attentions
            or self.position_embeddings_type != "relative_key"
            or (self.training and self.dropout.p > 0)
        ):
            return self._eager(
                hidden_states,
                attention_mask=attention_mask,
                relative_position_embeddings=relative_position_embeddings,
                output_attentions=output_attentions,
                **kwargs,
            )

        batch, frames, _ = hidden_states.shape
        left = self.left_max_position_embeddings
        right = self.right_max_position_embeddings
        embedding = self.distance_embedding.weight
        resolved = self._resolve_backend(hidden_states)

        if key_lengths is None and (attention_mask is not None or conv_attention_mask is not None):
            try:
                key_lengths = extract_key_lengths(
                    attention_mask, frames, conv_attention_mask=conv_attention_mask
                )
            except ValueError:
                if resolved == "pallas" and self.tpu_backend == "pallas":
                    raise
                resolved = "sdpa"

        if resolved == "pallas" and self._fused_qkv_supported():
            fused = self._fused_qkv_tensors()
            if fused is not None:
                # w2v-bert-fast layout: one packed projection read in place by the kernel.
                from w2vbert_tpu.kernels.fused_qkv_attention import fused_qkv_attention

                weight, bias, pair = fused
                qkv = F.linear(hidden_states, weight, bias)
                out = fused_qkv_attention(
                    qkv,
                    pair.to(qkv.dtype),
                    self.num_heads,
                    left,
                    right,
                    lengths=key_lengths,
                    tile_size=self.pallas_tile_size,
                )
                return self.linear_out(out), None

        query, key, value = self._project_qkv(hidden_states)
        if resolved == "pallas":
            from w2vbert_tpu.kernels.relative_attention import pallas_relative_key_attention

            emb_cache = self.__dict__.get("_emb_cache")
            padded_emb = None
            if emb_cache is not None and (
                torch.compiler.is_compiling() or emb_cache["sig"] == self._emb_signature()
            ):
                padded_emb = emb_cache["padded"]
            out = pallas_relative_key_attention(
                query,
                key,
                value,
                embedding,
                left,
                right,
                lengths=key_lengths,
                tile_size=self.pallas_tile_size,
                padded_embedding=padded_emb,
            )
        else:
            variant = self._native_variant_for(resolved, frames)
            eff_mask = tpu_additive_mask if tpu_additive_mask is not None else attention_mask
            out = native_relative_key_attention(
                query,
                key,
                value,
                embedding,
                left,
                right,
                attention_mask=eff_mask,
                lengths=None if eff_mask is not None else key_lengths,
                distance=rel_distance,
                flat_index=rel_flat_index,
                variant=variant,
            )

        out = out.transpose(1, 2).reshape(batch, frames, self.num_heads * self.head_size)
        return self.linear_out(out), None


class TPUEncoder(torch.nn.Module):
    """Encoder wrapper hoisting attention masks and relative distance tables across layers."""

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: Any,
    ):
        output_attentions = bool(kwargs.get("output_attentions", False))
        if output_attentions or self.embed_positions is not None:
            from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
                Wav2Vec2BertEncoder,
            )

            return Wav2Vec2BertEncoder.forward(
                self, hidden_states, attention_mask=attention_mask, **kwargs
            )

        conv_attention_mask = attention_mask
        key_lengths: Tensor | None = None
        if attention_mask is not None:
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
            key_lengths = attention_mask.to(torch.int32).sum(dim=-1, dtype=torch.int32)

        hidden_states = self.dropout(hidden_states)

        rel_distance: Tensor | None = None
        rel_flat_index: Tensor | None = None
        additive_mask: Tensor | None = None
        if len(self.layers) > 0:
            first_attn = getattr(self.layers[0], "self_attn", None)
            if isinstance(first_attn, TPUSelfAttention) and first_attn.tpu_backend != "eager":
                resolved = first_attn._resolve_backend(hidden_states)
                if resolved != "pallas":
                    frames = hidden_states.size(1)
                    left = first_attn.left_max_position_embeddings
                    right = first_attn.right_max_position_embeddings
                    variant = first_attn._native_variant_for(resolved, frames)
                    if variant.endswith("_index"):
                        rel_flat_index = build_flat_bucket_indices(
                            frames, frames, left, right, hidden_states.device
                        )
                    else:
                        rel_distance = build_distance_indices(
                            frames, frames, left, right, hidden_states.device
                        )
                    if key_lengths is not None:
                        additive_mask = make_additive_key_mask(
                            key_lengths, frames, hidden_states.dtype
                        )

        layer_kwargs = dict(kwargs)
        layer_kwargs.pop("output_attentions", None)
        for layer in self.layers:
            if self.training and self.config.layerdrop > 0.0:
                if torch.rand([]) < self.config.layerdrop:
                    continue
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                relative_position_embeddings=None,
                conv_attention_mask=conv_attention_mask,
                _tpu_key_lengths=key_lengths,
                _tpu_rel_distance=rel_distance,
                _tpu_rel_flat_index=rel_flat_index,
                _tpu_additive_mask=additive_mask,
                **layer_kwargs,
            )

        from transformers.modeling_outputs import BaseModelOutput

        return BaseModelOutput(last_hidden_state=hidden_states)


def _on_tpu(model: torch.nn.Module) -> bool:
    param = next(model.parameters(), None)
    return param is not None and param.device.type == "tpu"


def _use_eager_attention_config(model: torch.nn.Module) -> None:
    """Select Hugging Face's ``"eager"`` attention interface on ``model.config``.

    The TPU fast path never reads ``config._attn_implementation``. It only matters for the HF
    fallback used when ``output_attentions=True``: under ``"sdpa"`` (the transformers 5.x
    default) HF's output recorder silently drops attention weights, so they are never returned.
    """
    if getattr(model.config, "_attn_implementation", None) == "eager":
        return
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        setter("eager")
    else:
        model.config._attn_implementation = "eager"


def optimize(
    model: torch.nn.Module,
    *,
    backend: Backend = "auto",
    num_layers: int | None = None,
    compile: bool = False,
    compile_strategy: CompileStrategy = "whole",
    convolution: bool = False,
    prepack_qkv: bool = False,
    native_variant: NativeAttentionVariant = "auto",
    conv_impl: ConvImpl = "conv1d",
    pallas_tile_size: tuple[int, int] | None = None,
    fused_qkv: bool = True,
) -> torch.nn.Module:
    """Optimize a Hugging Face ``Wav2Vec2BertModel`` in place for TPU inference.

    Args:
        model: Loaded ``Wav2Vec2BertModel`` (or encoder).
        backend: Attention backend: ``"auto"``, ``"pallas"``, ``"native"``, ``"sdpa"``,
            or ``"eager"``.
        num_layers: Optional truncation to the first ``num_layers`` encoder layers.
        compile: Whether to compile with
            ``torch.compile(backend="tpu", dynamic=False, fullgraph=True)``.
        compile_strategy: ``"whole"`` (in-place whole-model compile) or ``"layer"``
            (per-layer compile).
        convolution: Whether to replace Conformer convolutions with channels-last
            ``TPUConvolutionModule``. Off by default: on TPU v5e the compiled Hugging Face module
            is 6-21% faster (``benchmarks/results/bench_convolution.md``).
        prepack_qkv: Whether to run Q/K/V as one prepacked ``[3d, d]`` matmul. Off by default:
            separate projections were faster on TPU v5e (``bench_attention.md``).
        native_variant: Native attention implementation (``"auto"`` picks by sequence length
            and avoids the long-sequence XLA gather cliff).
        conv_impl: Causal depthwise conv implementation used when ``convolution=True``:
            ``"conv1d"``, ``"unfold"``, ``"shift"`` or ``"pallas"``.
        pallas_tile_size: Optional ``(block_q, block_k)`` override for Pallas attention.
        fused_qkv: On the Pallas path, run Q/K/V as one packed projection that the kernel reads
            in place (head_dim 64, even heads), as in w2v-bert-fast.
    """
    from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import (
        Wav2Vec2BertConvolutionModule,
        Wav2Vec2BertEncoder,
        Wav2Vec2BertSelfAttention,
    )

    if backend not in ("auto", "pallas", "native", "sdpa", "eager"):
        raise ValueError(f"Unsupported backend: {backend}")
    if compile_strategy not in ("whole", "layer"):
        raise ValueError(f"Unsupported compile_strategy: {compile_strategy}")

    encoder = model.encoder if hasattr(model, "encoder") else model
    if num_layers is not None:
        if not 1 <= num_layers <= len(encoder.layers):
            raise ValueError(f"num_layers must be in 1..{len(encoder.layers)}, not {num_layers}")
        encoder.layers = encoder.layers[:num_layers]
        if hasattr(model, "config"):
            model.config.num_hidden_layers = num_layers

    # Disable training-only stochastic regularizers when inference optimization is applied.
    if hasattr(model, "config"):
        model.config.layerdrop = 0.0
        model.config.mask_time_prob = 0.0
        model.config.mask_feature_prob = 0.0
        _use_eager_attention_config(model)

    if backend != "eager" and isinstance(encoder, Wav2Vec2BertEncoder):
        if not isinstance(encoder, TPUEncoder):
            encoder.__class__ = _fast_class(type(encoder), TPUEncoder)

    attn_count = 0
    for module in model.modules():
        if (
            isinstance(module, Wav2Vec2BertSelfAttention)
            and getattr(module, "position_embeddings_type", None) == "relative_key"
        ):
            if not isinstance(module, TPUSelfAttention):
                module.__class__ = _fast_class(type(module), TPUSelfAttention)
            module.tpu_backend = backend
            module.prepack_qkv = prepack_qkv
            module.native_variant = native_variant
            module.pallas_tile_size = pallas_tile_size
            module.fused_qkv = fused_qkv
            module.prepack()
            attn_count += 1

    conv_count = 0
    if convolution and backend != "eager":
        for module in model.modules():
            if isinstance(module, Wav2Vec2BertConvolutionModule):
                if not isinstance(module, TPUConvolutionModule):
                    module.__class__ = _fast_class(type(module), TPUConvolutionModule)
                module.conv_impl = conv_impl
                module._invalidate_conv_cache()
                conv_count += 1

    if backend in ("auto", "pallas") and attn_count > 0 and _on_tpu(model):
        # jax_op registration cannot be traced by Dynamo; do it eagerly before any compile.
        from w2vbert_tpu.kernels.fused_qkv_attention import register_fused_ops
        from w2vbert_tpu.kernels.relative_attention import register_pallas_ops

        register_pallas_ops()
        if fused_qkv:
            register_fused_ops()

    if convolution and conv_impl == "pallas" and conv_count > 0 and _on_tpu(model):
        from w2vbert_tpu.kernels.causal_conv import register_conv_op

        register_conv_op()

    if compile:
        configure_compiler()
        if compile_strategy == "layer":
            for layer in encoder.layers:
                layer.compile(backend="tpu", dynamic=False, fullgraph=True)
        else:
            model.compile(backend="tpu", dynamic=False, fullgraph=True)

    model._w2vbert_tpu = {  # type: ignore[attr-defined]
        "backend": backend,
        "attention_modules": attn_count,
        "conv_modules": conv_count,
        "num_layers": len(encoder.layers),
        "compile": compile,
        "compile_strategy": compile_strategy if compile else None,
        "convolution": convolution,
        "prepack_qkv": prepack_qkv,
        "native_variant": native_variant,
        "conv_impl": conv_impl,
        "fused_qkv": fused_qkv,
    }
    return model


def load(
    name: str = "facebook/w2v-bert-2.0",
    *,
    dtype: torch.dtype | None = torch.bfloat16,
    device: torch.device | str | None = "tpu",
    compile: bool = True,
    backend: Backend = "auto",
    num_layers: int | None = None,
    compile_strategy: CompileStrategy = "whole",
    convolution: bool = False,
    prepack_qkv: bool = False,
    native_variant: NativeAttentionVariant = "auto",
    conv_impl: ConvImpl = "conv1d",
    pallas_tile_size: tuple[int, int] | None = None,
    fused_qkv: bool = True,
    **kwargs: Any,
) -> torch.nn.Module:
    """Load a pretrained ``Wav2Vec2BertModel`` and optimize it for TPU inference."""
    from transformers import Wav2Vec2BertModel

    if device is not None and torch.device(device).type == "tpu":
        import torch_tpu  # noqa: F401

        if not torch.accelerator.is_available():
            raise RuntimeError("TPU device requested, but no TPU accelerator is available")

    model = Wav2Vec2BertModel.from_pretrained(name, **kwargs).eval()
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    return optimize(
        model,
        backend=backend,
        num_layers=num_layers,
        compile=compile,
        compile_strategy=compile_strategy,
        convolution=convolution,
        prepack_qkv=prepack_qkv,
        native_variant=native_variant,
        conv_impl=conv_impl,
        pallas_tile_size=pallas_tile_size,
        fused_qkv=fused_qkv,
    )


__all__ = [
    "Backend",
    "CompileStrategy",
    "TPUEncoder",
    "TPUSelfAttention",
    "load",
    "optimize",
]
