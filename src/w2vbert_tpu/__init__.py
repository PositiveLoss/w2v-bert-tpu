"""TPU-native TorchTPU and Pallas inference for w2v-BERT 2.0 (Wav2Vec2BertModel)."""

from __future__ import annotations

from w2vbert_tpu.attention import (
    extract_key_lengths,
    native_relative_key_attention,
    relative_key_reference,
)
from w2vbert_tpu.batching import (
    DEFAULT_BATCH_BUCKETS,
    DEFAULT_FRAME_BUCKETS,
    BucketedBatch,
    pad_to_bucket,
    prewarm,
    select_batch_bucket,
    select_frame_bucket,
    unpad_output,
)
from w2vbert_tpu.compiler import (
    compile_tpu,
    configure_compiler,
    get_compile_metrics,
    reset_compiler,
)
from w2vbert_tpu.convolution import TPUConvolutionModule
from w2vbert_tpu.features import extract_features, load_audio
from w2vbert_tpu.model import (
    Backend,
    CompileStrategy,
    TPUEncoder,
    TPUSelfAttention,
    load,
    optimize,
)

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "BucketedBatch",
    "CompileStrategy",
    "DEFAULT_BATCH_BUCKETS",
    "DEFAULT_FRAME_BUCKETS",
    "TPUConvolutionModule",
    "TPUEncoder",
    "TPUSelfAttention",
    "__version__",
    "compile_tpu",
    "configure_compiler",
    "extract_features",
    "extract_key_lengths",
    "get_compile_metrics",
    "load",
    "load_audio",
    "native_relative_key_attention",
    "optimize",
    "pad_to_bucket",
    "prewarm",
    "relative_key_reference",
    "reset_compiler",
    "select_batch_bucket",
    "select_frame_bucket",
    "unpad_output",
]
