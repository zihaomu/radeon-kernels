"""Stable public operator APIs."""

from radeon_kernels.ops.add_rms_norm import add_rms_norm
from radeon_kernels.ops.gemm import gemm
from radeon_kernels.ops.gemv import gemv
from radeon_kernels.ops.kv_cache import kv_cache_append, kv_cache_copy
from radeon_kernels.ops.paged_attention import paged_attention_decode
from radeon_kernels.ops.rms_norm import rms_norm
from radeon_kernels.ops.rope import rope
from radeon_kernels.ops.sdpa import sdpa
from radeon_kernels.ops.softmax import softmax
from radeon_kernels.ops.swiglu import silu_mul

__all__ = [
    "add_rms_norm",
    "gemm",
    "gemv",
    "kv_cache_append",
    "kv_cache_copy",
    "paged_attention_decode",
    "rms_norm",
    "rope",
    "sdpa",
    "silu_mul",
    "softmax",
]
