from __future__ import annotations

import math
from typing import Any

from radeon_kernels.ops._common import (
    tensor_context,
    torch_runtime,
    validate_same,
    validate_tensor,
)
from radeon_kernels.ops._runtime import OperatorRuntime, default_operator_runtime
from radeon_kernels.ops.sdpa.reference import sdpa_fallback


_SEMANTIC_VERSION = "1.0"
_LAYOUT = "batch_head_sequence_dim"
_SUPPORTED_HEAD_DIMS = {64, 128, 256}


def _torch() -> Any:
    return torch_runtime("sdpa")


def _normalize_scale(scale: float | None, head_dim: int) -> tuple[float, str]:
    if scale is None:
        return 1.0 / math.sqrt(head_dim), "default"
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("sdpa scale must be a number")
    normalized = float(scale)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("sdpa scale must be positive and finite")
    return normalized, "explicit"


def _execute(
    query: Any,
    key: Any,
    value: Any,
    is_causal: bool = False,
    scale: float | None = None,
    *,
    torch_module: Any,
    runtime: OperatorRuntime,
) -> Any:
    operator = "sdpa"
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        validate_tensor(torch_module, tensor, name=name, rank=4, operator=operator)
    validate_same(operator, query, key, first_name="query", second_name="key")
    validate_same(operator, key, value, first_name="key", second_name="value")
    if not isinstance(is_causal, bool):
        raise TypeError("sdpa is_causal must be bool")

    batch, query_heads, query_sequence, head_dim = (int(size) for size in query.shape)
    key_batch, kv_heads, key_sequence, key_head_dim = (int(size) for size in key.shape)
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError("sdpa key and value must have the same shape")
    if min(batch, query_heads, query_sequence, kv_heads) <= 0:
        raise ValueError("sdpa tensor dimensions must be positive")
    if key_batch != batch:
        raise ValueError("sdpa query, key and value must share the batch dimension")
    if query_sequence != key_sequence:
        raise ValueError("sdpa 1.0 requires equal query and KV sequence lengths")
    if head_dim != key_head_dim or head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError("sdpa head_dim must match and be one of 64, 128, 256")
    if query_heads % kv_heads != 0:
        raise ValueError("sdpa query_heads must be divisible by kv_heads")

    context = tensor_context(torch_module, operator, query, key, value)
    if not context.contiguous:
        raise ValueError("sdpa inputs must be contiguous")
    if context.requires_grad:
        raise ValueError("sdpa is inference-only and does not support autograd")
    normalized_scale, scale_mode = _normalize_scale(scale, head_dim)
    signature = {
        "dtype": context.dtype,
        "batch": batch,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "sequence": query_sequence,
        "head_dim": head_dim,
        "layout": _LAYOUT,
        "attention_mode": "causal" if is_causal else "noncausal",
        "scale_mode": scale_mode,
        "contiguous": True,
        "requires_grad": False,
    }
    return runtime.execute(
        operator=operator,
        semantic_version=_SEMANTIC_VERSION,
        signature=signature,
        device_index=context.device_index,
        torch_module=torch_module,
        native_arguments=(query, key, value, is_causal, normalized_scale),
        fallback=lambda: sdpa_fallback(
            torch_module, query, key, value, is_causal, normalized_scale
        ),
        fallback_selected="torch/scaled_dot_product_attention",
    )


def sdpa(
    query: Any,
    key: Any,
    value: Any,
    *,
    is_causal: bool = False,
    scale: float | None = None,
) -> Any:
    """Run inference-only scaled dot-product self-attention with optional GQA."""

    return _execute(
        query,
        key,
        value,
        is_causal,
        scale,
        torch_module=_torch(),
        runtime=default_operator_runtime(),
    )
