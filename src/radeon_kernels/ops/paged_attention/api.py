from __future__ import annotations

import math
from typing import Any

from radeon_kernels.ops._common import (
    index_dtype_name,
    tensor_context,
    torch_runtime,
    validate_index_tensor,
    validate_same,
    validate_tensor,
)
from radeon_kernels.ops._runtime import OperatorRuntime, default_operator_runtime
from radeon_kernels.ops.paged_attention.reference import paged_attention_decode_reference


_SEMANTIC_VERSION = "1.0"
_CACHE_LAYOUT = "block_slot_head_dim"
_SUPPORTED_HEAD_DIMS = {64, 128, 256}


def _torch() -> Any:
    return torch_runtime("paged_attention_decode")


def _normalize_scale(scale: float | None, head_dim: int) -> tuple[float, str]:
    if scale is None:
        return 1.0 / math.sqrt(head_dim), "default"
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("paged_attention_decode scale must be a number")
    normalized = float(scale)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("paged_attention_decode scale must be positive and finite")
    return normalized, "explicit"


def _execute(
    query: Any,
    key_cache: Any,
    value_cache: Any,
    block_tables: Any,
    context_lengths: Any,
    scale: float | None = None,
    *,
    torch_module: Any,
    runtime: OperatorRuntime,
) -> Any:
    operator = "paged_attention_decode"
    validate_tensor(torch_module, query, name="query", rank=3, operator=operator)
    validate_tensor(torch_module, key_cache, name="key_cache", rank=4, operator=operator)
    validate_tensor(torch_module, value_cache, name="value_cache", rank=4, operator=operator)
    validate_same(operator, query, key_cache, first_name="query", second_name="key_cache")
    validate_same(
        operator,
        key_cache,
        value_cache,
        first_name="key_cache",
        second_name="value_cache",
    )
    if tuple(key_cache.shape) != tuple(value_cache.shape):
        raise ValueError(f"{operator} key_cache and value_cache must have the same shape")
    validate_index_tensor(
        torch_module, block_tables, name="block_tables", rank=2, operator=operator
    )
    validate_index_tensor(
        torch_module, context_lengths, name="context_lengths", rank=1, operator=operator
    )
    if block_tables.device != query.device or context_lengths.device != query.device:
        raise ValueError(f"{operator} index tensors must use the same device as query")
    if block_tables.dtype != context_lengths.dtype:
        raise TypeError(f"{operator} block_tables and context_lengths must use the same dtype")

    batch, query_heads, head_dim = (int(size) for size in query.shape)
    num_blocks, block_size, kv_heads, cache_head_dim = (
        int(size) for size in key_cache.shape
    )
    if min(batch, query_heads, num_blocks, block_size, kv_heads) <= 0:
        raise ValueError(f"{operator} tensor dimensions must be positive")
    if head_dim != cache_head_dim or head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"{operator} head_dim must match cache and be one of 64, 128, 256")
    if query_heads % kv_heads != 0:
        raise ValueError(f"{operator} query_heads must be divisible by kv_heads")
    if int(block_tables.shape[0]) != batch or int(block_tables.shape[1]) <= 0:
        raise ValueError(f"{operator} block_tables must have shape [batch, max_blocks]")
    if tuple(context_lengths.shape) != (batch,):
        raise ValueError(f"{operator} context_lengths must contain one length per sequence")
    context = tensor_context(
        torch_module, operator, query, key_cache, value_cache, block_tables, context_lengths
    )
    if not context.contiguous:
        raise ValueError(f"{operator} inputs must be contiguous")
    if context.requires_grad:
        raise ValueError(f"{operator} is inference-only and does not support autograd")
    normalized_scale, scale_mode = _normalize_scale(scale, head_dim)
    signature = {
        "dtype": context.dtype,
        "batch": batch,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "max_blocks_per_sequence": int(block_tables.shape[1]),
        "index_dtype": index_dtype_name(torch_module, block_tables.dtype, operator),
        "cache_layout": _CACHE_LAYOUT,
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
        native_arguments=(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            normalized_scale,
        ),
        fallback=lambda: paged_attention_decode_reference(
            torch_module,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            normalized_scale,
        ),
        fallback_selected="torch/paged_attention",
    )


def paged_attention_decode(
    query: Any,
    key_cache: Any,
    value_cache: Any,
    block_tables: Any,
    context_lengths: Any,
    scale: float | None = None,
) -> Any:
    """Attend one query token per sequence over a paged K/V cache."""

    return _execute(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        scale,
        torch_module=_torch(),
        runtime=default_operator_runtime(),
    )
