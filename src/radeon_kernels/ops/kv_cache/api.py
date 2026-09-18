from __future__ import annotations

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
from radeon_kernels.ops.kv_cache.reference import (
    kv_cache_append_reference,
    kv_cache_copy_reference,
)


_SEMANTIC_VERSION = "1.0"
_LAYOUT = "block_slot_head_dim"


def _torch() -> Any:
    return torch_runtime("kv_cache")


def _validate_cache_pair(
    torch_module: Any,
    key_cache: Any,
    value_cache: Any,
    *,
    operator: str,
) -> tuple[int, int, int, int]:
    validate_tensor(torch_module, key_cache, name="key_cache", rank=4, operator=operator)
    validate_tensor(torch_module, value_cache, name="value_cache", rank=4, operator=operator)
    validate_same(
        operator,
        key_cache,
        value_cache,
        first_name="key_cache",
        second_name="value_cache",
    )
    if tuple(key_cache.shape) != tuple(value_cache.shape):
        raise ValueError(f"{operator} key_cache and value_cache must have the same shape")
    shape = tuple(int(size) for size in key_cache.shape)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{operator} cache dimensions must be positive")
    return shape


def _normalize_result(
    operator: str,
    result: Any,
    key_cache: Any,
    value_cache: Any,
) -> tuple[Any, Any]:
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError(f"{operator} implementation must return key_cache and value_cache")
    for returned, expected in zip(result, (key_cache, value_cache)):
        if hasattr(returned, "data_ptr") and returned.data_ptr() != expected.data_ptr():
            raise RuntimeError(f"{operator} implementation must return the input cache tensors")
    return result[0], result[1]


def _execute_append(
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    slot_mapping: Any,
    *,
    torch_module: Any,
    runtime: OperatorRuntime,
) -> tuple[Any, Any]:
    operator = "kv_cache_append"
    validate_tensor(torch_module, key, name="key", rank=3, operator=operator)
    validate_tensor(torch_module, value, name="value", rank=3, operator=operator)
    validate_same(operator, key, value, first_name="key", second_name="value")
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError(f"{operator} key and value must have the same shape")
    num_blocks, block_size, kv_heads, head_dim = _validate_cache_pair(
        torch_module, key_cache, value_cache, operator=operator
    )
    validate_same(operator, key, key_cache, first_name="key", second_name="key_cache")
    validate_index_tensor(
        torch_module, slot_mapping, name="slot_mapping", rank=1, operator=operator
    )
    if slot_mapping.device != key.device:
        raise ValueError(f"{operator} slot_mapping must use the same device as key")
    tokens = int(key.shape[0])
    if tokens <= 0 or int(key.shape[1]) != kv_heads or int(key.shape[2]) != head_dim:
        raise ValueError(f"{operator} key/value shape must be [tokens, kv_heads, head_dim]")
    if tuple(slot_mapping.shape) != (tokens,):
        raise ValueError(f"{operator} slot_mapping must contain one slot per token")
    context = tensor_context(
        torch_module, operator, key, value, key_cache, value_cache, slot_mapping
    )
    if not context.contiguous:
        raise ValueError(f"{operator} paged-layout tensors must be contiguous")
    if context.requires_grad:
        raise ValueError(f"{operator} is inference-only and does not support autograd")
    signature = {
        "dtype": context.dtype,
        "tokens": tokens,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "index_dtype": index_dtype_name(torch_module, slot_mapping.dtype, operator),
        "layout": _LAYOUT,
        "contiguous": context.contiguous,
        "requires_grad": False,
    }
    result = runtime.execute(
        operator=operator,
        semantic_version=_SEMANTIC_VERSION,
        signature=signature,
        device_index=context.device_index,
        torch_module=torch_module,
        native_arguments=(key, value, key_cache, value_cache, slot_mapping),
        fallback=lambda: kv_cache_append_reference(
            torch_module, key, value, key_cache, value_cache, slot_mapping
        ),
        fallback_selected="torch/index_copy",
    )
    return _normalize_result(operator, result, key_cache, value_cache)


def _execute_copy(
    key_cache: Any,
    value_cache: Any,
    block_mapping: Any,
    *,
    torch_module: Any,
    runtime: OperatorRuntime,
) -> tuple[Any, Any]:
    operator = "kv_cache_copy"
    num_blocks, block_size, kv_heads, head_dim = _validate_cache_pair(
        torch_module, key_cache, value_cache, operator=operator
    )
    validate_index_tensor(
        torch_module, block_mapping, name="block_mapping", rank=2, operator=operator
    )
    if block_mapping.device != key_cache.device:
        raise ValueError(f"{operator} block_mapping must use the same device as key_cache")
    if int(block_mapping.shape[1]) != 2 or int(block_mapping.shape[0]) <= 0:
        raise ValueError(f"{operator} block_mapping must have shape [pairs, 2]")
    context = tensor_context(torch_module, operator, key_cache, value_cache, block_mapping)
    if not context.contiguous:
        raise ValueError(f"{operator} paged-layout tensors must be contiguous")
    if context.requires_grad:
        raise ValueError(f"{operator} is inference-only and does not support autograd")
    signature = {
        "dtype": context.dtype,
        "pairs": int(block_mapping.shape[0]),
        "num_blocks": num_blocks,
        "block_size": block_size,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "index_dtype": index_dtype_name(torch_module, block_mapping.dtype, operator),
        "layout": _LAYOUT,
        "contiguous": context.contiguous,
        "requires_grad": False,
    }
    result = runtime.execute(
        operator=operator,
        semantic_version=_SEMANTIC_VERSION,
        signature=signature,
        device_index=context.device_index,
        torch_module=torch_module,
        native_arguments=(key_cache, value_cache, block_mapping),
        fallback=lambda: kv_cache_copy_reference(
            torch_module, key_cache, value_cache, block_mapping
        ),
        fallback_selected="torch/index_copy",
    )
    return _normalize_result(operator, result, key_cache, value_cache)


def kv_cache_append(
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    slot_mapping: Any,
) -> tuple[Any, Any]:
    """Append token K/V tensors to physical slots in a paged cache."""

    return _execute_append(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        torch_module=_torch(),
        runtime=default_operator_runtime(),
    )


def kv_cache_copy(
    key_cache: Any,
    value_cache: Any,
    block_mapping: Any,
) -> tuple[Any, Any]:
    """Copy physical K/V pages from source to destination block IDs."""

    return _execute_copy(
        key_cache,
        value_cache,
        block_mapping,
        torch_module=_torch(),
        runtime=default_operator_runtime(),
    )
