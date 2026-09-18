from __future__ import annotations

from typing import Any


def kv_cache_append_reference(
    torch_module: Any,
    key: Any,
    value: Any,
    key_cache: Any,
    value_cache: Any,
    slot_mapping: Any,
) -> tuple[Any, Any]:
    """Scatter token K/V rows into the flattened physical page slots."""

    slots = slot_mapping.to(dtype=torch_module.int64)
    flat_shape = (-1, int(key_cache.shape[2]), int(key_cache.shape[3]))
    key_cache.view(flat_shape).index_copy_(0, slots, key)
    value_cache.view(flat_shape).index_copy_(0, slots, value)
    return key_cache, value_cache


def kv_cache_copy_reference(
    torch_module: Any,
    key_cache: Any,
    value_cache: Any,
    block_mapping: Any,
) -> tuple[Any, Any]:
    """Copy physical pages using a source snapshot for overlap-safe semantics."""

    mapping = block_mapping.to(dtype=torch_module.int64)
    sources = mapping[:, 0]
    destinations = mapping[:, 1]
    key_source = key_cache.index_select(0, sources)
    value_source = value_cache.index_select(0, sources)
    key_cache.index_copy_(0, destinations, key_source)
    value_cache.index_copy_(0, destinations, value_source)
    return key_cache, value_cache
