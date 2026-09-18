from __future__ import annotations

import math
from typing import Any


def paged_attention_decode_reference(
    torch_module: Any,
    query: Any,
    key_cache: Any,
    value_cache: Any,
    block_tables: Any,
    context_lengths: Any,
    scale: float | None = None,
) -> Any:
    """Evaluate one decode query per sequence against a paged K/V cache."""

    batch, query_heads, head_dim = (int(size) for size in query.shape)
    block_size = int(key_cache.shape[1])
    kv_heads = int(key_cache.shape[2])
    maximum_context = int(block_tables.shape[1]) * block_size
    positions = torch_module.arange(maximum_context, device=query.device)
    logical_blocks = positions // block_size
    block_offsets = positions % block_size
    physical_blocks = block_tables.to(dtype=torch_module.int64).index_select(
        1, logical_blocks
    )
    physical_slots = physical_blocks * block_size + block_offsets.unsqueeze(0)

    flat_shape = (-1, kv_heads, head_dim)
    keys = key_cache.view(flat_shape).index_select(0, physical_slots.reshape(-1))
    values = value_cache.view(flat_shape).index_select(0, physical_slots.reshape(-1))
    keys = keys.view(batch, maximum_context, kv_heads, head_dim)
    values = values.view(batch, maximum_context, kv_heads, head_dim)

    group_size = query_heads // kv_heads
    kv_head_indices = torch_module.arange(query_heads, device=query.device) // group_size
    keys = keys.index_select(2, kv_head_indices).float()
    values = values.index_select(2, kv_head_indices).float()
    scores = torch_module.einsum("bhd,bthd->bht", query.float(), keys)
    normalized_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    scores = scores * normalized_scale
    valid = positions.unsqueeze(0) < context_lengths.to(dtype=torch_module.int64).unsqueeze(1)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    probabilities = torch_module.softmax(scores, dim=-1, dtype=torch_module.float32)
    output = torch_module.einsum("bht,bthd->bhd", probabilities, values)
    return output.to(dtype=query.dtype)
