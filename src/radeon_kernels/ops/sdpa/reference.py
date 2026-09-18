from __future__ import annotations

from typing import Any


def _expanded_kv(torch_module: Any, query: Any, tensor: Any) -> Any:
    group_size = int(query.shape[1]) // int(tensor.shape[1])
    if group_size == 1:
        return tensor
    return tensor.repeat_interleave(group_size, dim=1)


def sdpa_reference(
    torch_module: Any,
    query: Any,
    key: Any,
    value: Any,
    is_causal: bool,
    scale: float,
) -> Any:
    """Independent explicit FP32 reference for the frozen SDPA 1.0 semantics."""

    expanded_key = _expanded_kv(torch_module, query, key).float()
    expanded_value = _expanded_kv(torch_module, query, value).float()
    scores = torch_module.matmul(
        query.float(), expanded_key.transpose(-2, -1)
    ) * scale
    if is_causal:
        sequence = int(query.shape[2])
        causal_mask = torch_module.ones(
            (sequence, sequence), device=query.device, dtype=torch_module.bool
        ).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    probabilities = torch_module.softmax(scores, dim=-1, dtype=torch_module.float32)
    return torch_module.matmul(probabilities, expanded_value).to(dtype=query.dtype)


def sdpa_fallback(
    torch_module: Any,
    query: Any,
    key: Any,
    value: Any,
    is_causal: bool,
    scale: float,
) -> Any:
    return torch_module.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=int(query.shape[1]) != int(key.shape[1]),
    )
