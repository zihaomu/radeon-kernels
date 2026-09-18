from __future__ import annotations

from typing import Any


def _rotate_half(torch_module: Any, value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return torch_module.cat((-value[..., half:], value[..., :half]), dim=-1)


def rope_reference(
    torch_module: Any, q: Any, k: Any, cos: Any, sin: Any
) -> tuple[Any, Any]:
    cos_fp32 = cos.float().unsqueeze(1)
    sin_fp32 = sin.float().unsqueeze(1)
    q_fp32 = q.float()
    k_fp32 = k.float()
    q_out = q_fp32 * cos_fp32 + _rotate_half(torch_module, q_fp32) * sin_fp32
    k_out = k_fp32 * cos_fp32 + _rotate_half(torch_module, k_fp32) * sin_fp32
    return q_out.to(dtype=q.dtype), k_out.to(dtype=k.dtype)
