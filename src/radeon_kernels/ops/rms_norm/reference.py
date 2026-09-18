from __future__ import annotations

from typing import Any


def rms_norm_reference(torch_module: Any, x: Any, weight: Any, eps: float) -> Any:
    source = x.float()
    variance = source.square().mean(dim=-1, keepdim=True)
    normalized = source * torch_module.rsqrt(variance + eps)
    return (normalized * weight.float()).to(dtype=x.dtype)
