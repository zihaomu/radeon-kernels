from __future__ import annotations

from typing import Any


def gemv_reference(torch_module: Any, x: Any, weight: Any) -> Any:
    """Independent FP32-accumulation oracle for ``x @ weight.T``."""

    return torch_module.mm(x.float(), weight.float().transpose(0, 1)).to(dtype=x.dtype)


def gemv_fallback(torch_module: Any, x: Any, weight: Any) -> Any:
    """Native-dtype PyTorch fallback used when no published winner matches."""

    return torch_module.mm(x, weight.transpose(0, 1))
