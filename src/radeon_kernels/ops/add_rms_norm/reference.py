from __future__ import annotations

from typing import Any

from radeon_kernels.ops.rms_norm.reference import rms_norm_reference


def add_rms_norm_reference(
    torch_module: Any, x: Any, residual: Any, weight: Any, eps: float
) -> tuple[Any, Any]:
    residual_out = x + residual
    return rms_norm_reference(torch_module, residual_out, weight, eps), residual_out
