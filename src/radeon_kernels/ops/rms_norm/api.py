from __future__ import annotations

from typing import Any

from radeon_kernels.ops._common import (
    tensor_context,
    torch_runtime,
    validate_eps,
    validate_same,
    validate_tensor,
)
from radeon_kernels.ops._runtime import OperatorRuntime, default_operator_runtime
from radeon_kernels.ops.rms_norm.reference import rms_norm_reference


_SEMANTIC_VERSION = "1.0"


def _torch() -> Any:
    return torch_runtime("rms_norm")


def _execute(
    x: Any,
    weight: Any,
    eps: float,
    *,
    torch_module: Any,
    runtime: OperatorRuntime,
) -> Any:
    validate_tensor(torch_module, x, name="x", rank=2, operator="rms_norm")
    validate_tensor(torch_module, weight, name="weight", rank=1, operator="rms_norm")
    validate_same("rms_norm", x, weight, first_name="x", second_name="weight")
    if tuple(weight.shape) != (int(x.shape[-1]),):
        raise ValueError("rms_norm weight must match the hidden dimension")
    normalized_eps = validate_eps(eps, "rms_norm")
    context = tensor_context(torch_module, "rms_norm", x, weight)
    signature = {
        "dtype": context.dtype,
        "tokens": int(x.shape[0]),
        "hidden": int(x.shape[1]),
        "eps": normalized_eps,
        "contiguous": context.contiguous,
        "requires_grad": context.requires_grad,
    }
    return runtime.execute(
        operator="rms_norm",
        semantic_version=_SEMANTIC_VERSION,
        signature=signature,
        device_index=context.device_index,
        torch_module=torch_module,
        native_arguments=(x, weight, normalized_eps),
        fallback=lambda: rms_norm_reference(torch_module, x, weight, normalized_eps),
        fallback_selected="torch/rms_norm",
    )


def rms_norm(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    """Apply last-dimension RMS normalization using a published static winner."""

    return _execute(x, weight, eps, torch_module=_torch(), runtime=default_operator_runtime())
