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
from radeon_kernels.ops.add_rms_norm.reference import add_rms_norm_reference


_SEMANTIC_VERSION = "1.0"


def _torch() -> Any:
    return torch_runtime("add_rms_norm")


def _execute(
    x: Any,
    residual: Any,
    weight: Any,
    eps: float,
    *,
    torch_module: Any,
    runtime: OperatorRuntime,
) -> tuple[Any, Any]:
    for name, tensor, rank in (("x", x, 2), ("residual", residual, 2), ("weight", weight, 1)):
        validate_tensor(torch_module, tensor, name=name, rank=rank, operator="add_rms_norm")
    validate_same("add_rms_norm", x, residual, first_name="x", second_name="residual")
    validate_same("add_rms_norm", x, weight, first_name="x", second_name="weight")
    if tuple(x.shape) != tuple(residual.shape):
        raise ValueError("add_rms_norm x and residual must have the same shape")
    if tuple(weight.shape) != (int(x.shape[-1]),):
        raise ValueError("add_rms_norm weight must match the hidden dimension")
    normalized_eps = validate_eps(eps, "add_rms_norm")
    context = tensor_context(torch_module, "add_rms_norm", x, residual, weight)
    signature = {
        "dtype": context.dtype,
        "tokens": int(x.shape[0]),
        "hidden": int(x.shape[1]),
        "eps": normalized_eps,
        "contiguous": context.contiguous,
        "requires_grad": context.requires_grad,
    }
    result = runtime.execute(
        operator="add_rms_norm",
        semantic_version=_SEMANTIC_VERSION,
        signature=signature,
        device_index=context.device_index,
        torch_module=torch_module,
        native_arguments=(x, residual, weight, normalized_eps),
        fallback=lambda: add_rms_norm_reference(
            torch_module, x, residual, weight, normalized_eps
        ),
        fallback_selected="torch/add_rms_norm",
    )
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError("add_rms_norm implementation must return two tensors")
    return result[0], result[1]


def add_rms_norm(
    x: Any, residual: Any, weight: Any, eps: float = 1e-6
) -> tuple[Any, Any]:
    """Add a residual and RMS-normalize it, returning normalized and residual outputs."""

    return _execute(
        x,
        residual,
        weight,
        eps,
        torch_module=_torch(),
        runtime=default_operator_runtime(),
    )
