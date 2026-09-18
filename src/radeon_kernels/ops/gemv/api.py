from __future__ import annotations

from typing import Any

from radeon_kernels.ops._common import (
    tensor_context,
    torch_runtime,
    validate_same,
    validate_tensor,
)
from radeon_kernels.ops._runtime import OperatorRuntime, default_operator_runtime
from radeon_kernels.ops.gemv.reference import gemv_fallback


_SEMANTIC_VERSION = "1.0"


def _torch() -> Any:
    return torch_runtime("gemv")


def _execute(
    x: Any,
    weight: Any,
    *,
    torch_module: Any,
    runtime: OperatorRuntime,
) -> Any:
    validate_tensor(torch_module, x, name="x", rank=2, operator="gemv")
    validate_tensor(torch_module, weight, name="weight", rank=2, operator="gemv")
    validate_same("gemv", x, weight, first_name="x", second_name="weight")
    m, k = int(x.shape[0]), int(x.shape[1])
    n, weight_k = int(weight.shape[0]), int(weight.shape[1])
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("gemv dimensions must be positive")
    if m > 32:
        raise ValueError("gemv M must be in [1, 32]")
    if weight_k != k:
        raise ValueError("gemv x and weight must share the K dimension")
    context = tensor_context(torch_module, "gemv", x, weight)
    signature = {
        "dtype": context.dtype,
        "m": m,
        "n": n,
        "k": k,
        "weight_layout": "nk",
        "contiguous": context.contiguous,
        "requires_grad": context.requires_grad,
    }
    return runtime.execute(
        operator="gemv",
        semantic_version=_SEMANTIC_VERSION,
        signature=signature,
        device_index=context.device_index,
        torch_module=torch_module,
        native_arguments=(x, weight),
        fallback=lambda: gemv_fallback(torch_module, x, weight),
        fallback_selected="torch/mm",
    )


def gemv(x: Any, weight: Any) -> Any:
    """Compute ``x @ weight.T`` for GEMV and skinny-GEMM inference workloads."""

    return _execute(
        x,
        weight,
        torch_module=_torch(),
        runtime=default_operator_runtime(),
    )
