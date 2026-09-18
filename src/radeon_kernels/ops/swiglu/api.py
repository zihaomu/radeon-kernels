from __future__ import annotations

from typing import Any

from radeon_kernels.ops._common import (
    tensor_context,
    torch_runtime,
    validate_same,
    validate_tensor,
)
from radeon_kernels.ops._runtime import OperatorRuntime, default_operator_runtime
from radeon_kernels.ops.swiglu.reference import silu_mul_reference


_SEMANTIC_VERSION = "1.0"


def _torch() -> Any:
    return torch_runtime("silu_mul")


def _execute(
    gate: Any,
    up: Any,
    *,
    torch_module: Any,
    runtime: OperatorRuntime,
) -> Any:
    validate_tensor(torch_module, gate, name="gate", rank=2, operator="silu_mul")
    validate_tensor(torch_module, up, name="up", rank=2, operator="silu_mul")
    validate_same("silu_mul", gate, up, first_name="gate", second_name="up")
    if tuple(gate.shape) != tuple(up.shape):
        raise ValueError("silu_mul gate and up must have the same shape")
    context = tensor_context(torch_module, "silu_mul", gate, up)
    signature = {
        "dtype": context.dtype,
        "tokens": int(gate.shape[0]),
        "hidden": int(gate.shape[1]),
        "contiguous": context.contiguous,
        "requires_grad": context.requires_grad,
    }
    return runtime.execute(
        operator="swiglu",
        semantic_version=_SEMANTIC_VERSION,
        signature=signature,
        device_index=context.device_index,
        torch_module=torch_module,
        native_arguments=(gate, up),
        fallback=lambda: silu_mul_reference(torch_module, gate, up),
        fallback_selected="torch/silu_mul",
    )


def silu_mul(gate: Any, up: Any) -> Any:
    """Compute ``silu(gate) * up`` using a published static winner."""

    return _execute(
        gate,
        up,
        torch_module=_torch(),
        runtime=default_operator_runtime(),
    )
