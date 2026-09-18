from __future__ import annotations

from typing import Any

from radeon_kernels.ops._common import tensor_context, torch_runtime, validate_tensor
from radeon_kernels.ops._runtime import OperatorRuntime, default_operator_runtime
from radeon_kernels.ops.softmax.reference import softmax_reference


_SEMANTIC_VERSION = "1.0"


def _torch() -> Any:
    return torch_runtime("softmax")


def _execute(
    x: Any,
    *,
    torch_module: Any,
    runtime: OperatorRuntime,
) -> Any:
    validate_tensor(torch_module, x, name="x", rank=2, operator="softmax")
    context = tensor_context(torch_module, "softmax", x)
    signature = {
        "dtype": context.dtype,
        "rows": int(x.shape[0]),
        "width": int(x.shape[1]),
        "contiguous": context.contiguous,
        "requires_grad": context.requires_grad,
    }
    return runtime.execute(
        operator="softmax",
        semantic_version=_SEMANTIC_VERSION,
        signature=signature,
        device_index=context.device_index,
        torch_module=torch_module,
        native_arguments=(x,),
        fallback=lambda: softmax_reference(torch_module, x),
        fallback_selected="torch/softmax",
    )


def softmax(x: Any) -> Any:
    """Apply numerically stable last-dimension softmax."""

    return _execute(x, torch_module=_torch(), runtime=default_operator_runtime())
