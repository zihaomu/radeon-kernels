from __future__ import annotations

from typing import Any

from radeon_kernels.ops._common import (
    tensor_context,
    torch_runtime,
    validate_same,
    validate_tensor,
)
from radeon_kernels.ops._runtime import OperatorRuntime, default_operator_runtime
from radeon_kernels.ops.rope.reference import rope_reference


_SEMANTIC_VERSION = "1.0"


def _torch() -> Any:
    return torch_runtime("rope")


def _execute(
    q: Any,
    k: Any,
    cos: Any,
    sin: Any,
    *,
    torch_module: Any,
    runtime: OperatorRuntime,
) -> tuple[Any, Any]:
    for name, tensor, rank in (("q", q, 3), ("k", k, 3), ("cos", cos, 2), ("sin", sin, 2)):
        validate_tensor(torch_module, tensor, name=name, rank=rank, operator="rope")
    for name, tensor in (("k", k), ("cos", cos), ("sin", sin)):
        validate_same("rope", q, tensor, first_name="q", second_name=name)
    tokens = int(q.shape[0])
    head_dim = int(q.shape[-1])
    if int(k.shape[0]) != tokens or int(k.shape[-1]) != head_dim:
        raise ValueError("rope q and k must share token count and head dimension")
    if tuple(cos.shape) != (tokens, head_dim) or tuple(sin.shape) != (tokens, head_dim):
        raise ValueError("rope cos and sin must have shape (tokens, head_dim)")
    if head_dim % 2:
        raise ValueError("rope head dimension must be even")
    context = tensor_context(torch_module, "rope", q, k, cos, sin)
    signature = {
        "dtype": context.dtype,
        "tokens": tokens,
        "q_heads": int(q.shape[1]),
        "kv_heads": int(k.shape[1]),
        "head_dim": head_dim,
        "mode": "split_half",
        "contiguous": context.contiguous,
        "requires_grad": context.requires_grad,
    }
    result = runtime.execute(
        operator="rope",
        semantic_version=_SEMANTIC_VERSION,
        signature=signature,
        device_index=context.device_index,
        torch_module=torch_module,
        native_arguments=(q, k, cos, sin),
        fallback=lambda: rope_reference(torch_module, q, k, cos, sin),
        fallback_selected="torch/rope",
    )
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError("rope implementation must return two tensors")
    return result[0], result[1]


def rope(q: Any, k: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
    """Apply split-half rotary position embeddings to Q and K tensors."""

    return _execute(
        q,
        k,
        cos,
        sin,
        torch_module=_torch(),
        runtime=default_operator_runtime(),
    )
