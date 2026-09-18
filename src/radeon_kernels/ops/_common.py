from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorContext:
    dtype: str
    device_index: int
    contiguous: bool
    requires_grad: bool


def torch_runtime(operator: str) -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(f"{operator} requires a ROCm PyTorch installation") from error
    return torch


def dtype_name(torch_module: Any, dtype: Any, operator: str) -> str:
    if dtype == torch_module.float16:
        return "fp16"
    if dtype == torch_module.bfloat16:
        return "bf16"
    raise TypeError(f"{operator} supports torch.float16 and torch.bfloat16 tensors")


def index_dtype_name(torch_module: Any, dtype: Any, operator: str) -> str:
    if dtype == torch_module.int32:
        return "int32"
    if dtype == torch_module.int64:
        return "int64"
    raise TypeError(f"{operator} index tensors must use torch.int32 or torch.int64")


def validate_tensor(
    torch_module: Any,
    tensor: Any,
    *,
    name: str,
    rank: int,
    operator: str,
) -> None:
    tensor_type = getattr(torch_module, "Tensor", ())
    if not isinstance(tensor, tensor_type):
        raise TypeError(f"{operator} {name} must be a PyTorch tensor")
    if tensor.dim() != rank:
        raise ValueError(f"{operator} {name} must be rank {rank}")
    if not tensor.is_cuda:
        raise ValueError(f"{operator} {name} must be on a visible ROCm GPU")
    dtype_name(torch_module, tensor.dtype, operator)


def validate_index_tensor(
    torch_module: Any,
    tensor: Any,
    *,
    name: str,
    rank: int,
    operator: str,
) -> None:
    tensor_type = getattr(torch_module, "Tensor", ())
    if not isinstance(tensor, tensor_type):
        raise TypeError(f"{operator} {name} must be a PyTorch tensor")
    if tensor.dim() != rank:
        raise ValueError(f"{operator} {name} must be rank {rank}")
    if not tensor.is_cuda:
        raise ValueError(f"{operator} {name} must be on a visible ROCm GPU")
    index_dtype_name(torch_module, tensor.dtype, operator)


def validate_same(
    operator: str,
    first: Any,
    second: Any,
    *,
    first_name: str,
    second_name: str,
) -> None:
    if first.device != second.device:
        raise ValueError(f"{operator} {first_name} and {second_name} must use the same device")
    if first.dtype != second.dtype:
        raise TypeError(f"{operator} {first_name} and {second_name} must use the same dtype")


def validate_eps(eps: float, operator: str) -> float:
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError(f"{operator} eps must be a number")
    normalized = float(eps)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{operator} eps must be positive and finite")
    return normalized


def tensor_context(torch_module: Any, operator: str, *tensors: Any) -> TensorContext:
    first = tensors[0]
    device_index = getattr(first.device, "index", None)
    if device_index is None:
        device_index = int(torch_module.cuda.current_device())
    return TensorContext(
        dtype=dtype_name(torch_module, first.dtype, operator),
        device_index=int(device_index),
        contiguous=all(bool(tensor.is_contiguous()) for tensor in tensors),
        requires_grad=any(bool(tensor.requires_grad) for tensor in tensors),
    )
