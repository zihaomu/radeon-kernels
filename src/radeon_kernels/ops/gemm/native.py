from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


SUPPORTED_ARCHITECTURES = frozenset({"gfx1151", "gfx1201"})
VARIANT_NAMES = (
    "m128_n64_wt64x32_k16",
    "m64_n128_wt32x64_k16",
    "m128_n64_wt32x32_k16",
    "m256_n32_wt64x32_k16",
    "m128_n64_wt32x64_k16",
    "m128_n64_wt64x32_k32",
    "m64_n128_wt32x64_k32",
    "m64_n256_wt32x64_k32",
    "m128_n128_wt32x64_k32",
    "m64_n128_wt32x64_k64",
    "m128_n128_wt32x64_k64",
)


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("native GEMM requires a ROCm PyTorch installation") from error
    return torch


def current_architecture() -> str:
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("native GEMM requires a visible ROCm GPU")
    architecture = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    architecture = architecture.split(":", 1)[0]
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise RuntimeError(
            f"native GEMM supports {sorted(SUPPORTED_ARCHITECTURES)}, got {architecture!r}"
        )
    return architecture


@lru_cache(maxsize=1)
def load_native_extension(*, verbose: bool = False) -> Any:
    torch = _torch()
    from torch.utils.cpp_extension import load

    architecture = current_architecture()
    source = Path(__file__).with_name("native") / "wmma_gemm.hip"
    os.environ["PYTORCH_ROCM_ARCH"] = architecture
    os.environ.setdefault("MAX_JOBS", "2")
    return load(
        name=f"radeon_kernels_wmma_{architecture}",
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-ffast-math",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
        ],
        with_cuda=True,
        verbose=verbose,
    )


def gemm(a: Any, b: Any, *, variant: int = 0) -> Any:
    """Compute row-major ``a @ b`` with an architecture-native WMMA kernel."""

    if not isinstance(variant, int) or not 0 <= variant < len(VARIANT_NAMES):
        raise ValueError(f"variant must be between 0 and {len(VARIANT_NAMES) - 1}")
    extension = load_native_extension()
    return extension.gemm(a, b, variant)


def gemm_out(a: Any, b: Any, output: Any, *, variant: int = 0) -> Any:
    """Compute ``a @ b`` into a caller-provided contiguous output tensor."""

    if not isinstance(variant, int) or not 0 <= variant < len(VARIANT_NAMES):
        raise ValueError(f"variant must be between 0 and {len(VARIANT_NAMES) - 1}")
    extension = load_native_extension()
    return extension.gemm_out(a, b, output, variant)


__all__ = [
    "SUPPORTED_ARCHITECTURES",
    "VARIANT_NAMES",
    "current_architecture",
    "gemm",
    "gemm_out",
    "load_native_extension",
]
