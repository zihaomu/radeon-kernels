from __future__ import annotations

from typing import Any, Mapping

from radeon_kernels.errors import RadeonKernelsError
from radeon_kernels.runtime.loader import PrecompiledArtifactLoader
from radeon_kernels.runtime.registry import ArtifactSpec


class NativeProviderError(RadeonKernelsError):
    """Raised when an approved native launch cannot be executed safely."""


def execute(
    loader: PrecompiledArtifactLoader,
    artifact: ArtifactSpec,
    launch: Mapping[str, str | int | float | bool],
    *arguments: Any,
) -> Any:
    """Invoke a published native entrypoint without operator-specific policy."""

    if artifact.provider != "native":
        raise NativeProviderError(f"unsupported native winner provider: {artifact.provider!r}")
    loaded = loader.load_entrypoint(artifact)
    returned = loaded.callable(*arguments, **dict(launch))
    if returned is None:
        raise NativeProviderError("native entrypoint returned no output")
    return returned


def gemm(
    loader: PrecompiledArtifactLoader,
    artifact: ArtifactSpec,
    launch: Mapping[str, str | int | float | bool],
    a: Any,
    b: Any,
    output: Any,
) -> Any:
    if artifact.provider != "native":
        raise NativeProviderError(f"unsupported GEMM winner provider: {artifact.provider!r}")
    if set(launch) != {"variant"}:
        raise NativeProviderError("native GEMM launch must contain exactly one variant")
    variant = launch["variant"]
    if isinstance(variant, bool) or not isinstance(variant, int) or variant < 0:
        raise NativeProviderError("native GEMM variant must be a non-negative integer")

    loaded = loader.load_entrypoint(artifact)
    returned = loaded.callable(a, b, output, variant)
    if returned is None:
        raise NativeProviderError("native GEMM returned no output tensor")
    if hasattr(returned, "data_ptr") and returned.data_ptr() != output.data_ptr():
        raise NativeProviderError("native GEMM did not return the caller-provided output")
    return returned
