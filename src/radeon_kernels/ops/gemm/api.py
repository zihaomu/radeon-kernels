from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable

from radeon_kernels.errors import RadeonKernelsError
from radeon_kernels.providers import native
from radeon_kernels.providers import torch_fallback
from radeon_kernels.runtime.dispatcher import DispatchDecision, DispatchRequest, Dispatcher
from radeon_kernels.runtime.explain import record_dispatch
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
from radeon_kernels.runtime.loader import ArtifactError, PrecompiledArtifactLoader
from radeon_kernels.runtime.packs import (
    KernelPackRegistry,
    PackResolutionError,
    default_pack_roots,
)
from radeon_kernels.runtime.registry import DispatchRegistry
from radeon_kernels.runtime.resources import builtin_dispatch_registry


_SEMANTIC_VERSION = "1.0"


class GemmDispatchError(RadeonKernelsError):
    """Raised when neither an approved GEMM winner nor a fallback can execute."""


@dataclass(frozen=True)
class _GemmCall:
    a: Any
    b: Any
    output: Any | None
    signature: dict[str, str | int | bool]
    device_index: int


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise GemmDispatchError("GEMM requires a ROCm PyTorch installation") from error
    return torch


def _dtype_name(torch_module: Any, dtype: Any) -> str:
    if dtype == torch_module.float16:
        return "fp16"
    if dtype == torch_module.bfloat16:
        return "bf16"
    raise TypeError("gemm supports torch.float16 and torch.bfloat16 tensors")


def _validate_call(torch_module: Any, a: Any, b: Any, output: Any | None) -> _GemmCall:
    tensor_type = getattr(torch_module, "Tensor", ())
    if not isinstance(a, tensor_type) or not isinstance(b, tensor_type):
        raise TypeError("gemm inputs must be PyTorch tensors")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("gemm inputs must be rank-2 matrices")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("gemm inputs must be on a visible ROCm GPU")
    if a.device != b.device:
        raise ValueError("gemm inputs must be on the same device")
    if a.dtype != b.dtype:
        raise TypeError("gemm inputs must have the same dtype")
    if a.shape[1] != b.shape[0]:
        raise ValueError("gemm inner dimensions must match")

    dtype = _dtype_name(torch_module, a.dtype)
    m, k = (int(value) for value in a.shape)
    n = int(b.shape[1])
    if output is not None:
        if not isinstance(output, tensor_type):
            raise TypeError("gemm output must be a PyTorch tensor")
        if tuple(output.shape) != (m, n):
            raise ValueError(f"gemm output must have shape {(m, n)}")
        if output.device != a.device or output.dtype != a.dtype:
            raise ValueError("gemm output must match the input device and dtype")
        if not output.is_contiguous():
            raise ValueError("gemm output must be contiguous")

    device_index = getattr(a.device, "index", None)
    if device_index is None:
        device_index = int(torch_module.cuda.current_device())
    return _GemmCall(
        a=a,
        b=b,
        output=output,
        signature={
            "dtype": dtype,
            "layout": "nn",
            "m": m,
            "n": n,
            "k": k,
            "contiguous": bool(a.is_contiguous() and b.is_contiguous()),
            "requires_grad": bool(a.requires_grad or b.requires_grad),
        },
        device_index=int(device_index),
    )


def _record_execution(
    decision: DispatchDecision,
    *,
    selected: str,
    artifact: str | None,
    evidence_id: str | None,
    reason: str,
) -> None:
    record = decision.to_dict()
    record.update(
        {
            "selected": selected,
            "artifact": artifact,
            "evidence_id": evidence_id,
            "reason": reason,
        }
    )
    record_dispatch(record)


class GemmRuntime:
    def __init__(
        self,
        *,
        registry: DispatchRegistry | None = None,
        pack_roots: Iterable[Path] | None = None,
        pack_registry: KernelPackRegistry | None = None,
        require_pack_signatures: bool = True,
        loader_factory: Callable[[Path], PrecompiledArtifactLoader] = PrecompiledArtifactLoader,
        fingerprint_detector: Callable[..., EnvironmentFingerprint] = EnvironmentFingerprint.detect,
    ) -> None:
        self._dispatcher = Dispatcher(registry or builtin_dispatch_registry())
        self._pack_registry = pack_registry or KernelPackRegistry.discover(
            tuple(pack_roots) if pack_roots is not None else default_pack_roots(),
            require_signatures=require_pack_signatures,
        )
        self._loader_factory = loader_factory
        self._fingerprint_detector = fingerprint_detector
        self._loaders: dict[Path, PrecompiledArtifactLoader] = {}
        self._fingerprints: dict[int, EnvironmentFingerprint] = {}

    def _fingerprint(self, torch_module: Any, device_index: int) -> EnvironmentFingerprint:
        if device_index not in self._fingerprints:
            self._fingerprints[device_index] = self._fingerprint_detector(
                torch_module=torch_module,
                device_index=device_index,
            )
        return self._fingerprints[device_index]

    def _loader(self, root: Path) -> PrecompiledArtifactLoader:
        if root not in self._loaders:
            self._loaders[root] = self._loader_factory(root)
        return self._loaders[root]

    def execute(self, a: Any, b: Any, *, output: Any | None = None, torch_module: Any | None = None) -> Any:
        torch_runtime = torch_module or _torch()
        call = _validate_call(torch_runtime, a, b, output)
        fingerprint = self._fingerprint(torch_runtime, call.device_index)
        decision = self._dispatcher.select(
            DispatchRequest("gemm", _SEMANTIC_VERSION, call.signature),
            fingerprint,
        )

        winner_failure: str | None = None
        if decision.matched:
            try:
                resolved = self._pack_registry.resolve(
                    fingerprint=fingerprint,
                    operator=decision.operator,
                    semantic_version=decision.semantic_version,
                    artifact=decision.winner,
                    evidence_id=decision.evidence_id,
                )
                target = (
                    call.output
                    if call.output is not None
                    else call.a.new_empty((call.signature["m"], call.signature["n"]))
                )
                result = native.gemm(
                    self._loader(resolved.pack.root),
                    decision.winner,
                    decision.launch,
                    call.a,
                    call.b,
                    target,
                )
                _record_execution(
                    decision,
                    selected=f"{decision.winner.provider}/{decision.winner.entrypoint}",
                    artifact=decision.winner.artifact,
                    evidence_id=decision.evidence_id,
                    reason=f"{decision.reason}; loaded pack {resolved.pack.pack_id} {resolved.pack.pack_version}",
                )
                return result
            except (PackResolutionError, ArtifactError, native.NativeProviderError, RuntimeError) as error:
                winner_failure = str(error)

        for provider in decision.fallbacks:
            if provider != "torch":
                continue
            try:
                result = torch_fallback.gemm(torch_runtime, call.a, call.b, call.output)
            except Exception as error:
                _record_execution(
                    decision,
                    selected="none",
                    artifact=None,
                    evidence_id=None,
                    reason=f"torch fallback failed: {error}",
                )
                raise
            reason = decision.reason
            if winner_failure is not None:
                reason = f"{reason}; winner unavailable: {winner_failure}"
            _record_execution(
                decision,
                selected="torch/mm",
                artifact=None,
                evidence_id=None,
                reason=f"{reason}; executed declared torch fallback",
            )
            return result

        reason = decision.reason
        if winner_failure is not None:
            reason = f"{reason}; winner unavailable: {winner_failure}"
        _record_execution(
            decision,
            selected="none",
            artifact=None,
            evidence_id=None,
            reason=f"{reason}; no supported fallback is available",
        )
        raise GemmDispatchError("no approved GEMM implementation is available")


_DEFAULT_RUNTIME: GemmRuntime | None = None
_DEFAULT_RUNTIME_LOCK = Lock()


def _default_runtime() -> GemmRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        with _DEFAULT_RUNTIME_LOCK:
            if _DEFAULT_RUNTIME is None:
                _DEFAULT_RUNTIME = GemmRuntime()
    return _DEFAULT_RUNTIME


def reset_runtime() -> None:
    """Reload dispatch and installed packs before the next public GEMM call."""

    global _DEFAULT_RUNTIME
    with _DEFAULT_RUNTIME_LOCK:
        _DEFAULT_RUNTIME = None


def gemm(a: Any, b: Any, *, out: Any | None = None) -> Any:
    """Compute ``a @ b`` using a published winner or deterministic fallback."""

    return _default_runtime().execute(a, b, output=out)


__all__ = ["GemmDispatchError", "GemmRuntime", "gemm", "reset_runtime"]
