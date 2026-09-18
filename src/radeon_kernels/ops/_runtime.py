from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Mapping

from radeon_kernels.errors import RadeonKernelsError
from radeon_kernels.providers import native
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


class OperatorDispatchError(RadeonKernelsError):
    """Raised when neither a published winner nor the declared fallback can run."""


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


class OperatorRuntime:
    """Shared static-dispatch executor for published non-GEMM operators."""

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

    def execute(
        self,
        *,
        operator: str,
        semantic_version: str,
        signature: Mapping[str, Any],
        device_index: int,
        torch_module: Any,
        native_arguments: tuple[Any, ...],
        fallback: Callable[[], Any],
        fallback_selected: str,
    ) -> Any:
        fingerprint = self._fingerprint(torch_module, device_index)
        decision = self._dispatcher.select(
            DispatchRequest(operator, semantic_version, signature), fingerprint
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
                result = native.execute(
                    self._loader(resolved.pack.root),
                    decision.winner,
                    decision.launch,
                    *native_arguments,
                )
                _record_execution(
                    decision,
                    selected=f"{decision.winner.provider}/{decision.winner.entrypoint}",
                    artifact=decision.winner.artifact,
                    evidence_id=decision.evidence_id,
                    reason=(
                        f"{decision.reason}; loaded pack {resolved.pack.pack_id} "
                        f"{resolved.pack.pack_version}"
                    ),
                )
                return result
            except (
                PackResolutionError,
                ArtifactError,
                native.NativeProviderError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                winner_failure = str(error)

        if "torch" in decision.fallbacks:
            try:
                result = fallback()
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
                selected=fallback_selected,
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
        raise OperatorDispatchError(
            f"no approved implementation is available for {operator} {semantic_version}"
        )


_DEFAULT_RUNTIME: OperatorRuntime | None = None
_DEFAULT_RUNTIME_LOCK = Lock()


def default_operator_runtime() -> OperatorRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        with _DEFAULT_RUNTIME_LOCK:
            if _DEFAULT_RUNTIME is None:
                _DEFAULT_RUNTIME = OperatorRuntime()
    return _DEFAULT_RUNTIME


def reset_operator_runtime() -> None:
    global _DEFAULT_RUNTIME
    with _DEFAULT_RUNTIME_LOCK:
        _DEFAULT_RUNTIME = None


__all__ = [
    "OperatorDispatchError",
    "OperatorRuntime",
    "default_operator_runtime",
    "reset_operator_runtime",
]
