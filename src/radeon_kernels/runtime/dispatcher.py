from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from radeon_kernels.runtime.explain import record_dispatch
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
from radeon_kernels.runtime.registry import ArtifactSpec, DispatchEntry, DispatchRegistry


@dataclass(frozen=True)
class DispatchRequest:
    operator: str
    semantic_version: str
    signature: Mapping[str, Any]


@dataclass(frozen=True)
class DispatchDecision:
    operator: str
    semantic_version: str
    fingerprint: EnvironmentFingerprint
    entry_id: str | None
    winner: ArtifactSpec | None
    launch: Mapping[str, str | int | float | bool]
    fallbacks: tuple[str, ...]
    evidence_id: str | None
    reason: str

    @property
    def matched(self) -> bool:
        return self.winner is not None

    def to_dict(self) -> dict[str, Any]:
        selected = None
        artifact = None
        if self.winner is not None:
            selected = f"{self.winner.provider}/{self.winner.entrypoint}"
            artifact = self.winner.artifact
        return {
            "operator": self.operator,
            "semantic_version": self.semantic_version,
            "architecture": self.fingerprint.architecture,
            "rocm_abi": self.fingerprint.rocm_abi,
            "python_abi": self.fingerprint.python_abi,
            "pytorch_version": self.fingerprint.pytorch_version,
            "entry_id": self.entry_id,
            "selected": selected,
            "artifact": artifact,
            "launch": dict(self.launch),
            "fallbacks": list(self.fallbacks),
            "evidence_id": self.evidence_id,
            "reason": self.reason,
        }


class Dispatcher:
    def __init__(self, registry: DispatchRegistry) -> None:
        self._registry = registry

    def select(
        self,
        request: DispatchRequest,
        fingerprint: EnvironmentFingerprint,
    ) -> DispatchDecision:
        manifest = self._registry.get(request.operator, request.semantic_version)
        matches: list[DispatchEntry] = [
            entry
            for entry in manifest.entries
            if entry.matches(fingerprint, request.signature)
        ]
        matches.sort(key=lambda entry: entry.priority, reverse=True)

        if matches:
            entry = matches[0]
            decision = DispatchDecision(
                operator=request.operator,
                semantic_version=request.semantic_version,
                fingerprint=fingerprint,
                entry_id=entry.id,
                winner=entry.winner,
                launch=entry.launch,
                fallbacks=entry.fallbacks,
                evidence_id=entry.evidence_id,
                reason=f"matched dispatch entry {entry.id!r} at priority {entry.priority}",
            )
        else:
            decision = DispatchDecision(
                operator=request.operator,
                semantic_version=request.semantic_version,
                fingerprint=fingerprint,
                entry_id=None,
                winner=None,
                launch={},
                fallbacks=manifest.default_fallbacks,
                evidence_id=None,
                reason="no compatible environment and workload region matched",
            )
        record_dispatch(decision.to_dict())
        return decision
