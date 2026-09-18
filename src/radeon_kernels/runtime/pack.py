from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from radeon_kernels.errors import ConfigError
from radeon_kernels.runtime.compatibility import EnvironmentConstraint
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
from radeon_kernels.runtime.loader import PrecompiledArtifactLoader
from radeon_kernels.runtime.registry import ArtifactSpec


_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_VERSION_RE = re.compile(r"^\d+\.\d+$")
_PACK_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_SUPPORTED_SCHEMA_VERSION = "1.0"


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ConfigError(path, "expected a lowercase identifier")
    return value


def _version(value: Any, path: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ConfigError(path, "invalid version")
    return value


@dataclass(frozen=True)
class KernelPackArtifact:
    operator: str
    semantic_version: str
    artifact: ArtifactSpec
    evidence_id: str

    @classmethod
    def from_mapping(cls, value: Any, path: str) -> KernelPackArtifact:
        if not isinstance(value, Mapping):
            raise ConfigError(path, "expected a mapping")
        fields = {
            "operator",
            "semantic_version",
            "provider",
            "artifact",
            "entrypoint",
            "sha256",
            "format",
            "evidence_id",
        }
        unknown = set(value) - fields
        missing = fields - set(value)
        if unknown:
            raise ConfigError(path, f"unknown fields: {sorted(unknown)}")
        if missing:
            raise ConfigError(path, f"missing fields: {sorted(missing)}")
        artifact = ArtifactSpec.from_mapping(
            {
                "provider": value["provider"],
                "artifact": value["artifact"],
                "entrypoint": value["entrypoint"],
                "sha256": value["sha256"],
                "format": value["format"],
            },
            path,
        )
        return cls(
            operator=_identifier(value["operator"], f"{path}.operator"),
            semantic_version=_version(
                value["semantic_version"], f"{path}.semantic_version", _VERSION_RE
            ),
            artifact=artifact,
            evidence_id=_identifier(value["evidence_id"], f"{path}.evidence_id"),
        )


@dataclass(frozen=True)
class KernelPack:
    root: Path
    schema_version: str
    pack_id: str
    pack_version: str
    environment: EnvironmentConstraint
    artifacts: tuple[KernelPackArtifact, ...]

    @classmethod
    def from_directory(cls, root: Path, *, verify_artifacts: bool = True) -> KernelPack:
        try:
            resolved_root = root.resolve(strict=True)
        except OSError as error:
            raise ConfigError(str(root), "kernel pack directory does not exist") from error
        manifest_path = resolved_root / "manifest.json"
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ConfigError(str(manifest_path), str(error)) from error
        except json.JSONDecodeError as error:
            raise ConfigError(str(manifest_path), f"invalid JSON: {error.msg}") from error
        if not isinstance(value, Mapping):
            raise ConfigError(str(manifest_path), "expected a mapping")

        path = str(manifest_path)
        fields = {"schema_version", "pack_id", "pack_version", "environment", "artifacts"}
        unknown = set(value) - fields
        missing = fields - set(value)
        if unknown:
            raise ConfigError(path, f"unknown fields: {sorted(unknown)}")
        if missing:
            raise ConfigError(path, f"missing fields: {sorted(missing)}")
        schema_version = _version(value["schema_version"], f"{path}.schema_version", _VERSION_RE)
        if schema_version != _SUPPORTED_SCHEMA_VERSION:
            raise ConfigError(f"{path}.schema_version", f"unsupported version {schema_version!r}")
        raw_artifacts = value["artifacts"]
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise ConfigError(f"{path}.artifacts", "expected a non-empty list")

        pack = cls(
            root=resolved_root,
            schema_version=schema_version,
            pack_id=_identifier(value["pack_id"], f"{path}.pack_id"),
            pack_version=_version(value["pack_version"], f"{path}.pack_version", _PACK_VERSION_RE),
            environment=EnvironmentConstraint.from_mapping(value["environment"], f"{path}.environment"),
            artifacts=tuple(
                KernelPackArtifact.from_mapping(item, f"{path}.artifacts[{index}]")
                for index, item in enumerate(raw_artifacts)
            ),
        )
        pack._validate_relations(path)
        if verify_artifacts:
            loader = PrecompiledArtifactLoader(pack.root)
            for item in pack.artifacts:
                loader.resolve_and_verify(item.artifact)
        return pack

    def _validate_relations(self, path: str) -> None:
        keys = [(item.operator, item.semantic_version) for item in self.artifacts]
        if len(set(keys)) != len(keys):
            raise ConfigError(f"{path}.artifacts", "operator semantic versions must be unique")
        artifact_paths = [item.artifact.artifact for item in self.artifacts]
        if len(set(artifact_paths)) != len(artifact_paths):
            raise ConfigError(f"{path}.artifacts", "artifact paths must be unique")

    def matches(self, fingerprint: EnvironmentFingerprint) -> bool:
        return self.environment.matches(fingerprint)

    def get(self, operator: str, semantic_version: str) -> KernelPackArtifact:
        for item in self.artifacts:
            if item.operator == operator and item.semantic_version == semantic_version:
                return item
        raise ConfigError(
            self.pack_id,
            f"pack has no artifact for {operator!r} semantic version {semantic_version!r}",
        )
