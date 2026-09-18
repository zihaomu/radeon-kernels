from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from radeon_kernels.errors import ConfigError
from radeon_kernels.runtime.compatibility import EnvironmentConstraint
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint


_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_VERSION_RE = re.compile(r"^\d+\.\d+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCALAR_TYPES = (str, int, float, bool)
_SUPPORTED_SCHEMA_VERSION = "1.0"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(path, "expected a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(path, "keys must be strings")
    return value


def _strict_fields(
    value: Mapping[str, Any], *, allowed: set[str], required: set[str], path: str
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ConfigError(path, f"unknown fields: {sorted(unknown)}")
    if missing:
        raise ConfigError(path, f"missing fields: {sorted(missing)}")


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ConfigError(path, "expected a lowercase identifier")
    return value


def _version(value: Any, path: str) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise ConfigError(path, "expected MAJOR.MINOR")
    return value


def _scalar(value: Any, path: str) -> str | int | float | bool:
    if value is None or not isinstance(value, _SCALAR_TYPES):
        raise ConfigError(path, "expected a string, number or boolean")
    return value


@dataclass(frozen=True)
class ValueConstraint:
    kind: str
    value: Any

    @classmethod
    def from_raw(cls, value: Any, path: str) -> ValueConstraint:
        if isinstance(value, list):
            if (
                len(value) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
                or value[0] <= 0
                or value[1] < value[0]
            ):
                raise ConfigError(path, "integer ranges must be [minimum, maximum]")
            return cls("range", (value[0], value[1]))
        if isinstance(value, Mapping):
            raw = _mapping(value, path)
            _strict_fields(raw, allowed={"one_of"}, required={"one_of"}, path=path)
            choices = raw["one_of"]
            if isinstance(choices, (str, bytes)) or not isinstance(choices, list) or not choices:
                raise ConfigError(f"{path}.one_of", "expected a non-empty list")
            normalized = tuple(_scalar(item, f"{path}.one_of") for item in choices)
            if len(set(normalized)) != len(normalized):
                raise ConfigError(f"{path}.one_of", "contains duplicate values")
            return cls("one_of", normalized)
        return cls("eq", _scalar(value, path))

    def matches(self, value: Any) -> bool:
        if self.kind == "eq":
            return type(value) is type(self.value) and value == self.value
        if self.kind == "range":
            return (
                isinstance(value, int)
                and not isinstance(value, bool)
                and self.value[0] <= value <= self.value[1]
            )
        return any(type(value) is type(choice) and value == choice for choice in self.value)

    def overlaps(self, other: ValueConstraint) -> bool:
        if self.kind == "range" and other.kind == "range":
            return max(self.value[0], other.value[0]) <= min(self.value[1], other.value[1])
        if self.kind == "range":
            return any(self.matches(item) for item in other.values())
        if other.kind == "range":
            return any(other.matches(item) for item in self.values())
        return any(other.matches(item) for item in self.values())

    def is_subset_of(self, other: ValueConstraint) -> bool:
        if self.kind == "range":
            if other.kind == "range":
                return other.value[0] <= self.value[0] and self.value[1] <= other.value[1]
            if self.value[0] != self.value[1]:
                return False
            return other.matches(self.value[0])
        return all(other.matches(item) for item in self.values())

    def values(self) -> tuple[Any, ...]:
        if self.kind == "eq":
            return (self.value,)
        if self.kind == "one_of":
            return self.value
        if self.value[0] == self.value[1]:
            return (self.value[0],)
        return (self.value[0], self.value[1])

    def to_raw(self) -> Any:
        if self.kind == "range":
            return list(self.value)
        if self.kind == "one_of":
            return {"one_of": list(self.value)}
        return self.value


@dataclass(frozen=True)
class WorkloadConstraint:
    values: Mapping[str, ValueConstraint]

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], path: str
    ) -> WorkloadConstraint:
        raw = _mapping(value, path)
        if not raw:
            raise ConfigError(path, "must contain at least one constraint")
        normalized: dict[str, ValueConstraint] = {}
        for key, item in raw.items():
            if _IDENTIFIER_RE.fullmatch(key) is None:
                raise ConfigError(path, f"invalid workload field: {key!r}")
            normalized[key] = ValueConstraint.from_raw(item, f"{path}.{key}")
        return cls(normalized)

    def matches(self, signature: Mapping[str, Any]) -> bool:
        return all(key in signature and constraint.matches(signature[key]) for key, constraint in self.values.items())

    def overlaps(self, other: WorkloadConstraint) -> bool:
        shared = set(self.values) & set(other.values)
        return all(self.values[key].overlaps(other.values[key]) for key in shared)

    def is_subset_of(self, other: WorkloadConstraint) -> bool:
        for key, constraint in other.values.items():
            if key not in self.values or not self.values[key].is_subset_of(constraint):
                return False
        return True

    def is_strict_subset_of(self, other: WorkloadConstraint) -> bool:
        if not self.is_subset_of(other):
            return False
        if set(self.values) > set(other.values):
            return True
        return any(
            not other.values[key].is_subset_of(self.values[key])
            for key in other.values
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: value.to_raw() for key, value in self.values.items()}


@dataclass(frozen=True)
class ArtifactSpec:
    provider: str
    artifact: str
    entrypoint: str
    sha256: str
    format: str = "python_extension"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], path: str) -> ArtifactSpec:
        raw = _mapping(value, path)
        _strict_fields(
            raw,
            allowed={"provider", "artifact", "entrypoint", "sha256", "format"},
            required={"provider", "artifact", "entrypoint", "sha256"},
            path=path,
        )
        provider = _identifier(raw["provider"], f"{path}.provider")
        artifact = raw["artifact"]
        entrypoint = raw["entrypoint"]
        digest = raw["sha256"]
        artifact_format = raw.get("format", "python_extension")
        if not isinstance(artifact, str) or not artifact:
            raise ConfigError(f"{path}.artifact", "expected a relative path")
        parsed_path = PurePosixPath(artifact)
        if parsed_path.is_absolute() or ".." in parsed_path.parts or parsed_path.as_posix() != artifact:
            raise ConfigError(f"{path}.artifact", "must be a normalized relative path")
        if not isinstance(entrypoint, str) or not entrypoint.isidentifier():
            raise ConfigError(f"{path}.entrypoint", "expected a Python/C identifier")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ConfigError(f"{path}.sha256", "expected a lowercase SHA-256")
        if artifact_format not in {"python_extension", "hsaco"}:
            raise ConfigError(f"{path}.format", "expected python_extension or hsaco")
        return cls(provider, artifact, entrypoint, digest, artifact_format)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "artifact": self.artifact,
            "entrypoint": self.entrypoint,
            "sha256": self.sha256,
            "format": self.format,
        }


@dataclass(frozen=True)
class DispatchEntry:
    id: str
    priority: int
    environment: EnvironmentConstraint
    workload: WorkloadConstraint
    winner: ArtifactSpec
    launch: Mapping[str, str | int | float | bool]
    fallbacks: tuple[str, ...]
    evidence_id: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], path: str) -> DispatchEntry:
        raw = _mapping(value, path)
        fields = {
            "id",
            "priority",
            "environment",
            "workload",
            "winner",
            "launch",
            "fallbacks",
            "evidence_id",
        }
        _strict_fields(raw, allowed=fields, required=fields, path=path)
        priority = raw["priority"]
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
            raise ConfigError(f"{path}.priority", "expected a non-negative integer")
        fallbacks = raw["fallbacks"]
        if isinstance(fallbacks, (str, bytes)) or not isinstance(fallbacks, list):
            raise ConfigError(f"{path}.fallbacks", "expected a list")
        normalized_fallbacks = tuple(
            _identifier(item, f"{path}.fallbacks") for item in fallbacks
        )
        if len(set(normalized_fallbacks)) != len(normalized_fallbacks):
            raise ConfigError(f"{path}.fallbacks", "contains duplicates")
        launch = _mapping(raw["launch"], f"{path}.launch")
        normalized_launch: dict[str, str | int | float | bool] = {}
        for key, item in launch.items():
            if _IDENTIFIER_RE.fullmatch(key) is None:
                raise ConfigError(f"{path}.launch", f"invalid parameter: {key!r}")
            normalized_launch[key] = _scalar(item, f"{path}.launch.{key}")
        return cls(
            id=_identifier(raw["id"], f"{path}.id"),
            priority=priority,
            environment=EnvironmentConstraint.from_mapping(raw["environment"], f"{path}.environment"),
            workload=WorkloadConstraint.from_mapping(raw["workload"], f"{path}.workload"),
            winner=ArtifactSpec.from_mapping(raw["winner"], f"{path}.winner"),
            launch=MappingProxyType(normalized_launch),
            fallbacks=normalized_fallbacks,
            evidence_id=_identifier(raw["evidence_id"], f"{path}.evidence_id"),
        )

    def matches(
        self, fingerprint: EnvironmentFingerprint, signature: Mapping[str, Any]
    ) -> bool:
        return self.environment.matches(fingerprint) and self.workload.matches(signature)


@dataclass(frozen=True)
class DispatchManifest:
    schema_version: str
    operator: str
    semantic_version: str
    default_fallbacks: tuple[str, ...]
    entries: tuple[DispatchEntry, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], path: str = "dispatch") -> DispatchManifest:
        raw = _mapping(value, path)
        fields = {
            "schema_version",
            "operator",
            "semantic_version",
            "default_fallbacks",
            "entries",
        }
        _strict_fields(raw, allowed=fields, required=fields, path=path)
        entries = raw["entries"]
        fallbacks = raw["default_fallbacks"]
        if isinstance(entries, (str, bytes)) or not isinstance(entries, list):
            raise ConfigError(f"{path}.entries", "expected a list")
        if isinstance(fallbacks, (str, bytes)) or not isinstance(fallbacks, list) or not fallbacks:
            raise ConfigError(f"{path}.default_fallbacks", "expected a non-empty list")

        schema_version = _version(raw["schema_version"], f"{path}.schema_version")
        if schema_version != _SUPPORTED_SCHEMA_VERSION:
            raise ConfigError(
                f"{path}.schema_version",
                f"unsupported dispatch schema version {schema_version!r}",
            )

        manifest = cls(
            schema_version=schema_version,
            operator=_identifier(raw["operator"], f"{path}.operator"),
            semantic_version=_version(raw["semantic_version"], f"{path}.semantic_version"),
            default_fallbacks=tuple(
                _identifier(item, f"{path}.default_fallbacks") for item in fallbacks
            ),
            entries=tuple(
                DispatchEntry.from_mapping(item, f"{path}.entries[{index}]")
                for index, item in enumerate(entries)
            ),
        )
        manifest._validate_relations(path)
        return manifest

    def _validate_relations(self, path: str) -> None:
        if len(set(self.default_fallbacks)) != len(self.default_fallbacks):
            raise ConfigError(f"{path}.default_fallbacks", "contains duplicates")
        identifiers = [entry.id for entry in self.entries]
        if len(set(identifiers)) != len(identifiers):
            raise ConfigError(f"{path}.entries", "entry ids must be unique")

        for left_index, left in enumerate(self.entries):
            for right in self.entries[left_index + 1 :]:
                if not left.environment.overlaps(right.environment):
                    continue
                if not left.workload.overlaps(right.workload):
                    continue
                if left.priority == right.priority:
                    raise ConfigError(
                        f"{path}.entries",
                        f"ambiguous overlapping entries {left.id!r} and {right.id!r}",
                    )
                high, low = (left, right) if left.priority > right.priority else (right, left)
                environment_subset = high.environment.is_subset_of(low.environment)
                workload_subset = high.workload.is_subset_of(low.workload)
                strict = high.environment.is_strict_subset_of(low.environment) or high.workload.is_strict_subset_of(low.workload)
                if not (environment_subset and workload_subset and strict):
                    raise ConfigError(
                        f"{path}.entries",
                        f"higher-priority entry {high.id!r} must be a strict subset of {low.id!r}",
                    )

    @classmethod
    def from_file(cls, path: Path) -> DispatchManifest:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ConfigError(str(path), str(error)) from error
        except json.JSONDecodeError as error:
            raise ConfigError(str(path), f"invalid JSON: {error.msg}") from error
        return cls.from_mapping(payload, str(path))


class DispatchRegistry:
    def __init__(self, manifests: Sequence[DispatchManifest] = ()) -> None:
        self._manifests: dict[tuple[str, str], DispatchManifest] = {}
        for manifest in manifests:
            self.add(manifest)

    def add(self, manifest: DispatchManifest) -> None:
        key = (manifest.operator, manifest.semantic_version)
        if key in self._manifests:
            raise ConfigError("dispatch", f"duplicate manifest for {key[0]} {key[1]}")
        self._manifests[key] = manifest

    def get(self, operator: str, semantic_version: str) -> DispatchManifest:
        key = (operator, semantic_version)
        try:
            return self._manifests[key]
        except KeyError as error:
            raise ConfigError(
                "dispatch",
                f"no manifest for operator {operator!r} semantic version {semantic_version!r}",
            ) from error

    @classmethod
    def from_directory(cls, directory: Path) -> DispatchRegistry:
        if not directory.is_dir():
            raise ConfigError(str(directory), "dispatch directory does not exist")
        files = sorted(directory.rglob("*.json"))
        if not files:
            raise ConfigError(str(directory), "contains no dispatch manifests")
        return cls(DispatchManifest.from_file(path) for path in files)

    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._manifests))
