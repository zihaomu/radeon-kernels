from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from radeon_kernels.errors import ConfigError
from radeon_kernels.runtime.fingerprint import (
    EnvironmentFingerprint,
    FingerprintError,
    normalize_rocm_abi,
)


_FIELDS = {
    "architecture",
    "wavefront_size",
    "rocm_abi",
    "python_abi",
    "pytorch_version",
    "features",
}
_ARCHITECTURE_RE = re.compile(r"^gfx[0-9a-f]+$")


@dataclass(frozen=True)
class EnvironmentConstraint:
    architecture: str
    wavefront_size: int
    rocm_abi: str
    python_abi: str
    pytorch_version: str
    features: frozenset[str] = frozenset()

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], path: str
    ) -> EnvironmentConstraint:
        if not isinstance(value, Mapping):
            raise ConfigError(path, "expected a mapping")
        unknown = set(value) - _FIELDS
        if unknown:
            raise ConfigError(path, f"unknown fields: {sorted(unknown)}")
        missing = {
            "architecture",
            "wavefront_size",
            "rocm_abi",
            "python_abi",
            "pytorch_version",
        } - set(value)
        if missing:
            raise ConfigError(path, f"missing fields: {sorted(missing)}")

        architecture = value["architecture"]
        wavefront_size = value["wavefront_size"]
        features = value.get("features", [])
        python_abi = value["python_abi"]
        pytorch_version = value["pytorch_version"]
        if not isinstance(architecture, str) or _ARCHITECTURE_RE.fullmatch(architecture) is None:
            raise ConfigError(f"{path}.architecture", "expected a gfx architecture")
        if isinstance(wavefront_size, bool) or wavefront_size not in (32, 64):
            raise ConfigError(f"{path}.wavefront_size", "expected 32 or 64")
        if isinstance(features, (str, bytes)) or not isinstance(features, list):
            raise ConfigError(f"{path}.features", "expected a list")
        if any(
            not isinstance(item, str)
            or not item
            or not item.replace("_", "").isalnum()
            for item in features
        ):
            raise ConfigError(f"{path}.features", "contains an invalid feature")
        if (
            not isinstance(python_abi, str)
            or not python_abi.startswith("cp")
            or not python_abi[2:].isdigit()
        ):
            raise ConfigError(f"{path}.python_abi", "expected a cpXY ABI tag")
        if not isinstance(pytorch_version, str) or not pytorch_version:
            raise ConfigError(f"{path}.pytorch_version", "expected an exact build version")
        try:
            rocm_abi = normalize_rocm_abi(value["rocm_abi"])
        except (FingerprintError, AttributeError) as error:
            raise ConfigError(f"{path}.rocm_abi", "expected MAJOR.MINOR") from error

        return cls(
            architecture=architecture,
            wavefront_size=wavefront_size,
            rocm_abi=rocm_abi,
            python_abi=python_abi,
            pytorch_version=pytorch_version,
            features=frozenset(features),
        )

    def matches(self, fingerprint: EnvironmentFingerprint) -> bool:
        return (
            self.architecture == fingerprint.architecture
            and self.wavefront_size == fingerprint.wavefront_size
            and self.rocm_abi == fingerprint.rocm_abi
            and self.python_abi == fingerprint.python_abi
            and self.pytorch_version == fingerprint.pytorch_version
            and self.features.issubset(fingerprint.features)
        )

    def overlaps(self, other: EnvironmentConstraint) -> bool:
        return (
            self.architecture == other.architecture
            and self.wavefront_size == other.wavefront_size
            and self.rocm_abi == other.rocm_abi
            and self.python_abi == other.python_abi
            and self.pytorch_version == other.pytorch_version
        )

    def is_subset_of(self, other: EnvironmentConstraint) -> bool:
        return self.overlaps(other) and self.features.issuperset(other.features)

    def is_strict_subset_of(self, other: EnvironmentConstraint) -> bool:
        return self.is_subset_of(other) and self.features != other.features

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "wavefront_size": self.wavefront_size,
            "rocm_abi": self.rocm_abi,
            "python_abi": self.python_abi,
            "pytorch_version": self.pytorch_version,
            "features": sorted(self.features),
        }
