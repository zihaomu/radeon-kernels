from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from radeon_kernels.errors import RadeonKernelsError


_ARCHITECTURE_RE = re.compile(r"^gfx[0-9a-f]+$")
_ROCM_VERSION_RE = re.compile(r"^(\d+)\.(\d+)")
_PYTHON_ABI_RE = re.compile(r"^cp\d{2,3}$")


class FingerprintError(RadeonKernelsError):
    """Raised when the local ROCm environment cannot be identified safely."""


def normalize_rocm_abi(value: str) -> str:
    match = _ROCM_VERSION_RE.match(value.strip())
    if match is None:
        raise FingerprintError(f"invalid ROCm version: {value!r}")
    return f"{int(match.group(1))}.{int(match.group(2))}"


def _features(values: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or not value.replace("_", "").isalnum():
            raise FingerprintError(f"invalid architecture feature: {value!r}")
        normalized.append(value)
    return tuple(sorted(set(normalized)))


@dataclass(frozen=True)
class EnvironmentFingerprint:
    architecture: str
    wavefront_size: int
    rocm_abi: str
    python_abi: str
    pytorch_version: str
    features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if _ARCHITECTURE_RE.fullmatch(self.architecture) is None:
            raise FingerprintError(f"invalid AMDGPU architecture: {self.architecture!r}")
        if self.wavefront_size not in (32, 64):
            raise FingerprintError("wavefront size must be 32 or 64")
        object.__setattr__(self, "rocm_abi", normalize_rocm_abi(self.rocm_abi))
        if _PYTHON_ABI_RE.fullmatch(self.python_abi) is None:
            raise FingerprintError("Python ABI must use the cpXY form")
        if not self.pytorch_version:
            raise FingerprintError("PyTorch version must not be empty")
        object.__setattr__(self, "features", _features(self.features))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EnvironmentFingerprint:
        allowed = {
            "architecture",
            "wavefront_size",
            "rocm_abi",
            "python_abi",
            "pytorch_version",
            "features",
        }
        unknown = set(value) - allowed
        if unknown:
            raise FingerprintError(f"unknown fingerprint fields: {sorted(unknown)}")
        try:
            features = value.get("features", ())
            if isinstance(features, (str, bytes)):
                raise TypeError
            return cls(
                architecture=value["architecture"],
                wavefront_size=value["wavefront_size"],
                rocm_abi=value["rocm_abi"],
                python_abi=value["python_abi"],
                pytorch_version=value["pytorch_version"],
                features=tuple(features),
            )
        except (KeyError, TypeError) as error:
            raise FingerprintError("fingerprint has missing or invalid fields") from error

    @classmethod
    def detect(
        cls,
        *,
        torch_module: Any | None = None,
        device_index: int = 0,
        features: Iterable[str] = (),
        python_abi: str | None = None,
    ) -> EnvironmentFingerprint:
        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as error:
                raise FingerprintError("PyTorch is required to detect the ROCm environment") from error

        if not torch_module.cuda.is_available():
            raise FingerprintError("no visible ROCm GPU")
        rocm_version = getattr(torch_module.version, "hip", None)
        if not isinstance(rocm_version, str) or not rocm_version:
            raise FingerprintError("the installed PyTorch build does not report a ROCm version")

        properties = torch_module.cuda.get_device_properties(device_index)
        architecture = getattr(properties, "gcnArchName", "").split(":", 1)[0]
        wavefront_size = getattr(properties, "warp_size", None)
        if not isinstance(wavefront_size, int):
            raise FingerprintError("PyTorch did not report the GPU wavefront size")

        return cls(
            architecture=architecture,
            wavefront_size=wavefront_size,
            rocm_abi=rocm_version,
            python_abi=python_abi or f"cp{sys.version_info.major}{sys.version_info.minor}",
            pytorch_version=str(torch_module.__version__),
            features=tuple(features),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "wavefront_size": self.wavefront_size,
            "rocm_abi": self.rocm_abi,
            "python_abi": self.python_abi,
            "pytorch_version": self.pytorch_version,
            "features": list(self.features),
        }
