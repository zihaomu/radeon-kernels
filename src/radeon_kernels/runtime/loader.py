from __future__ import annotations

import hashlib
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from radeon_kernels.errors import RadeonKernelsError
from radeon_kernels.runtime.registry import ArtifactSpec


_PYTHON_INIT_SYMBOL = re.compile(
    rb"(?<![^\x00])PyInit_([A-Za-z_][A-Za-z0-9_]*)\x00"
)


class ArtifactError(RadeonKernelsError):
    """Base class for precompiled kernel-pack failures."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when an artifact path or digest is invalid."""


class ArtifactLoadError(ArtifactError):
    """Raised when a verified artifact cannot be loaded."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_extension_module_name(path: Path) -> str:
    try:
        exported_names = {
            match.group(1).decode("ascii") for match in _PYTHON_INIT_SYMBOL.finditer(path.read_bytes())
        }
    except OSError as error:
        raise ArtifactLoadError(f"cannot inspect Python extension {path.name}: {error}") from error
    if len(exported_names) > 1:
        raise ArtifactLoadError(
            f"Python extension {path.name} exports multiple PyInit symbols: "
            f"{sorted(exported_names)}"
        )
    if exported_names:
        return next(iter(exported_names))
    return path.name.split(".", 1)[0]


@dataclass(frozen=True)
class LoadedEntrypoint:
    artifact: Path
    provider: str
    entrypoint: str
    callable: Callable[..., Any]


class PrecompiledArtifactLoader:
    def __init__(self, pack_root: Path) -> None:
        try:
            self._pack_root = pack_root.resolve(strict=True)
        except OSError as error:
            raise ArtifactIntegrityError(f"kernel pack root is unavailable: {pack_root}") from error
        if not self._pack_root.is_dir():
            raise ArtifactIntegrityError(f"kernel pack root is not a directory: {pack_root}")
        self._modules: dict[Path, ModuleType] = {}

    def resolve_and_verify(self, artifact: ArtifactSpec) -> Path:
        candidate = (self._pack_root / artifact.artifact).resolve()
        if not candidate.is_relative_to(self._pack_root):
            raise ArtifactIntegrityError("artifact escapes the kernel pack root")
        if not candidate.is_file():
            raise ArtifactIntegrityError(f"artifact is missing: {artifact.artifact}")
        actual = _sha256(candidate)
        if actual != artifact.sha256:
            raise ArtifactIntegrityError(
                f"artifact SHA-256 mismatch for {artifact.artifact}: expected {artifact.sha256}, got {actual}"
            )
        return candidate

    def load_module(self, artifact: ArtifactSpec) -> ModuleType:
        path = self.resolve_and_verify(artifact)
        if artifact.format != "python_extension":
            raise ArtifactLoadError(
                f"artifact format {artifact.format!r} requires a provider-specific launcher"
            )
        if path in self._modules:
            return self._modules[path]

        module_name = _python_extension_module_name(path)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ArtifactLoadError(f"cannot create an import specification for {path.name}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise ArtifactLoadError(f"failed to load {path.name}: {error}") from error
        self._modules[path] = module
        return module

    def load_entrypoint(self, artifact: ArtifactSpec) -> LoadedEntrypoint:
        module = self.load_module(artifact)
        function = getattr(module, artifact.entrypoint, None)
        if not callable(function):
            raise ArtifactLoadError(
                f"artifact {artifact.artifact} does not export callable {artifact.entrypoint!r}"
            )
        return LoadedEntrypoint(
            artifact=self.resolve_and_verify(artifact),
            provider=artifact.provider,
            entrypoint=artifact.entrypoint,
            callable=function,
        )
