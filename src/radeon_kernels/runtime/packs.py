from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from radeon_kernels.errors import ConfigError, RadeonKernelsError
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
from radeon_kernels.runtime.pack import KernelPack, KernelPackArtifact
from radeon_kernels.runtime.registry import ArtifactSpec
from radeon_kernels.runtime.signing import (
    SignatureVerificationError,
    TrustedKeyring,
    builtin_trusted_keyring,
    verify_detached_file,
)

PACK_PATH_ENVIRONMENT_VARIABLE = "RADEON_KERNELS_PACK_PATH"


class PackResolutionError(RadeonKernelsError):
    """Raised when no installed pack can satisfy an approved dispatch entry."""


@dataclass(frozen=True)
class ResolvedKernelPack:
    pack: KernelPack
    artifact: KernelPackArtifact


def default_pack_roots(
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    prefix: Path | None = None,
) -> tuple[Path, ...]:
    environment = os.environ if environ is None else environ
    roots: list[Path] = []
    configured = environment.get(PACK_PATH_ENVIRONMENT_VARIABLE, "")
    roots.extend(Path(item).expanduser() for item in configured.split(os.pathsep) if item)
    roots.append((home or Path.home()) / ".local" / "share" / "radeon-kernels" / "packs")
    roots.append((prefix or Path(sys.prefix)) / "share" / "radeon-kernels" / "packs")

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        normalized = root.absolute()
        if normalized not in seen:
            unique.append(normalized)
            seen.add(normalized)
    return tuple(unique)


def _pack_directories(roots: Iterable[Path]) -> tuple[Path, ...]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw_root in roots:
        root = raw_root.expanduser()
        candidates: Sequence[Path]
        if (root / "manifest.json").is_file():
            candidates = (root,)
        elif root.is_dir():
            candidates = tuple(
                child for child in sorted(root.iterdir()) if (child / "manifest.json").is_file()
            )
        else:
            candidates = ()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                discovered.append(resolved)
                seen.add(resolved)
    return tuple(discovered)


class KernelPackRegistry:
    def __init__(
        self,
        packs: Sequence[KernelPack] = (),
        *,
        rejected: Sequence[str] = (),
    ) -> None:
        self._packs = tuple(packs)
        self._rejected = tuple(rejected)

    @classmethod
    def discover(
        cls,
        roots: Iterable[Path],
        *,
        require_signatures: bool = True,
        trusted_keys: TrustedKeyring | None = None,
    ) -> KernelPackRegistry:
        packs: list[KernelPack] = []
        rejected: list[str] = []
        keyring = trusted_keys
        if require_signatures and keyring is None:
            keyring = builtin_trusted_keyring()
        for directory in _pack_directories(roots):
            try:
                if require_signatures:
                    verify_detached_file(
                        directory / "manifest.json",
                        directory / "manifest.sig.json",
                        keyring,
                        signed_file="manifest.json",
                    )
                packs.append(KernelPack.from_directory(directory, verify_artifacts=False))
            except (ConfigError, SignatureVerificationError) as error:
                rejected.append(f"{directory}: {error}")
        return cls(packs, rejected=rejected)

    @property
    def packs(self) -> tuple[KernelPack, ...]:
        return self._packs

    @property
    def rejected(self) -> tuple[str, ...]:
        return self._rejected

    def resolve(
        self,
        *,
        fingerprint: EnvironmentFingerprint,
        operator: str,
        semantic_version: str,
        artifact: ArtifactSpec,
        evidence_id: str,
    ) -> ResolvedKernelPack:
        for pack in self._packs:
            if not pack.matches(fingerprint):
                continue
            try:
                item = pack.get(operator, semantic_version)
            except ConfigError:
                continue
            if item.evidence_id != evidence_id or item.artifact != artifact:
                continue
            return ResolvedKernelPack(pack=pack, artifact=item)

        suffix = f"; {len(self._rejected)} invalid pack(s) were ignored" if self._rejected else ""
        raise PackResolutionError(
            f"no installed kernel pack matches {operator} {semantic_version}, "
            f"{fingerprint.architecture}, ROCm {fingerprint.rocm_abi}{suffix}"
        )
