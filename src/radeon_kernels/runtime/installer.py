from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from radeon_kernels.errors import ConfigError, RadeonKernelsError
from radeon_kernels.runtime.compatibility import EnvironmentConstraint
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
from radeon_kernels.runtime.pack import KernelPack, KernelPackArtifact
from radeon_kernels.runtime.signing import (
    DetachedSignature,
    TrustedKeyring,
    builtin_trusted_keyring,
    sha256_file,
    verify_detached_content,
    verify_detached_file,
)


class PackInstallError(RadeonKernelsError):
    """Raised when a published pack cannot be installed safely."""


def _required_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(path, "expected non-empty text")
    return value


def _required_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(path, "expected a positive integer")
    return value


def _safe_release_name(value: Any, path: str) -> str:
    result = _required_string(value, path)
    parsed = PurePosixPath(result)
    if parsed.name != result or not result.endswith(".tar.gz"):
        raise ConfigError(path, "expected a .tar.gz filename")
    return result


@dataclass(frozen=True)
class IndexedPack:
    pack_id: str
    pack_version: str
    environment: EnvironmentConstraint
    manifest_sha256: str
    manifest_signature: DetachedSignature
    release_file: str
    release_sha256: str
    release_size: int
    artifacts: tuple[KernelPackArtifact, ...]
    approval_id: str

    @classmethod
    def from_mapping(cls, value: Any, path: str) -> IndexedPack:
        if not isinstance(value, Mapping):
            raise ConfigError(path, "expected a mapping")
        required = {
            "pack_id",
            "pack_version",
            "environment",
            "manifest_sha256",
            "manifest_signature",
            "release_file",
            "release_sha256",
            "release_size",
            "artifacts",
            "approval_id",
        }
        unknown = set(value) - required
        missing = required - set(value)
        if unknown or missing:
            raise ConfigError(path, f"unknown={sorted(unknown)}, missing={sorted(missing)}")
        raw_artifacts = value["artifacts"]
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise ConfigError(f"{path}.artifacts", "expected a non-empty list")
        manifest_sha256 = _required_string(value["manifest_sha256"], f"{path}.manifest_sha256")
        release_sha256 = _required_string(value["release_sha256"], f"{path}.release_sha256")
        for digest, digest_path in (
            (manifest_sha256, f"{path}.manifest_sha256"),
            (release_sha256, f"{path}.release_sha256"),
        ):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ConfigError(digest_path, "expected a lowercase SHA-256")
        signature = DetachedSignature.from_mapping(
            value["manifest_signature"], f"{path}.manifest_signature"
        )
        if signature.signed_file != "manifest.json" or signature.signed_sha256 != manifest_sha256:
            raise ConfigError(
                f"{path}.manifest_signature",
                "must sign the indexed manifest SHA-256",
            )
        return cls(
            pack_id=_required_string(value["pack_id"], f"{path}.pack_id"),
            pack_version=_required_string(value["pack_version"], f"{path}.pack_version"),
            environment=EnvironmentConstraint.from_mapping(
                value["environment"], f"{path}.environment"
            ),
            manifest_sha256=manifest_sha256,
            manifest_signature=signature,
            release_file=_safe_release_name(value["release_file"], f"{path}.release_file"),
            release_sha256=release_sha256,
            release_size=_required_integer(value["release_size"], f"{path}.release_size"),
            artifacts=tuple(
                KernelPackArtifact.from_mapping(item, f"{path}.artifacts[{index}]")
                for index, item in enumerate(raw_artifacts)
            ),
            approval_id=_required_string(value["approval_id"], f"{path}.approval_id"),
        )


def index_records(index: Mapping[str, Any], *, path: str = "pack-index") -> tuple[IndexedPack, ...]:
    if index.get("schema_version") != "1.0":
        raise ConfigError(f"{path}.schema_version", "unsupported version")
    packs = index.get("packs")
    if not isinstance(packs, list):
        raise ConfigError(f"{path}.packs", "expected a list")
    return tuple(
        IndexedPack.from_mapping(item, f"{path}.packs[{position}]")
        for position, item in enumerate(packs)
    )


def compatible_records(
    index: Mapping[str, Any], fingerprint: EnvironmentFingerprint
) -> tuple[IndexedPack, ...]:
    return tuple(record for record in index_records(index) if record.environment.matches(fingerprint))


def _select_record(
    index: Mapping[str, Any],
    fingerprint: EnvironmentFingerprint,
    release_file: str | None,
) -> IndexedPack:
    records = compatible_records(index, fingerprint)
    if release_file is not None:
        records = tuple(record for record in records if record.release_file == release_file)
    if not records:
        suffix = f" and release {release_file!r}" if release_file else ""
        raise PackInstallError(
            f"no indexed pack matches {fingerprint.architecture}, ROCm {fingerprint.rocm_abi}{suffix}"
        )
    if len(records) != 1:
        names = ", ".join(record.release_file for record in records)
        raise PackInstallError(f"multiple compatible packs found; select one of: {names}")
    return records[0]


def _verify_archive(archive: Path, record: IndexedPack) -> None:
    try:
        size = archive.stat().st_size
    except OSError as error:
        raise PackInstallError(f"cannot read release archive {archive}: {error}") from error
    if size != record.release_size:
        raise PackInstallError(
            f"release size mismatch: expected {record.release_size}, got {size}"
        )
    actual = sha256_file(archive)
    if not hmac.compare_digest(actual, record.release_sha256):
        raise PackInstallError(
            f"release SHA-256 mismatch: expected {record.release_sha256}, got {actual}"
        )


def _normalized_member(member: tarfile.TarInfo) -> str:
    value = member.name.rstrip("/")
    parsed = PurePosixPath(value)
    if not value or parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != value:
        raise PackInstallError(f"release contains unsafe path {member.name!r}")
    return value


def _extract_archive(archive: Path, staging: Path, record: IndexedPack) -> Path:
    expected_files = {
        f"{record.pack_id}/manifest.json",
        f"{record.pack_id}/manifest.sig.json",
        *(f"{record.pack_id}/{item.artifact.artifact}" for item in record.artifacts),
    }
    seen: set[str] = set()
    regular_files: set[str] = set()
    try:
        handle = tarfile.open(archive, "r:gz")
    except (OSError, tarfile.TarError) as error:
        raise PackInstallError(f"cannot open release archive {archive}: {error}") from error
    with handle:
        members = handle.getmembers()
        if len(members) > 256:
            raise PackInstallError("release contains too many archive members")
        for member in members:
            normalized = _normalized_member(member)
            if normalized in seen:
                raise PackInstallError(f"release contains duplicate path {normalized!r}")
            seen.add(normalized)
            if PurePosixPath(normalized).parts[0] != record.pack_id:
                raise PackInstallError("release root does not match the indexed pack id")
            if member.isdir():
                continue
            if not member.isfile():
                raise PackInstallError(f"release contains unsupported member {normalized!r}")
            regular_files.add(normalized)
        if regular_files != expected_files:
            missing = sorted(expected_files - regular_files)
            unexpected = sorted(regular_files - expected_files)
            raise PackInstallError(
                f"release payload differs from index: missing={missing}, unexpected={unexpected}"
            )
        for member in members:
            normalized = _normalized_member(member)
            destination = staging.joinpath(*PurePosixPath(normalized).parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise PackInstallError(f"cannot read release member {normalized!r}")
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            os.chmod(destination, 0o755 if destination.suffix in {".so", ".hsaco"} else 0o644)
    return staging / record.pack_id


def _artifact_identity(item: KernelPackArtifact) -> tuple[Any, ...]:
    artifact = item.artifact
    return (
        item.operator,
        item.semantic_version,
        artifact.provider,
        artifact.artifact,
        artifact.entrypoint,
        artifact.sha256,
        artifact.format,
        item.evidence_id,
    )


def verify_installed_pack(
    pack_root: Path,
    *,
    trusted_keys: TrustedKeyring | None = None,
    record: IndexedPack | None = None,
) -> KernelPack:
    keyring = trusted_keys or builtin_trusted_keyring()
    verify_detached_file(
        pack_root / "manifest.json",
        pack_root / "manifest.sig.json",
        keyring,
        signed_file="manifest.json",
    )
    pack = KernelPack.from_directory(pack_root, verify_artifacts=True)
    if record is None:
        return pack
    manifest_sha256 = sha256_file(pack.root / "manifest.json")
    if not hmac.compare_digest(manifest_sha256, record.manifest_sha256):
        raise PackInstallError("installed manifest does not match the signed pack index")
    signature_value = json.loads((pack.root / "manifest.sig.json").read_text(encoding="utf-8"))
    installed_signature = DetachedSignature.from_mapping(
        signature_value, str(pack.root / "manifest.sig.json")
    )
    if installed_signature != record.manifest_signature:
        raise PackInstallError("manifest signature does not match the signed pack index")
    if pack.pack_id != record.pack_id or pack.pack_version != record.pack_version:
        raise PackInstallError("installed pack identity does not match the signed pack index")
    if pack.environment != record.environment:
        raise PackInstallError("installed pack environment does not match the signed pack index")
    if tuple(map(_artifact_identity, pack.artifacts)) != tuple(
        map(_artifact_identity, record.artifacts)
    ):
        raise PackInstallError("installed artifacts do not match the signed pack index")
    return pack


def install_release(
    archive: Path,
    index: Mapping[str, Any],
    fingerprint: EnvironmentFingerprint,
    destination_root: Path,
    *,
    trusted_keys: TrustedKeyring | None = None,
    release_file: str | None = None,
) -> KernelPack:
    record = _select_record(index, fingerprint, release_file)
    _verify_archive(archive, record)
    keyring = trusted_keys or builtin_trusted_keyring()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / record.pack_id
    if destination.exists():
        raise PackInstallError(f"refusing to overwrite installed pack {destination}")
    staging = Path(tempfile.mkdtemp(prefix=".rk-install-", dir=destination_root))
    try:
        staged_pack = _extract_archive(archive, staging, record)
        manifest = (staged_pack / "manifest.json").read_bytes()
        verify_detached_content(
            manifest,
            record.manifest_signature,
            keyring,
            signed_file="manifest.json",
        )
        pack = verify_installed_pack(staged_pack, trusted_keys=keyring, record=record)
        os.replace(staged_pack, destination)
        return KernelPack.from_directory(destination, verify_artifacts=True)
    except (OSError, tarfile.TarError, json.JSONDecodeError) as error:
        raise PackInstallError(f"cannot install release archive: {error}") from error
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def fetch_release(record: IndexedPack, base_url: str, output: Path) -> Path:
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", urllib.parse.quote(record.release_file))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.download")
    try:
        with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as handle:
            remaining = record.release_size
            while remaining:
                chunk = response.read(min(1024 * 1024, remaining + 1))
                if not chunk:
                    break
                if len(chunk) > remaining:
                    raise PackInstallError("download exceeds indexed release size")
                handle.write(chunk)
                remaining -= len(chunk)
        _verify_archive(temporary, record)
        os.replace(temporary, output)
        return output
    except OSError as error:
        raise PackInstallError(f"cannot download {url}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
