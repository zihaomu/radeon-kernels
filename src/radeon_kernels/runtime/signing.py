from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from radeon_kernels.errors import ConfigError, RadeonKernelsError

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE_DOMAIN = b"radeon-kernels-ed25519-v1\x00"


class SignatureVerificationError(RadeonKernelsError):
    """Raised when a release signature cannot be trusted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_base64(value: Any, path: str, expected_length: int) -> bytes:
    if not isinstance(value, str):
        raise ConfigError(path, "expected base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ConfigError(path, "invalid base64 text") from error
    if len(decoded) != expected_length:
        raise ConfigError(path, f"expected {expected_length} decoded bytes")
    return decoded


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ConfigError(path, "expected a lowercase identifier")
    return value


def _relative_file(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(path, "expected a relative file path")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != value:
        raise ConfigError(path, "expected a normalized relative file path")
    return value


def signature_payload(signed_file: str, signed_sha256: str) -> bytes:
    normalized_file = _relative_file(signed_file, "signature.signed_file")
    if _SHA256_RE.fullmatch(signed_sha256) is None:
        raise ConfigError("signature.signed_sha256", "expected a lowercase SHA-256")
    return _SIGNATURE_DOMAIN + normalized_file.encode("utf-8") + b"\x00" + signed_sha256.encode("ascii")


@dataclass(frozen=True)
class TrustedKey:
    key_id: str
    public_key: bytes
    status: str
    created_at: str
    reason: str | None = None


class TrustedKeyring:
    def __init__(self, keys: Sequence[TrustedKey]) -> None:
        if not keys:
            raise ConfigError("trusted-keys.keys", "expected at least one key")
        identifiers = [key.key_id for key in keys]
        if len(set(identifiers)) != len(identifiers):
            raise ConfigError("trusted-keys.keys", "key ids must be unique")
        self._keys = {key.key_id: key for key in keys}

    @classmethod
    def from_mapping(cls, value: Any, path: str = "trusted-keys") -> TrustedKeyring:
        if not isinstance(value, Mapping):
            raise ConfigError(path, "expected a mapping")
        fields = {"schema_version", "keys"}
        if set(value) != fields:
            raise ConfigError(path, f"expected exactly {sorted(fields)}")
        if value["schema_version"] != "1.0":
            raise ConfigError(f"{path}.schema_version", "unsupported version")
        raw_keys = value["keys"]
        if isinstance(raw_keys, (str, bytes)) or not isinstance(raw_keys, list):
            raise ConfigError(f"{path}.keys", "expected a list")
        keys: list[TrustedKey] = []
        for index, item in enumerate(raw_keys):
            item_path = f"{path}.keys[{index}]"
            if not isinstance(item, Mapping):
                raise ConfigError(item_path, "expected a mapping")
            allowed = {"key_id", "algorithm", "public_key", "status", "created_at", "reason"}
            required = allowed - {"reason"}
            unknown = set(item) - allowed
            missing = required - set(item)
            if unknown or missing:
                raise ConfigError(item_path, f"unknown={sorted(unknown)}, missing={sorted(missing)}")
            if item["algorithm"] != "ed25519":
                raise ConfigError(f"{item_path}.algorithm", "expected ed25519")
            if item["status"] not in {"active", "revoked"}:
                raise ConfigError(f"{item_path}.status", "expected active or revoked")
            created_at = item["created_at"]
            if not isinstance(created_at, str) or not created_at:
                raise ConfigError(f"{item_path}.created_at", "expected a timestamp")
            reason = item.get("reason")
            if reason is not None and (not isinstance(reason, str) or not reason):
                raise ConfigError(f"{item_path}.reason", "expected non-empty text")
            keys.append(
                TrustedKey(
                    key_id=_identifier(item["key_id"], f"{item_path}.key_id"),
                    public_key=_decode_base64(item["public_key"], f"{item_path}.public_key", 32),
                    status=item["status"],
                    created_at=created_at,
                    reason=reason,
                )
            )
        return cls(keys)

    @classmethod
    def from_file(cls, path: Path) -> TrustedKeyring:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ConfigError(str(path), f"cannot read trusted keys: {error}") from error
        return cls.from_mapping(value, str(path))

    def active_public_key(self, key_id: str) -> Ed25519PublicKey:
        try:
            key = self._keys[key_id]
        except KeyError as error:
            raise SignatureVerificationError(f"signature uses unknown key {key_id!r}") from error
        if key.status != "active":
            raise SignatureVerificationError(f"signature key {key_id!r} is not active")
        return Ed25519PublicKey.from_public_bytes(key.public_key)


@dataclass(frozen=True)
class DetachedSignature:
    key_id: str
    signed_file: str
    signed_sha256: str
    signature: bytes

    @classmethod
    def from_mapping(cls, value: Any, path: str = "signature") -> DetachedSignature:
        if not isinstance(value, Mapping):
            raise ConfigError(path, "expected a mapping")
        fields = {
            "schema_version",
            "algorithm",
            "key_id",
            "signed_file",
            "signed_sha256",
            "signature",
        }
        if set(value) != fields:
            raise ConfigError(path, f"expected exactly {sorted(fields)}")
        if value["schema_version"] != "1.0" or value["algorithm"] != "ed25519":
            raise ConfigError(path, "unsupported signature version or algorithm")
        digest = value["signed_sha256"]
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ConfigError(f"{path}.signed_sha256", "expected a lowercase SHA-256")
        return cls(
            key_id=_identifier(value["key_id"], f"{path}.key_id"),
            signed_file=_relative_file(value["signed_file"], f"{path}.signed_file"),
            signed_sha256=digest,
            signature=_decode_base64(value["signature"], f"{path}.signature", 64),
        )

    @classmethod
    def from_file(cls, path: Path) -> DetachedSignature:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SignatureVerificationError(f"cannot read detached signature {path}: {error}") from error
        try:
            return cls.from_mapping(value, str(path))
        except ConfigError as error:
            raise SignatureVerificationError(str(error)) from error


def signature_document(
    *,
    private_key: Ed25519PrivateKey,
    key_id: str,
    signed_file: str,
    signed_sha256: str,
) -> dict[str, str]:
    normalized_key_id = _identifier(key_id, "signature.key_id")
    signature = private_key.sign(signature_payload(signed_file, signed_sha256))
    return {
        "schema_version": "1.0",
        "algorithm": "ed25519",
        "key_id": normalized_key_id,
        "signed_file": signed_file,
        "signed_sha256": signed_sha256,
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def verify_detached_file(
    file_path: Path,
    signature_path: Path,
    keyring: TrustedKeyring,
    *,
    signed_file: str | None = None,
) -> DetachedSignature:
    try:
        content = file_path.read_bytes()
    except OSError as error:
        raise SignatureVerificationError(f"cannot read signed file {file_path}: {error}") from error
    envelope = DetachedSignature.from_file(signature_path)
    return verify_detached_content(
        content,
        envelope,
        keyring,
        signed_file=signed_file or file_path.name,
    )


def verify_detached_content(
    content: bytes,
    signature: DetachedSignature | Mapping[str, Any],
    keyring: TrustedKeyring,
    *,
    signed_file: str,
) -> DetachedSignature:
    envelope = (
        signature
        if isinstance(signature, DetachedSignature)
        else DetachedSignature.from_mapping(signature)
    )
    expected_file = _relative_file(signed_file, "signed_file")
    if envelope.signed_file != expected_file:
        raise SignatureVerificationError(
            f"signature names {envelope.signed_file!r}, expected {expected_file!r}"
        )
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(actual_sha256, envelope.signed_sha256):
        raise SignatureVerificationError(
            f"signed file SHA-256 mismatch: expected {envelope.signed_sha256}, got {actual_sha256}"
        )
    public_key = keyring.active_public_key(envelope.key_id)
    try:
        public_key.verify(
            envelope.signature,
            signature_payload(envelope.signed_file, envelope.signed_sha256),
        )
    except InvalidSignature as error:
        raise SignatureVerificationError("invalid Ed25519 signature") from error
    return envelope


def builtin_trusted_keyring() -> TrustedKeyring:
    try:
        resource = files("radeon_kernels.keys").joinpath("trusted-keys.json")
        return TrustedKeyring.from_mapping(
            json.loads(resource.read_text(encoding="utf-8")),
            str(resource),
        )
    except ModuleNotFoundError:
        development_path = Path(__file__).resolve().parents[3] / "keys" / "trusted-keys.json"
        return TrustedKeyring.from_file(development_path)
