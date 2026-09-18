from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from radeon_kernels.errors import ConfigError
from radeon_kernels.runtime.signing import sha256_file, signature_document


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise


def generate_private_key_document(key_id: str) -> tuple[dict[str, str], dict[str, str]]:
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    created_at = datetime.now(UTC).isoformat()
    private_document = {
        "schema_version": "1.0",
        "algorithm": "ed25519",
        "key_id": key_id,
        "private_key": base64.b64encode(private_bytes).decode("ascii"),
        "public_key": base64.b64encode(public_bytes).decode("ascii"),
        "created_at": created_at,
    }
    public_record = {
        "key_id": key_id,
        "algorithm": "ed25519",
        "public_key": private_document["public_key"],
        "status": "active",
        "created_at": created_at,
    }
    return private_document, public_record


def generate_private_key(path: Path, key_id: str) -> dict[str, str]:
    private_document, public_record = generate_private_key_document(key_id)
    _write_private_json(path, private_document)
    return public_record


def load_private_key(path: Path) -> tuple[str, Ed25519PrivateKey]:
    try:
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise ConfigError(str(path), "private key permissions must be 0600 or stricter")
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(str(path), f"cannot read private key: {error}") from error
    if not isinstance(value, Mapping):
        raise ConfigError(str(path), "expected a mapping")
    fields = {
        "schema_version",
        "algorithm",
        "key_id",
        "private_key",
        "public_key",
        "created_at",
    }
    if set(value) != fields or value["schema_version"] != "1.0" or value["algorithm"] != "ed25519":
        raise ConfigError(str(path), "invalid private key document")
    try:
        private_bytes = base64.b64decode(value["private_key"], validate=True)
        public_bytes = base64.b64decode(value["public_key"], validate=True)
        private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
    except (TypeError, ValueError) as error:
        raise ConfigError(str(path), "invalid Ed25519 key material") from error
    actual_public = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    if actual_public != public_bytes:
        raise ConfigError(str(path), "private and public key material do not match")
    return str(value["key_id"]), private_key


def sign_file(
    *,
    private_key_path: Path,
    file_path: Path,
    signed_file: str,
    output: Path,
    refuse_overwrite: bool = True,
) -> dict[str, str]:
    if refuse_overwrite and output.exists():
        raise FileExistsError(f"refusing to overwrite signature: {output}")
    key_id, private_key = load_private_key(private_key_path)
    document = signature_document(
        private_key=private_key,
        key_id=key_id,
        signed_file=signed_file,
        signed_sha256=sha256_file(file_path),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document
