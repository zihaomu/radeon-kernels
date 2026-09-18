from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

from radeon_kernels.errors import ConfigError
from radeon_kernels.runtime.registry import DispatchManifest, DispatchRegistry
from radeon_kernels.runtime.signing import (
    DetachedSignature,
    TrustedKeyring,
    builtin_trusted_keyring,
    verify_detached_content,
)


def _manifest_documents() -> Iterable[tuple[str, Any]]:
    try:
        root = files("radeon_kernels.dispatch")
        for resource in sorted(root.iterdir(), key=lambda item: item.name):
            if resource.name.endswith(".json"):
                yield str(resource), json.loads(resource.read_text(encoding="utf-8"))
        return
    except ModuleNotFoundError:
        pass

    development_root = Path(__file__).resolve().parents[3] / "dispatch"
    for path in sorted(development_root.glob("*.json")):
        yield str(path), json.loads(path.read_text(encoding="utf-8"))


def builtin_dispatch_registry() -> DispatchRegistry:
    manifests: list[DispatchManifest] = []
    try:
        for path, payload in _manifest_documents():
            manifests.append(DispatchManifest.from_mapping(payload, path))
    except json.JSONDecodeError as error:
        raise ConfigError("built-in dispatch", f"invalid JSON: {error.msg}") from error
    if not manifests:
        raise ConfigError("built-in dispatch", "contains no manifests")
    return DispatchRegistry(manifests)


def _verified_resource_document(
    *,
    package: str,
    development_directory: str,
    filename: str,
    trusted_keys: TrustedKeyring | None = None,
) -> Mapping[str, Any]:
    keyring = trusted_keys or builtin_trusted_keyring()
    signature_name = f"{Path(filename).stem}.sig.json"
    try:
        root = files(package)
        content = root.joinpath(filename).read_bytes()
        signature = json.loads(root.joinpath(signature_name).read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        root_path = Path(__file__).resolve().parents[3] / development_directory
        content = (root_path / filename).read_bytes()
        signature = json.loads((root_path / signature_name).read_text(encoding="utf-8"))
    try:
        verify_detached_content(
            content,
            DetachedSignature.from_mapping(signature, f"{package}/{signature_name}"),
            keyring,
            signed_file=filename,
        )
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise ConfigError(f"{package}/{filename}", f"invalid JSON: {error.msg}") from error
    if not isinstance(value, Mapping):
        raise ConfigError(f"{package}/{filename}", "expected a mapping")
    return value


def builtin_pack_index(
    operator: str,
    semantic_version: str,
    *,
    trusted_keys: TrustedKeyring | None = None,
) -> Mapping[str, Any]:
    """Load a signed public pack index from the installed wheel or source tree."""

    filename = f"{operator}-{semantic_version}.json"
    value = _verified_resource_document(
        package="radeon_kernels.pack_index",
        development_directory="pack-index",
        filename=filename,
        trusted_keys=trusted_keys,
    )
    if value.get("operator") != operator or value.get("semantic_version") != semantic_version:
        raise ConfigError(filename, "pack index identity does not match request")
    if not isinstance(value.get("packs"), list):
        raise ConfigError(filename, "pack index contains no pack list")
    return value


def builtin_pack_indexes(
    *, trusted_keys: TrustedKeyring | None = None
) -> tuple[Mapping[str, Any], ...]:
    """Load every signed public pack index from the wheel or source tree."""

    filenames: list[str] = []
    try:
        root = files("radeon_kernels.pack_index")
        filenames = sorted(
            resource.name
            for resource in root.iterdir()
            if resource.name.endswith(".json") and not resource.name.endswith(".sig.json")
        )
    except ModuleNotFoundError:
        root_path = Path(__file__).resolve().parents[3] / "pack-index"
        filenames = sorted(
            path.name
            for path in root_path.glob("*.json")
            if not path.name.endswith(".sig.json")
        )
    return tuple(
        _verified_resource_document(
            package="radeon_kernels.pack_index",
            development_directory="pack-index",
            filename=filename,
            trusted_keys=trusted_keys,
        )
        for filename in filenames
    )
