from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from radeon_kernels.errors import ConfigError


WORKSPACE_FILENAME = ".radeon-workspace.yaml"


@dataclass(frozen=True)
class Workspace:
    root: Path
    config_file: Path
    project_dir: Path
    private_dir: Path
    targets_file: Path
    runs_dir: Path
    state_dir: Path
    cache_dir: Path
    exports_dir: Path


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(path, "expected a mapping")
    return value


def _string(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}.{key}", "expected a non-empty string")
    return value


def _reject_unknown(
    mapping: Mapping[str, Any], allowed: set[str], path: str
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(path, f"unknown field(s): {', '.join(unknown)}")


def _inside(root: Path, raw_path: str, path: str) -> Path:
    candidate = (root / raw_path).resolve()
    if not candidate.is_relative_to(root):
        raise ConfigError(path, "must resolve inside the workspace root")
    return candidate


def find_workspace_file(start: Path | None = None) -> Path:
    cursor = (start or Path.cwd()).resolve()
    if cursor.is_file():
        cursor = cursor.parent
    for directory in (cursor, *cursor.parents):
        candidate = directory / WORKSPACE_FILENAME
        if candidate.is_file():
            return candidate
    raise ConfigError(
        "workspace",
        f"could not find {WORKSPACE_FILENAME} from {cursor}",
    )


def load_workspace(explicit: Path | str | None = None) -> Workspace:
    if explicit is None:
        config_file = find_workspace_file()
    else:
        supplied = Path(explicit).expanduser().resolve()
        config_file = supplied / WORKSPACE_FILENAME if supplied.is_dir() else supplied
        if not config_file.is_file():
            raise ConfigError("workspace", f"file does not exist: {config_file}")

    try:
        payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError("workspace", f"invalid YAML: {exc}") from exc

    root = config_file.parent.resolve()
    data = _mapping(payload, "workspace")
    _reject_unknown(
        data,
        {"schema_version", "project_dir", "private_dir", "inventory", "storage"},
        "workspace",
    )
    if data.get("schema_version") != 1:
        raise ConfigError("workspace.schema_version", "expected 1")

    inventory = _mapping(data.get("inventory"), "workspace.inventory")
    storage = _mapping(data.get("storage"), "workspace.storage")
    _reject_unknown(inventory, {"targets"}, "workspace.inventory")
    _reject_unknown(
        storage,
        {"runs", "state", "cache", "exports"},
        "workspace.storage",
    )

    return Workspace(
        root=root,
        config_file=config_file,
        project_dir=_inside(
            root,
            _string(data, "project_dir", "workspace"),
            "workspace.project_dir",
        ),
        private_dir=_inside(
            root,
            _string(data, "private_dir", "workspace"),
            "workspace.private_dir",
        ),
        targets_file=_inside(
            root,
            _string(inventory, "targets", "workspace.inventory"),
            "workspace.inventory.targets",
        ),
        runs_dir=_inside(
            root,
            _string(storage, "runs", "workspace.storage"),
            "workspace.storage.runs",
        ),
        state_dir=_inside(
            root,
            _string(storage, "state", "workspace.storage"),
            "workspace.storage.state",
        ),
        cache_dir=_inside(
            root,
            _string(storage, "cache", "workspace.storage"),
            "workspace.storage.cache",
        ),
        exports_dir=_inside(
            root,
            _string(storage, "exports", "workspace.storage"),
            "workspace.storage.exports",
        ),
    )

