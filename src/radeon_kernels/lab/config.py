from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath, Path
from typing import Any, Mapping, Sequence

import yaml

from radeon_kernels.errors import ConfigError


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SSH_ALIAS_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ARCH_RE = re.compile(r"^gfx[0-9a-f]+$")
_IMAGE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-fA-F]{64}$")
_BUSY_POLICIES = {"wait", "skip", "fail"}
_RUNTIMES = {"docker", "podman"}


@dataclass(frozen=True)
class MountConfig:
    source: str
    target: str
    read_only: bool


@dataclass(frozen=True)
class ContainerConfig:
    runtime: str
    image: str
    workdir: str
    devices: tuple[str, ...]
    security_options: tuple[str, ...]
    mounts: tuple[MountConfig, ...]
    environment_from_host: tuple[str, ...]


@dataclass(frozen=True)
class TargetConfig:
    id: str
    ssh_host: str
    expected_architecture: str
    remote_root: str
    gpu_ids: tuple[int, ...]
    gpus_per_job: int
    max_parallel_jobs: int
    busy_policy: str
    busy_timeout_seconds: int
    container: ContainerConfig


@dataclass(frozen=True)
class LabConfig:
    schema_version: int
    targets: tuple[TargetConfig, ...]


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(path, "expected a mapping")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(path, "expected a list")
    return value


def _reject_unknown(
    mapping: Mapping[str, Any], allowed: set[str], path: str
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(path, f"unknown field(s): {', '.join(unknown)}")


def _required_string(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}.{key}", "expected a non-empty string")
    return value


def _positive_int(
    mapping: Mapping[str, Any], key: str, path: str, default: int
) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{path}.{key}", "expected a positive integer")
    return value


def _string_list(value: Any, path: str) -> tuple[str, ...]:
    items = _sequence(value, path)
    result: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{path}[{index}]", "expected a non-empty string")
        result.append(item)
    if len(result) != len(set(result)):
        raise ConfigError(path, "must not contain duplicates")
    return tuple(result)


def _parse_mount(value: Any, path: str) -> MountConfig:
    data = _mapping(value, path)
    _reject_unknown(data, {"source", "target", "read_only"}, path)
    source = _required_string(data, "source", path)
    target = _required_string(data, "target", path)
    if not PurePosixPath(target).is_absolute():
        raise ConfigError(f"{path}.target", "expected an absolute container path")
    read_only = data.get("read_only", True)
    if not isinstance(read_only, bool):
        raise ConfigError(f"{path}.read_only", "expected a boolean")
    return MountConfig(source=source, target=target, read_only=read_only)


def _parse_container(value: Any, path: str) -> ContainerConfig:
    data = _mapping(value, path)
    _reject_unknown(
        data,
        {
            "runtime",
            "image",
            "workdir",
            "devices",
            "security_options",
            "mounts",
            "environment_from_host",
        },
        path,
    )
    runtime = _required_string(data, "runtime", path)
    if runtime not in _RUNTIMES:
        raise ConfigError(f"{path}.runtime", "expected docker or podman")
    image = _required_string(data, "image", path)
    if not _IMAGE_RE.fullmatch(image):
        raise ConfigError(
            f"{path}.image",
            "expected an immutable image reference ending in @sha256:<64 hex chars>",
        )
    workdir = _required_string(data, "workdir", path)
    if not PurePosixPath(workdir).is_absolute():
        raise ConfigError(f"{path}.workdir", "expected an absolute container path")

    mount_values = _sequence(data.get("mounts", []), f"{path}.mounts")
    mounts = tuple(
        _parse_mount(item, f"{path}.mounts[{index}]")
        for index, item in enumerate(mount_values)
    )
    environment_from_host = _string_list(
        data.get("environment_from_host", []),
        f"{path}.environment_from_host",
    )
    for index, name in enumerate(environment_from_host):
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
            raise ConfigError(
                f"{path}.environment_from_host[{index}]",
                "expected an uppercase environment variable name",
            )

    return ContainerConfig(
        runtime=runtime,
        image=image,
        workdir=workdir,
        devices=_string_list(data.get("devices", []), f"{path}.devices"),
        security_options=_string_list(
            data.get("security_options", []),
            f"{path}.security_options",
        ),
        mounts=mounts,
        environment_from_host=environment_from_host,
    )


def _parse_target(value: Any, index: int) -> TargetConfig:
    path = f"targets[{index}]"
    data = _mapping(value, path)
    _reject_unknown(
        data,
        {
            "id",
            "ssh_host",
            "expected_architecture",
            "remote_root",
            "gpu_ids",
            "gpus_per_job",
            "max_parallel_jobs",
            "busy_policy",
            "busy_timeout_seconds",
            "container",
        },
        path,
    )

    target_id = _required_string(data, "id", path)
    if not _ID_RE.fullmatch(target_id):
        raise ConfigError(f"{path}.id", "contains unsupported characters")
    ssh_host = _required_string(data, "ssh_host", path)
    if not _SSH_ALIAS_RE.fullmatch(ssh_host):
        raise ConfigError(
            f"{path}.ssh_host",
            "expected an SSH config alias without spaces or shell characters",
        )
    architecture = data.get("expected_architecture", "auto")
    if not isinstance(architecture, str) or (
        architecture != "auto" and not _ARCH_RE.fullmatch(architecture)
    ):
        raise ConfigError(
            f"{path}.expected_architecture",
            "expected auto or a gfx architecture such as gfx1100",
        )

    remote_root = _required_string(data, "remote_root", path)
    remote_path = PurePosixPath(remote_root)
    if not remote_path.is_absolute() or remote_root == "/" or ".." in remote_path.parts:
        raise ConfigError(
            f"{path}.remote_root",
            "expected a non-root absolute path without '..'",
        )

    gpu_values = _sequence(data.get("gpu_ids"), f"{path}.gpu_ids")
    gpu_ids: list[int] = []
    for gpu_index, gpu_id in enumerate(gpu_values):
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
            raise ConfigError(
                f"{path}.gpu_ids[{gpu_index}]",
                "expected a non-negative integer",
            )
        gpu_ids.append(gpu_id)
    if not gpu_ids:
        raise ConfigError(f"{path}.gpu_ids", "must contain at least one GPU")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ConfigError(f"{path}.gpu_ids", "must not contain duplicates")

    gpus_per_job = _positive_int(data, "gpus_per_job", path, 1)
    max_parallel_jobs = _positive_int(data, "max_parallel_jobs", path, 1)
    if gpus_per_job > len(gpu_ids):
        raise ConfigError(
            f"{path}.gpus_per_job",
            "cannot exceed the number of allowed gpu_ids",
        )
    if gpus_per_job * max_parallel_jobs > len(gpu_ids):
        raise ConfigError(
            f"{path}.max_parallel_jobs",
            "would oversubscribe the configured gpu_ids",
        )

    busy_policy = data.get("busy_policy", "wait")
    if busy_policy not in _BUSY_POLICIES:
        raise ConfigError(
            f"{path}.busy_policy",
            "expected wait, skip, or fail",
        )

    return TargetConfig(
        id=target_id,
        ssh_host=ssh_host,
        expected_architecture=architecture,
        remote_root=remote_root.rstrip("/"),
        gpu_ids=tuple(gpu_ids),
        gpus_per_job=gpus_per_job,
        max_parallel_jobs=max_parallel_jobs,
        busy_policy=busy_policy,
        busy_timeout_seconds=_positive_int(
            data,
            "busy_timeout_seconds",
            path,
            1800,
        ),
        container=_parse_container(data.get("container"), f"{path}.container"),
    )


def load_targets(path: Path | str) -> LabConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError("targets", f"file does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError("targets", f"invalid YAML: {exc}") from exc

    data = _mapping(payload, "targets")
    _reject_unknown(data, {"schema_version", "targets"}, "targets")
    if data.get("schema_version") != 1:
        raise ConfigError("targets.schema_version", "expected 1")
    target_values = _sequence(data.get("targets"), "targets.targets")
    if not target_values:
        raise ConfigError("targets.targets", "must contain at least one target")

    targets = tuple(_parse_target(value, index) for index, value in enumerate(target_values))
    ids = [target.id for target in targets]
    aliases = [target.ssh_host for target in targets]
    if len(ids) != len(set(ids)):
        raise ConfigError("targets.targets", "target ids must be unique")
    if len(aliases) != len(set(aliases)):
        raise ConfigError(
            "targets.targets",
            "ssh_host aliases must be unique per physical target",
        )
    return LabConfig(schema_version=1, targets=targets)
