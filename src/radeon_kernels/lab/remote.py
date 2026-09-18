from __future__ import annotations

import shlex
import subprocess
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from pathlib import Path
from typing import Callable, Sequence

from radeon_kernels.lab.config import TargetConfig


class RemoteStatus(StrEnum):
    READY = "READY"
    UNREACHABLE = "UNREACHABLE"
    INVALID = "INVALID"


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class RemoteProbe:
    target_id: str
    ssh_host: str
    status: RemoteStatus
    detected_architecture: str | None
    runtime: str | None
    image_present: bool
    remote_root_writable: bool
    user: str | None
    message: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


Runner = Callable[[Sequence[str], int], CommandResult]


def _subprocess_runner(command: Sequence[str], timeout: int) -> CommandResult:
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            returncode=124,
            stdout=exc.stdout or "",
            stderr=f"remote probe timed out after {timeout}s",
        )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _remote_script(target: TargetConfig) -> str:
    runtime = shlex.quote(target.container.runtime)
    image = shlex.quote(target.container.image)
    remote_root = shlex.quote(target.remote_root)
    parent = shlex.quote(str(PurePosixPath(target.remote_root).parent))
    return "\n".join(
        [
            "set -u",
            'printf "user=%s\\n" "$(id -un)"',
            'if command -v rocm_agent_enumerator >/dev/null 2>&1; then '
            'arch=$(rocm_agent_enumerator 2>/dev/null | sort -u | tr "\\n" "," | sed "s/,$//"); '
            'else arch=""; fi',
            'printf "architecture=%s\\n" "$arch"',
            f'if command -v {runtime} >/dev/null 2>&1; then echo "runtime={target.container.runtime}"; '
            'else echo "runtime="; fi',
            f'if {runtime} image inspect {image} >/dev/null 2>&1; then '
            'echo "image_present=yes"; else echo "image_present=no"; fi',
            f'if [ -d {remote_root} ]; then root={remote_root}; else root={parent}; fi',
            'if [ -d "$root" ] && [ -w "$root" ]; then '
            'echo "remote_root_writable=yes"; else echo "remote_root_writable=no"; fi',
        ]
    )


def _parse_key_values(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def probe_target(
    target: TargetConfig,
    *,
    timeout_seconds: int = 20,
    known_hosts_file: Path | None = None,
    accept_new_host_keys: bool = False,
    runner: Runner = _subprocess_runner,
) -> RemoteProbe:
    command: list[str] = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        f"ConnectTimeout={min(timeout_seconds, 10)}",
    ]
    if known_hosts_file is not None:
        command.extend(
            [
                "-o",
                f"UserKnownHostsFile={known_hosts_file}",
                "-o",
                "StrictHostKeyChecking="
                + ("accept-new" if accept_new_host_keys else "yes"),
            ]
        )
    command.extend([target.ssh_host, _remote_script(target)])
    result = runner(command, timeout_seconds)
    if result.returncode != 0:
        message = result.stderr.strip() or f"ssh exited with {result.returncode}"
        return RemoteProbe(
            target_id=target.id,
            ssh_host=target.ssh_host,
            status=RemoteStatus.UNREACHABLE,
            detected_architecture=None,
            runtime=None,
            image_present=False,
            remote_root_writable=False,
            user=None,
            message=message,
        )

    values = _parse_key_values(result.stdout)
    architecture = values.get("architecture") or None
    runtime = values.get("runtime") or None
    image_present = values.get("image_present") == "yes"
    root_writable = values.get("remote_root_writable") == "yes"
    problems: list[str] = []
    if architecture is None:
        problems.append("could not detect a Radeon architecture")
    elif target.expected_architecture != "auto" and architecture != target.expected_architecture:
        problems.append(
            f"expected {target.expected_architecture}, detected {architecture}"
        )
    if runtime != target.container.runtime:
        problems.append(f"container runtime {target.container.runtime} is unavailable")
    if not image_present:
        problems.append("configured image digest is not present")
    if not root_writable:
        problems.append("remote workspace or its parent is not writable")

    return RemoteProbe(
        target_id=target.id,
        ssh_host=target.ssh_host,
        status=RemoteStatus.INVALID if problems else RemoteStatus.READY,
        detected_architecture=architecture,
        runtime=runtime,
        image_present=image_present,
        remote_root_writable=root_writable,
        user=values.get("user") or None,
        message="; ".join(problems) if problems else "ready",
    )


def probe_targets(
    targets: Sequence[TargetConfig],
    *,
    timeout_seconds: int = 20,
    known_hosts_file: Path | None = None,
    accept_new_host_keys: bool = False,
    runner: Runner = _subprocess_runner,
) -> tuple[RemoteProbe, ...]:
    return tuple(
        probe_target(
            target,
            timeout_seconds=timeout_seconds,
            known_hosts_file=known_hosts_file,
            accept_new_host_keys=accept_new_host_keys,
            runner=runner,
        )
        for target in targets
    )
