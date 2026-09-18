from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from radeon_kernels.errors import ConfigError
from radeon_kernels.lab.config import LabConfig, TargetConfig


@dataclass(frozen=True)
class PlannedTarget:
    target_id: str
    ssh_host: str
    expected_architecture: str
    gpu_ids: tuple[int, ...]
    max_parallel_jobs: int
    busy_policy: str


@dataclass(frozen=True)
class ExecutionPlan:
    schema_version: int
    targets: tuple[PlannedTarget, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _planned_target(target: TargetConfig) -> PlannedTarget:
    return PlannedTarget(
        target_id=target.id,
        ssh_host=target.ssh_host,
        expected_architecture=target.expected_architecture,
        gpu_ids=target.gpu_ids,
        max_parallel_jobs=target.max_parallel_jobs,
        busy_policy=target.busy_policy,
    )


def build_plan(
    config: LabConfig, selected_targets: Iterable[str] | None = None
) -> ExecutionPlan:
    requested = tuple(selected_targets or ())
    if len(requested) != len(set(requested)):
        raise ConfigError("selection", "target selectors must not repeat")

    known = {target.id: target for target in config.targets}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ConfigError(
            "selection",
            f"unknown target(s): {', '.join(unknown)}",
        )

    selected = set(requested)
    targets = tuple(
        _planned_target(target)
        for target in config.targets
        if not requested or target.id in selected
    )
    return ExecutionPlan(schema_version=1, targets=targets)

