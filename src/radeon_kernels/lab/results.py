from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from typing import Any, Mapping, Sequence

from radeon_kernels.errors import ConfigError


class TaskStatus(StrEnum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RunStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


def aggregate_run_status(
    statuses: Sequence[TaskStatus], *, needs_review: bool = False
) -> RunStatus:
    if needs_review:
        return RunStatus.NEEDS_REVIEW
    if not statuses:
        return RunStatus.FAILED
    if all(status is TaskStatus.SUCCEEDED for status in statuses):
        return RunStatus.SUCCEEDED

    succeeded = sum(status is TaskStatus.SUCCEEDED for status in statuses)
    skipped = sum(status is TaskStatus.SKIPPED for status in statuses)
    if succeeded == 0:
        if skipped == len(statuses):
            return RunStatus.PARTIAL
        return RunStatus.FAILED
    return RunStatus.PARTIAL


_GPU_FIELDS = {
    "architecture",
    "model",
    "compute_units",
    "wavefront_size",
    "memory_capacity_bytes",
    "features",
}
_TOOLCHAIN_FIELDS = {"driver", "runtime", "triton"}
_WORKLOAD_FIELDS = {
    "dtype",
    "layout",
    "m",
    "n",
    "k",
    "seed",
}
_STATISTIC_FIELDS = {
    "median_ms",
    "cv",
    "confidence_lower_pct",
    "confidence_upper_pct",
    "sample_count",
}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "source_commit",
    "candidate_content_hash",
    "image_digest",
    "operator",
    "workload",
    "gpu",
    "toolchain",
    "candidate_id",
    "configuration",
    "samples_ms",
    "statistics",
}


def _allow_mapping(
    value: Any, allowed: set[str], path: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(path, "expected a mapping")
    return {key: deepcopy(value[key]) for key in allowed if key in value}


def _public_configuration(value: Any, path: str = "evidence.configuration") -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > 128 or any(token in value for token in ("/", "\\", "@", ":")):
            raise ConfigError(path, "contains a path-like or credential-like value")
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _public_configuration(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.replace("_", "").isalnum():
                raise ConfigError(path, "contains an unsupported key")
            if any(term in key.lower() for term in ("secret", "token", "password", "path")):
                raise ConfigError(f"{path}.{key}", "secret-like fields are not public")
            result[key] = _public_configuration(item, f"{path}.{key}")
        return result
    raise ConfigError(path, f"unsupported public value type: {type(value).__name__}")


def sanitize_evidence(private_record: Mapping[str, Any]) -> dict[str, Any]:
    """Build public evidence from an explicit field allowlist."""

    public = {
        key: deepcopy(private_record[key])
        for key in _TOP_LEVEL_FIELDS
        if key in private_record
    }
    public["gpu"] = _allow_mapping(private_record.get("gpu"), _GPU_FIELDS, "evidence.gpu")
    public["toolchain"] = _allow_mapping(
        private_record.get("toolchain"),
        _TOOLCHAIN_FIELDS,
        "evidence.toolchain",
    )
    public["workload"] = _allow_mapping(
        private_record.get("workload"),
        _WORKLOAD_FIELDS,
        "evidence.workload",
    )
    public["statistics"] = _allow_mapping(
        private_record.get("statistics"),
        _STATISTIC_FIELDS,
        "evidence.statistics",
    )
    public["configuration"] = _public_configuration(
        private_record.get("configuration", {})
    )
    samples = private_record.get("samples_ms", [])
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise ConfigError("evidence.samples_ms", "expected a numeric list")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in samples):
        raise ConfigError("evidence.samples_ms", "expected a numeric list")
    public["samples_ms"] = list(samples)
    return public

