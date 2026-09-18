#!/usr/bin/env python3
"""Create a deterministic, privacy-reviewed promotion proposal from private evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from radeon_kernels.errors import ConfigError
from radeon_kernels.runtime.pack import KernelPack
from radeon_kernels.runtime.registry import DispatchManifest
from radeon_kernels.runtime.signing import sha256_file

_PUBLIC_CASE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(str(path), f"cannot read JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise ConfigError(str(path), "expected a mapping")
    return value


def _exact_dimension(value: Any, path: str) -> int:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or isinstance(value[0], bool)
        or not isinstance(value[0], int)
        or value[0] != value[1]
    ):
        raise ConfigError(path, "promotion currently requires an exact shape")
    return value[0]


def _latency_reduction(candidate: Sequence[float], baseline: Sequence[float]) -> float:
    return (1.0 - statistics.median(candidate) / statistics.median(baseline)) * 100.0


def _bootstrap_interval(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int = 20260917,
    resamples: int = 10_000,
) -> tuple[float, float]:
    if len(candidate) != len(baseline) or len(candidate) < 2:
        raise ConfigError("external.samples_ms", "candidate and baseline samples must be paired")
    generator = random.Random(seed)
    count = len(candidate)
    estimates: list[float] = []
    for _ in range(resamples):
        indices = [generator.randrange(count) for _ in range(count)]
        candidate_sample = [candidate[index] for index in indices]
        baseline_sample = [baseline[index] for index in indices]
        estimates.append(_latency_reduction(candidate_sample, baseline_sample))
    estimates.sort()
    lower = estimates[int(0.025 * (resamples - 1))]
    upper = estimates[int(0.975 * (resamples - 1))]
    return lower, upper


def _case_public(case: Mapping[str, Any]) -> dict[str, Any]:
    case_id = case["case_id"]
    if not isinstance(case_id, str) or _PUBLIC_CASE_ID_RE.fullmatch(case_id) is None:
        raise ConfigError("oracle.cases.case_id", "expected a public-safe case identifier")
    return {
        "case_id": case_id,
        "runtime_ok": bool(case["runtime_ok"]),
        "finite": bool(case["candidate_finite"]),
        "guards_intact": bool(case["guards_intact"]),
        "cosine_similarity": float(case["cosine_similarity"]),
        "max_absolute_error": float(case["max_absolute_error"]),
    }


def _artifact_mapping(item: Any) -> dict[str, Any]:
    result = item.artifact.to_dict()
    result.update(
        {
            "operator": item.operator,
            "semantic_version": item.semantic_version,
            "evidence_id": item.evidence_id,
        }
    )
    return result


def _environment_mapping(environment: Any) -> dict[str, Any]:
    return {
        "architecture": environment.architecture,
        "wavefront_size": environment.wavefront_size,
        "rocm_abi": environment.rocm_abi,
        "python_abi": environment.python_abi,
        "pytorch_version": environment.pytorch_version,
    }


def create_proposal(
    *,
    pack_path: Path,
    dispatch_path: Path,
    entry_id: str,
    oracle_path: Path,
    timing_path: Path,
    external_path: Path,
    pack_verification_path: Path,
    runtime_verification_path: Path,
    minimum_improvement_pct: float = 3.0,
    maximum_cv: float = 0.03,
) -> dict[str, Any]:
    pack = KernelPack.from_directory(pack_path, verify_artifacts=True)
    dispatch_payload = _json(dispatch_path)
    manifest = DispatchManifest.from_mapping(dispatch_payload, str(dispatch_path))
    entries = [entry for entry in manifest.entries if entry.id == entry_id]
    if len(entries) != 1:
        raise ConfigError(str(dispatch_path), f"expected exactly one entry {entry_id!r}")
    entry = entries[0]
    raw_entry = next(item for item in dispatch_payload["entries"] if item["id"] == entry_id)
    item = pack.get(manifest.operator, manifest.semantic_version)

    oracle = _json(oracle_path)
    timing = _json(timing_path)
    external = _json(external_path)
    pack_verification = _json(pack_verification_path)
    runtime_verification = _json(runtime_verification_path)
    sources = {
        "dispatch": sha256_file(dispatch_path),
        "external": sha256_file(external_path),
        "manifest": sha256_file(pack.root / "manifest.json"),
        "oracle": sha256_file(oracle_path),
        "pack_verification": sha256_file(pack_verification_path),
        "runtime_verification": sha256_file(runtime_verification_path),
        "timing": sha256_file(timing_path),
    }

    workload = {
        "dtype": entry.workload.values["dtype"].value,
        "layout": entry.workload.values["layout"].value,
        "m": _exact_dimension(raw_entry["workload"]["m"], "dispatch.workload.m"),
        "n": _exact_dimension(raw_entry["workload"]["n"], "dispatch.workload.n"),
        "k": _exact_dimension(raw_entry["workload"]["k"], "dispatch.workload.k"),
        "variant": entry.launch.get("variant"),
        "contiguous": entry.workload.values["contiguous"].value,
        "requires_grad": entry.workload.values["requires_grad"].value,
    }
    shape = {key: workload[key] for key in ("m", "n", "k")}
    oracle_cases = oracle.get("cases")
    if not isinstance(oracle_cases, list) or not oracle_cases:
        raise ConfigError(str(oracle_path), "contains no correctness cases")
    public_cases = [_case_public(case) for case in oracle_cases]

    external_samples = external.get("samples_ms")
    if not isinstance(external_samples, Mapping):
        raise ConfigError(str(external_path), "contains no timing samples")
    candidate_samples = [float(value) for value in external_samples["native"]]
    baseline_samples = [float(value) for value in external_samples["torch_mm"]]
    confidence_interval = _bootstrap_interval(candidate_samples, baseline_samples)
    summaries = external["summaries"]
    candidate_summary = summaries["native"]
    baseline_summary = summaries["torch_mm"]
    latency_reduction = _latency_reduction(candidate_samples, baseline_samples)

    expected_artifact = item.artifact.to_dict()
    gates = {
        "artifact_matches_dispatch": expected_artifact == entry.winner.to_dict(),
        "artifact_hash_consistent": all(
            value == item.artifact.sha256
            for value in (
                oracle.get("artifact_sha256"),
                external.get("artifact_sha256"),
                pack_verification.get("artifact", {}).get("sha256"),
            )
        ),
        "environment_consistent": (
            oracle.get("architecture") == pack.environment.architecture
            and external.get("architecture") == pack.environment.architecture
            and pack_verification.get("fingerprint") == {
                **pack.environment.to_dict(),
                "features": [],
            }
            and runtime_verification.get("winner", {}).get("dispatch", {}).get("architecture")
            == pack.environment.architecture
        ),
        "workload_consistent": (
            oracle.get("shape") == shape
            and timing.get("shape") == shape
            and external.get("shape") == shape
            and oracle.get("dtype") == workload["dtype"]
            and timing.get("dtype") == workload["dtype"]
            and external.get("dtype") == workload["dtype"]
            and oracle.get("variant") == workload["variant"]
            and timing.get("variant") == workload["variant"]
            and external.get("variant") == workload["variant"]
        ),
        "correctness_passed": all(
            case["runtime_ok"]
            and case["finite"]
            and case["guards_intact"]
            and case["cosine_similarity"] >= 0.9999
            and case["max_absolute_error"] <= 0.001
            for case in public_cases
        ),
        "internal_timing_valid": bool(timing.get("timing_valid")),
        "pack_verification_passed": (
            pack_verification.get("passed") is True
            and pack_verification.get("compiled_during_verification") is False
            and pack_verification.get("pack_id") == pack.pack_id
        ),
        "runtime_dispatch_passed": (
            runtime_verification.get("passed") is True
            and runtime_verification.get("compiled_during_verification") is False
            and runtime_verification.get("searched_during_verification") is False
            and runtime_verification.get("winner", {}).get("selected_provider")
            == "native/gemm_out"
            and runtime_verification.get("winner", {}).get("dispatch", {}).get("artifact")
            == item.artifact.artifact
            and runtime_verification.get("winner", {}).get("dispatch", {}).get("evidence_id")
            == item.evidence_id
            and runtime_verification.get("fallback", {}).get("selected_provider") == "torch/mm"
        ),
        "external_timing_stable": (
            external.get("stable") is True
            and float(candidate_summary["cv"]) <= maximum_cv
            and float(baseline_summary["cv"]) <= maximum_cv
        ),
        "improvement_threshold_passed": latency_reduction >= minimum_improvement_pct,
        "confidence_threshold_passed": confidence_interval[0] >= minimum_improvement_pct,
    }
    failed = [name for name, passed in gates.items() if not passed]
    recommended_decision = "promote" if not failed else "needs_review"
    reasons = (
        ["all deterministic promotion gates passed"]
        if not failed
        else [f"failed gate: {name}" for name in failed]
    )

    evidence_id = item.evidence_id
    public_evidence = {
        "schema_version": "1.0",
        "evidence_id": evidence_id,
        "operator": item.operator,
        "semantic_version": item.semantic_version,
        "candidate_content_hash": item.artifact.sha256,
        "captured_at": external["captured_at"],
        "environment": _environment_mapping(pack.environment),
        "workload": workload,
        "correctness": {
            "passed": gates["correctness_passed"],
            "oracle": "torch.mm",
            "cases": public_cases,
        },
        "timing": {
            "baseline": "torch.mm",
            "candidate_samples_ms": candidate_samples,
            "baseline_samples_ms": baseline_samples,
            "candidate_median_ms": float(candidate_summary["median_ms"]),
            "baseline_median_ms": float(baseline_summary["median_ms"]),
            "candidate_cv": float(candidate_summary["cv"]),
            "baseline_cv": float(baseline_summary["cv"]),
            "latency_reduction_pct": latency_reduction,
            "confidence_interval_pct": list(confidence_interval),
            "stable": bool(external.get("stable")),
        },
        "source_evidence_hashes": sources,
        "decision": recommended_decision,
        "guarantee_boundary": (
            f"Exact {pack.environment.architecture}, ROCm {pack.environment.rocm_abi}, "
            f"{pack.environment.python_abi}, PyTorch {pack.environment.pytorch_version}, "
            f"{workload['dtype']} NN {workload['m']}x{workload['n']}x{workload['k']}, "
            f"contiguous inference inputs only."
        ),
    }
    compatibility = {
        "id": f"{item.operator}-{pack.environment.architecture}-{evidence_id}",
        **_environment_mapping(pack.environment),
        "provider": item.artifact.provider,
        "status": "supported" if recommended_decision == "promote" else "experimental",
        "evidence_id": evidence_id,
        "reason": "; ".join(reasons),
    }
    identity = hashlib.sha256(
        json.dumps(
            {"evidence_id": evidence_id, "sources": sources},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema_version": "1.0",
        "proposal_id": f"proposal-{evidence_id}-{identity}",
        "created_at": external["captured_at"],
        "operator": item.operator,
        "semantic_version": item.semantic_version,
        "entry_id": entry.id,
        "pack": {
            "pack_id": pack.pack_id,
            "pack_version": pack.pack_version,
            "manifest_sha256": sources["manifest"],
            "environment": _environment_mapping(pack.environment),
            "artifacts": [_artifact_mapping(artifact) for artifact in pack.artifacts],
        },
        "source_evidence_hashes": sources,
        "gates": gates,
        "minimum_improvement_pct": minimum_improvement_pct,
        "maximum_cv": maximum_cv,
        "recommended_decision": recommended_decision,
        "reasons": reasons,
        "dispatch_entry": raw_entry,
        "public_evidence": public_evidence,
        "compatibility_record": compatibility,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pack", type=Path, required=True)
    result.add_argument("--dispatch", type=Path, required=True)
    result.add_argument("--entry-id", required=True)
    result.add_argument("--oracle", type=Path, required=True)
    result.add_argument("--timing", type=Path, required=True)
    result.add_argument("--external", type=Path, required=True)
    result.add_argument("--pack-verification", type=Path, required=True)
    result.add_argument("--runtime-verification", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    proposal = create_proposal(
        pack_path=args.pack,
        dispatch_path=args.dispatch,
        entry_id=args.entry_id,
        oracle_path=args.oracle,
        timing_path=args.timing,
        external_path=args.external,
        pack_verification_path=args.pack_verification,
        runtime_verification_path=args.runtime_verification,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite proposal: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proposal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"proposal_id": proposal["proposal_id"], "recommended_decision": proposal["recommended_decision"], "reasons": proposal["reasons"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
