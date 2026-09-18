#!/usr/bin/env python3
"""Create a generic signed-pack proposal from an M5 evaluation result."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from radeon_kernels.errors import ConfigError
from radeon_kernels.runtime.fingerprint import normalize_rocm_abi
from radeon_kernels.runtime.pack import KernelPack
from radeon_kernels.runtime.registry import DispatchManifest
from radeon_kernels.runtime.signing import sha256_file


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(str(path), f"cannot read JSON: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(str(path), "expected a mapping")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _latency_reduction(candidate: Sequence[float], baseline: Sequence[float]) -> float:
    return (1.0 - statistics.median(candidate) / statistics.median(baseline)) * 100.0


def _bootstrap_interval(
    candidate: Sequence[float], baseline: Sequence[float], *, resamples: int = 10_000
) -> tuple[float, float]:
    if len(candidate) != len(baseline) or len(candidate) < 2:
        raise ConfigError("evaluation.samples", "candidate and baseline samples must be paired")
    generator = random.Random(20260917)
    estimates: list[float] = []
    for _ in range(resamples):
        indexes = [generator.randrange(len(candidate)) for _ in candidate]
        estimates.append(
            _latency_reduction(
                [candidate[index] for index in indexes],
                [baseline[index] for index in indexes],
            )
        )
    estimates.sort()
    return estimates[int(0.025 * (resamples - 1))], estimates[int(0.975 * (resamples - 1))]


def _environment(pack: KernelPack) -> dict[str, Any]:
    return {
        "architecture": pack.environment.architecture,
        "wavefront_size": pack.environment.wavefront_size,
        "rocm_abi": pack.environment.rocm_abi,
        "python_abi": pack.environment.python_abi,
        "pytorch_version": pack.environment.pytorch_version,
    }


def _artifact(item: Any) -> dict[str, Any]:
    return {
        "operator": item.operator,
        "semantic_version": item.semantic_version,
        **item.artifact.to_dict(),
        "evidence_id": item.evidence_id,
    }


def _method_name(evaluation: Mapping[str, Any], key: str, default: str) -> str:
    method = evaluation.get("method")
    if not isinstance(method, Mapping):
        return default
    value = method.get(key)
    return value if isinstance(value, str) and value else default


def create_proposal(
    *,
    evaluation_path: Path,
    pack_path: Path,
    dispatch_path: Path,
    entry_id: str,
    minimum_improvement_pct: float = 3.0,
    maximum_cv: float = 0.03,
) -> dict[str, Any]:
    evaluation = _load(evaluation_path)
    pack = KernelPack.from_directory(pack_path, verify_artifacts=True)
    dispatch_payload = _load(dispatch_path)
    manifest = DispatchManifest.from_mapping(dispatch_payload, str(dispatch_path))
    raw_entries = [item for item in dispatch_payload["entries"] if item["id"] == entry_id]
    entries = [item for item in manifest.entries if item.id == entry_id]
    if len(entries) != 1 or len(raw_entries) != 1:
        raise ConfigError(str(dispatch_path), f"expected one dispatch entry {entry_id!r}")
    entry = entries[0]
    raw_entry = raw_entries[0]
    item = pack.get(manifest.operator, manifest.semantic_version)
    results = [
        result
        for result in evaluation["results"]
        if result["operator"] == manifest.operator
        and entry_id
        == (
            f"{manifest.operator}-{evaluation['environment']['architecture']}-"
            f"{result['case_id']}-{result['dtype']}"
        )
        and result["dtype"] == raw_entry["workload"]["dtype"]
    ]
    if len(results) != 1:
        raise ConfigError(str(evaluation_path), "cannot identify one evaluated dispatch workload")
    result = results[0]
    candidate = [float(value) for value in result["timing"]["candidate"]["samples_ms"]]
    baseline = [float(value) for value in result["timing"]["baseline"]["samples_ms"]]
    confidence = _bootstrap_interval(candidate, baseline)
    latency_reduction = _latency_reduction(candidate, baseline)
    evaluation_environment = evaluation["environment"]
    expected_environment = _environment(pack)
    workload = dict(raw_entry["workload"])
    artifact_mapping = item.artifact.to_dict()
    gates = {
        "artifact_matches_dispatch": artifact_mapping == entry.winner.to_dict(),
        "artifact_hash_consistent": evaluation["candidate"]["artifact_sha256"] == item.artifact.sha256,
        "environment_consistent": (
            evaluation_environment["architecture"] == pack.environment.architecture
            and int(evaluation_environment["wavefront_size"]) == pack.environment.wavefront_size
            and normalize_rocm_abi(evaluation_environment["rocm"]) == pack.environment.rocm_abi
            and evaluation_environment["pytorch"] == pack.environment.pytorch_version
        ),
        "workload_consistent": (
            result["dtype"] == workload["dtype"]
            and all(workload.get(key) == value for key, value in result["shape"].items())
            and workload["contiguous"] is True
            and workload["requires_grad"] is False
        ),
        "correctness_passed": all(case["passed"] for case in result["correctness"]),
        "candidate_stable": result["timing"]["candidate"]["coefficient_of_variation"] <= maximum_cv,
        "baseline_stable": result["timing"]["baseline"]["coefficient_of_variation"] <= maximum_cv,
        "improvement_threshold_passed": latency_reduction >= minimum_improvement_pct,
        "confidence_threshold_passed": confidence[0] >= minimum_improvement_pct,
        "worker_gates_passed": result["decision"] == "promote" and all(result["gates"].values()),
    }
    failed = [name for name, passed in gates.items() if not passed]
    decision = "promote" if not failed else "needs_review"
    sources = {
        "candidate_source": evaluation["candidate"]["source_sha256"],
        "dispatch": sha256_file(dispatch_path),
        "evaluation": sha256_file(evaluation_path),
        "manifest": sha256_file(pack.root / "manifest.json"),
    }
    public_cases = [
        {
            "case_id": f"{result['case_id']}-{result['dtype']}-seed{case['seed']}",
            "seed": int(case["seed"]),
            "runtime_ok": bool(case["passed"]),
            "finite": bool(case["passed"]),
            "max_absolute_error": float(case["max_absolute_error"]),
        }
        for case in result["correctness"]
    ]
    public_evidence = {
        "schema_version": "1.0",
        "evidence_id": item.evidence_id,
        "operator": item.operator,
        "semantic_version": item.semantic_version,
        "candidate_content_hash": item.artifact.sha256,
        "captured_at": evaluation["captured_at"],
        "environment": expected_environment,
        "workload": workload,
        "correctness": {
            "passed": gates["correctness_passed"],
            "oracle": _method_name(evaluation, "oracle", "torch_fp32_reference"),
            "cases": public_cases,
        },
        "timing": {
            "baseline": _method_name(evaluation, "baseline", "torch_fp32_reference"),
            "candidate_samples_ms": candidate,
            "baseline_samples_ms": baseline,
            "candidate_median_ms": float(result["timing"]["candidate"]["median_ms"]),
            "baseline_median_ms": float(result["timing"]["baseline"]["median_ms"]),
            "candidate_cv": float(result["timing"]["candidate"]["coefficient_of_variation"]),
            "baseline_cv": float(result["timing"]["baseline"]["coefficient_of_variation"]),
            "latency_reduction_pct": latency_reduction,
            "confidence_interval_pct": list(confidence),
            "stable": gates["candidate_stable"] and gates["baseline_stable"],
        },
        "source_evidence_hashes": sources,
        "decision": decision,
        "guarantee_boundary": (
            f"Exact {pack.environment.architecture}, ROCm {pack.environment.rocm_abi}, "
            f"{pack.environment.python_abi}, PyTorch {pack.environment.pytorch_version}, "
            f"workload {json.dumps(workload, sort_keys=True)}, contiguous inference only."
        ),
    }
    compatibility = {
        "id": f"{item.operator}-{pack.environment.architecture}-{item.evidence_id}",
        **expected_environment,
        "provider": item.artifact.provider,
        "status": "supported" if decision == "promote" else "experimental",
        "evidence_id": item.evidence_id,
        "reason": "all deterministic promotion gates passed" if not failed else "; ".join(failed),
    }
    identity = hashlib.sha256(
        json.dumps({"evidence_id": item.evidence_id, "sources": sources}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return {
        "schema_version": "1.0",
        "proposal_id": f"proposal-{item.evidence_id}-{identity}",
        "created_at": evaluation["captured_at"],
        "operator": item.operator,
        "semantic_version": item.semantic_version,
        "entry_id": entry.id,
        "pack": {
            "pack_id": pack.pack_id,
            "pack_version": pack.pack_version,
            "manifest_sha256": sources["manifest"],
            "environment": expected_environment,
            "artifacts": [_artifact(artifact) for artifact in pack.artifacts],
        },
        "source_evidence_hashes": sources,
        "gates": gates,
        "minimum_improvement_pct": minimum_improvement_pct,
        "maximum_cv": maximum_cv,
        "recommended_decision": decision,
        "reasons": ["all deterministic promotion gates passed"] if not failed else failed,
        "dispatch_entry": raw_entry,
        "public_evidence": public_evidence,
        "compatibility_record": compatibility,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--entry-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    proposal = create_proposal(
        evaluation_path=args.evaluation,
        pack_path=args.pack,
        dispatch_path=args.dispatch,
        entry_id=args.entry_id,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite proposal: {args.output}")
    _write(args.output, proposal)
    print(json.dumps({"proposal_id": proposal["proposal_id"], "recommended_decision": proposal["recommended_decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
