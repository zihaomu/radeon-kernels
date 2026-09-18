#!/usr/bin/env python3
"""Prepare an immutable pack and staged dispatch for a measured winner replacement."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from developer.promotion.build_kernel_pack import build
from radeon_kernels.runtime.fingerprint import normalize_rocm_abi
from radeon_kernels.runtime.registry import DispatchManifest
from radeon_kernels.runtime.signing import sha256_file


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite staged output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _python_abi(version: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(f"invalid Python version: {version!r}")
    return f"cp{match.group(1)}{match.group(2)}"


def _environment(evaluation: dict[str, Any]) -> dict[str, Any]:
    source = evaluation["environment"]
    return {
        "architecture": source["architecture"],
        "wavefront_size": int(source["wavefront_size"]),
        "rocm_abi": normalize_rocm_abi(source["rocm"]),
        "python_abi": _python_abi(source["python"]),
        "pytorch_version": source["pytorch"],
    }


def prepare(
    *,
    evaluation_path: Path,
    artifact_path: Path,
    dispatch_path: Path,
    staged_dispatch_path: Path,
    pack_path: Path,
    entry_id: str,
    pack_id: str,
    pack_version: str,
    run_label: str,
) -> dict[str, Any]:
    evaluation = _load(evaluation_path)
    dispatch = _load(dispatch_path)
    DispatchManifest.from_mapping(dispatch, str(dispatch_path))
    environment = _environment(evaluation)
    architecture = environment["architecture"]
    artifact_sha256 = sha256_file(artifact_path)
    if artifact_sha256 != evaluation["candidate"]["artifact_sha256"]:
        raise ValueError("artifact SHA-256 does not match evaluation")

    matching_entries = [item for item in dispatch["entries"] if item["id"] == entry_id]
    if len(matching_entries) != 1:
        raise ValueError(f"expected exactly one dispatch entry {entry_id!r}")
    entry = matching_entries[0]
    if entry["environment"] != environment:
        raise ValueError("evaluation environment does not match the existing dispatch entry")

    matching_results = [
        item
        for item in evaluation["results"]
        if item["operator"] == dispatch["operator"]
        and entry_id
        == f"{dispatch['operator']}-{architecture}-{item['case_id']}-{item['dtype']}"
    ]
    if len(matching_results) != 1:
        raise ValueError("cannot identify exactly one evaluated replacement workload")
    result = matching_results[0]
    if result["decision"] != "promote" or not all(result["gates"].values()):
        raise ValueError("replacement result did not pass every worker gate")
    expected_workload = {
        "dtype": result["dtype"],
        **result["shape"],
        "contiguous": True,
        "requires_grad": False,
    }
    if entry["workload"] != expected_workload:
        raise ValueError("evaluated workload does not exactly match the existing dispatch entry")

    evidence_id = (
        f"{run_label}-{dispatch['operator']}-{architecture}-"
        f"{result['case_id']}-{result['dtype']}"
    )
    artifact_mapping = {
        "provider": "native",
        "artifact": f"lib/{artifact_path.name}",
        "entrypoint": evaluation["candidate"]["entrypoints"][dispatch["operator"]],
        "sha256": artifact_sha256,
        "format": "python_extension",
    }
    candidate_dispatch = deepcopy(dispatch)
    candidate_entry = next(item for item in candidate_dispatch["entries"] if item["id"] == entry_id)
    previous_winner = deepcopy(candidate_entry["winner"])
    previous_evidence_id = candidate_entry["evidence_id"]
    candidate_entry["winner"] = artifact_mapping
    candidate_entry["evidence_id"] = evidence_id
    DispatchManifest.from_mapping(candidate_dispatch, str(staged_dispatch_path))

    pack = build(
        SimpleNamespace(
            source_artifact=artifact_path,
            expected_sha256=artifact_sha256,
            output=pack_path,
            pack_id=pack_id,
            pack_version=pack_version,
            architecture=environment["architecture"],
            wavefront_size=environment["wavefront_size"],
            rocm_abi=environment["rocm_abi"],
            python_abi=environment["python_abi"],
            pytorch_version=environment["pytorch_version"],
            operator=dispatch["operator"],
            semantic_version=dispatch["semantic_version"],
            provider="native",
            entrypoint=artifact_mapping["entrypoint"],
            format="python_extension",
            evidence_id=evidence_id,
        )
    )
    _write_new(staged_dispatch_path, candidate_dispatch)
    return {
        "operator": dispatch["operator"],
        "architecture": architecture,
        "entry_id": entry_id,
        "evidence_id": evidence_id,
        "pack_id": pack.pack_id,
        "pack_version": pack.pack_version,
        "pack_path": str(pack.root),
        "staged_dispatch_path": str(staged_dispatch_path),
        "previous_winner": previous_winner,
        "previous_evidence_id": previous_evidence_id,
        "replacement_winner": artifact_mapping,
        "latency_reduction_vs_torch_pct": result["timing"]["latency_reduction_pct"],
        "latency_reduction_vs_previous_pct": result["timing"][
            "latency_reduction_vs_three_pass_pct"
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--staged-dispatch", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--entry-id", required=True)
    parser.add_argument("--pack-id", required=True)
    parser.add_argument("--pack-version", default="0.1.0")
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = prepare(
        evaluation_path=args.evaluation,
        artifact_path=args.artifact,
        dispatch_path=args.dispatch,
        staged_dispatch_path=args.staged_dispatch,
        pack_path=args.pack,
        entry_id=args.entry_id,
        pack_id=args.pack_id,
        pack_version=args.pack_version,
        run_label=args.run_label,
    )
    _write_new(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
