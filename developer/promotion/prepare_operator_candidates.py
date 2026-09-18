#!/usr/bin/env python3
"""Stage every promotable result for one operator without changing public dispatch."""

from __future__ import annotations

import argparse
import json
import re
import shutil
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


def _promotable_results(evaluation: dict[str, Any], operator: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for result in evaluation["results"]:
        if result.get("operator") != operator or result.get("decision") != "promote":
            continue
        gates = result.get("gates")
        if not isinstance(gates, dict) or not gates or not all(gates.values()):
            raise ValueError(
                f"promote result {result.get('case_id')!r}/{result.get('dtype')!r} "
                "did not pass every worker gate"
            )
        selected.append(result)
    if not selected:
        raise ValueError(f"no promotable result for {operator}")
    return sorted(selected, key=lambda item: (item["case_id"], item["dtype"]))


def _workload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "dtype": result["dtype"],
        **result["shape"],
        "contiguous": True,
        "requires_grad": False,
    }


def prepare(
    *,
    evaluation_path: Path,
    artifact_path: Path,
    dispatch_path: Path,
    staged_dispatch_path: Path,
    pack_root: Path,
    run_label: str,
    pack_version: str = "0.1.0",
) -> dict[str, Any]:
    if staged_dispatch_path.exists():
        raise FileExistsError(f"refusing to overwrite staged output: {staged_dispatch_path}")
    evaluation = _load(evaluation_path)
    dispatch = _load(dispatch_path)
    manifest = DispatchManifest.from_mapping(dispatch, str(dispatch_path))
    operator = manifest.operator
    environment = _environment(evaluation)
    architecture = environment["architecture"]
    artifact_sha256 = sha256_file(artifact_path)
    if artifact_sha256 != evaluation["candidate"]["artifact_sha256"]:
        raise ValueError("artifact SHA-256 does not match evaluation")

    candidate_dispatch = deepcopy(dispatch)
    existing_ids = {item["id"] for item in candidate_dispatch["entries"]}
    entrypoints = evaluation["candidate"].get("entrypoints", {})
    plans: list[dict[str, Any]] = []
    for result in _promotable_results(evaluation, operator):
        entrypoint = result.get("selected_entrypoint")
        if not isinstance(entrypoint, str) or entrypoints.get(entrypoint) != entrypoint:
            raise ValueError(
                f"selected entrypoint is not present in evaluation candidate: {entrypoint!r}"
            )
        suffix = f"{result['case_id']}-{result['dtype']}"
        entry_id = f"{operator}-{architecture}-{suffix}"
        if entry_id in existing_ids:
            raise FileExistsError(f"dispatch entry already exists: {entry_id}")
        existing_ids.add(entry_id)
        evidence_id = f"{run_label}-{operator}-{architecture}-{suffix}"
        pack_id = f"rk-{operator.replace('_', '-')}-{architecture}-{run_label}-{suffix}"
        pack_path = pack_root / pack_id
        if pack_path.exists():
            raise FileExistsError(f"refusing to overwrite kernel pack: {pack_path}")
        artifact_mapping = {
            "provider": "native",
            "artifact": f"lib/{artifact_path.name}",
            "entrypoint": entrypoint,
            "sha256": artifact_sha256,
            "format": "python_extension",
        }
        candidate_dispatch["entries"].append(
            {
                "id": entry_id,
                "priority": 100,
                "environment": environment,
                "workload": _workload(result),
                "winner": artifact_mapping,
                "launch": {},
                "fallbacks": ["torch"],
                "evidence_id": evidence_id,
            }
        )
        plans.append(
            {
                "result": result,
                "entry_id": entry_id,
                "entrypoint": entrypoint,
                "evidence_id": evidence_id,
                "pack_id": pack_id,
                "pack_path": pack_path,
            }
        )

    candidate_dispatch["entries"].sort(key=lambda item: item["id"])
    DispatchManifest.from_mapping(candidate_dispatch, str(staged_dispatch_path))

    built_paths: list[Path] = []
    receipts: list[dict[str, Any]] = []
    try:
        for plan in plans:
            pack = build(
                SimpleNamespace(
                    source_artifact=artifact_path,
                    expected_sha256=artifact_sha256,
                    output=plan["pack_path"],
                    pack_id=plan["pack_id"],
                    pack_version=pack_version,
                    architecture=environment["architecture"],
                    wavefront_size=environment["wavefront_size"],
                    rocm_abi=environment["rocm_abi"],
                    python_abi=environment["python_abi"],
                    pytorch_version=environment["pytorch_version"],
                    operator=operator,
                    semantic_version=manifest.semantic_version,
                    provider="native",
                    entrypoint=plan["entrypoint"],
                    format="python_extension",
                    evidence_id=plan["evidence_id"],
                )
            )
            built_paths.append(pack.root)
            result = plan["result"]
            receipts.append(
                {
                    "entry_id": plan["entry_id"],
                    "entrypoint": plan["entrypoint"],
                    "evidence_id": plan["evidence_id"],
                    "pack_id": pack.pack_id,
                    "pack_path": str(pack.root),
                    "case_id": result["case_id"],
                    "dtype": result["dtype"],
                    "latency_reduction_pct": result["timing"]["latency_reduction_pct"],
                }
            )
        _write_new(staged_dispatch_path, candidate_dispatch)
    except Exception:
        for path in built_paths:
            shutil.rmtree(path, ignore_errors=True)
        raise

    return {
        "operator": operator,
        "architecture": architecture,
        "environment": environment,
        "source_dispatch_sha256": sha256_file(dispatch_path),
        "staged_dispatch_path": str(staged_dispatch_path),
        "staged_dispatch_sha256": sha256_file(staged_dispatch_path),
        "packs": receipts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--staged-dispatch", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--pack-version", default="0.1.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
    receipt = prepare(
        evaluation_path=args.evaluation,
        artifact_path=args.artifact,
        dispatch_path=args.dispatch,
        staged_dispatch_path=args.staged_dispatch,
        pack_root=args.pack_root,
        run_label=args.run_label,
        pack_version=args.pack_version,
    )
    _write_new(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
