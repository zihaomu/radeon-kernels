#!/usr/bin/env python3
"""Select stable M5 winners and build architecture-specific candidate packs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from developer.promotion.build_kernel_pack import build
from radeon_kernels.runtime.fingerprint import normalize_rocm_abi
from radeon_kernels.runtime.registry import DispatchManifest
from radeon_kernels.runtime.signing import sha256_file


ENTRYPOINTS = {
    "rms_norm": "rms_norm",
    "add_rms_norm": "add_rms_norm",
    "rope": "rope",
    "swiglu": "silu_mul",
    "softmax": "softmax",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _python_abi(version: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(f"invalid Python version: {version!r}")
    return f"cp{match.group(1)}{match.group(2)}"


def _best_results(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for operator in ENTRYPOINTS:
        candidates = [
            item
            for item in evaluation["results"]
            if item["operator"] == operator
            and item["decision"] == "promote"
            and all(item["gates"].values())
        ]
        if not candidates:
            raise ValueError(f"no promotable result for {operator}")
        selected[operator] = max(
            candidates, key=lambda item: item["timing"]["latency_reduction_pct"]
        )
    return selected


def _workload(result: dict[str, Any]) -> dict[str, Any]:
    values = {
        "dtype": result["dtype"],
        **result["shape"],
        "contiguous": True,
        "requires_grad": False,
    }
    if result["operator"] == "rope":
        values["mode"] = "split_half"
    return values


def prepare(
    *,
    evaluation_path: Path,
    artifact_path: Path,
    dispatch_root: Path,
    pack_root: Path,
    run_label: str,
) -> dict[str, Any]:
    evaluation = _load(evaluation_path)
    environment = evaluation["environment"]
    architecture = environment["architecture"]
    artifact_sha256 = sha256_file(artifact_path)
    if artifact_sha256 != evaluation["candidate"]["artifact_sha256"]:
        raise ValueError("artifact SHA-256 does not match evaluation")
    normalized_environment = {
        "architecture": architecture,
        "wavefront_size": int(environment["wavefront_size"]),
        "rocm_abi": normalize_rocm_abi(environment["rocm"]),
        "python_abi": _python_abi(environment["python"]),
        "pytorch_version": environment["pytorch"],
    }
    receipts: list[dict[str, Any]] = []
    for operator, result in _best_results(evaluation).items():
        evidence_id = (
            f"{run_label}-{operator}-{architecture}-{result['case_id']}-{result['dtype']}"
        )
        entry_id = f"{operator}-{architecture}-{result['case_id']}-{result['dtype']}"
        pack_id = f"rk-{operator.replace('_', '-')}-{architecture}-{run_label}"
        dispatch_path = dispatch_root / f"{operator}-1.0.json"
        dispatch = _load(dispatch_path)
        if any(item["id"] == entry_id for item in dispatch["entries"]):
            raise FileExistsError(f"dispatch entry already exists: {entry_id}")
        entry = {
            "id": entry_id,
            "priority": 100,
            "environment": normalized_environment,
            "workload": _workload(result),
            "winner": {
                "provider": "native",
                "artifact": f"lib/{artifact_path.name}",
                "entrypoint": ENTRYPOINTS[operator],
                "sha256": artifact_sha256,
                "format": "python_extension",
            },
            "launch": {},
            "fallbacks": ["torch"],
            "evidence_id": evidence_id,
        }
        dispatch["entries"].append(entry)
        dispatch["entries"].sort(key=lambda item: item["id"])
        DispatchManifest.from_mapping(dispatch, str(dispatch_path))
        _write(dispatch_path, dispatch)
        pack_path = pack_root / pack_id
        pack = build(
            SimpleNamespace(
                source_artifact=artifact_path,
                expected_sha256=artifact_sha256,
                output=pack_path,
                pack_id=pack_id,
                pack_version="0.1.0",
                architecture=architecture,
                wavefront_size=normalized_environment["wavefront_size"],
                rocm_abi=normalized_environment["rocm_abi"],
                python_abi=normalized_environment["python_abi"],
                pytorch_version=normalized_environment["pytorch_version"],
                operator=operator,
                semantic_version="1.0",
                provider="native",
                entrypoint=ENTRYPOINTS[operator],
                format="python_extension",
                evidence_id=evidence_id,
            )
        )
        receipts.append(
            {
                "operator": operator,
                "entry_id": entry_id,
                "evidence_id": evidence_id,
                "pack_id": pack.pack_id,
                "pack_path": str(pack.root),
                "dispatch_path": str(dispatch_path),
                "case_id": result["case_id"],
                "dtype": result["dtype"],
                "latency_reduction_pct": result["timing"]["latency_reduction_pct"],
            }
        )
    return {"architecture": architecture, "environment": normalized_environment, "packs": receipts}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dispatch-root", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--run-label", default="m5-20260917")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = prepare(
        evaluation_path=args.evaluation,
        artifact_path=args.artifact,
        dispatch_root=args.dispatch_root,
        pack_root=args.pack_root,
        run_label=args.run_label,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
    _write(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
