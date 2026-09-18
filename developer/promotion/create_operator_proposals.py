#!/usr/bin/env python3
"""Create proposals for every pack listed in an operator staging receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from developer.promotion.create_operator_proposal import create_proposal


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite proposal: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def create_all(
    *,
    evaluation_path: Path,
    receipt_path: Path,
    dispatch_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    receipt = _load(receipt_path)
    packs = receipt.get("packs")
    if not isinstance(packs, list) or not packs:
        raise ValueError("staging receipt contains no packs")
    plans = [
        {
            "entry_id": item["entry_id"],
            "pack_path": Path(item["pack_path"]),
            "output": output_directory / f"{item['entry_id']}.json",
        }
        for item in packs
    ]
    existing = [plan["output"] for plan in plans if plan["output"].exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite proposal: {existing[0]}")

    proposals = [
        (
            plan,
            create_proposal(
                evaluation_path=evaluation_path,
                pack_path=plan["pack_path"],
                dispatch_path=dispatch_path,
                entry_id=plan["entry_id"],
            ),
        )
        for plan in plans
    ]
    for plan, proposal in proposals:
        _write_new(plan["output"], proposal)
    return {
        "operator": receipt["operator"],
        "architecture": receipt["architecture"],
        "proposals": [
            {
                "proposal_id": proposal["proposal_id"],
                "entry_id": plan["entry_id"],
                "recommended_decision": proposal["recommended_decision"],
                "path": str(plan["output"]),
            }
            for plan, proposal in proposals
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = create_all(
        evaluation_path=args.evaluation,
        receipt_path=args.receipt,
        dispatch_path=args.dispatch,
        output_directory=args.output_directory,
    )
    _write_new(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
