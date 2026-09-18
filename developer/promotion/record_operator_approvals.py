#!/usr/bin/env python3
"""Record one explicit decision for every proposal in a private batch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from developer.promotion.record_approval import record_decision


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite approval output: {path}")
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


def record_batch(
    *,
    proposal_directory: Path,
    output_directory: Path,
    reviewer: str,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    proposal_paths = sorted(proposal_directory.glob("*.json"))
    if not proposal_paths:
        raise ValueError("proposal directory contains no JSON proposals")
    decisions = [
        (
            path,
            record_decision(
                proposal_path=path,
                reviewer=reviewer,
                decision=decision,
                reason=reason,
            ),
        )
        for path in proposal_paths
    ]
    plans = [
        (output_directory / f"{approval['approval_id']}.json", approval)
        for _, approval in decisions
    ]
    existing = [path for path, _ in plans if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite approval output: {existing[0]}")
    for path, approval in plans:
        _write_new(path, approval)
    return {
        "reviewer": reviewer,
        "decision": decision,
        "count": len(plans),
        "override_count": sum(approval.get("override") is True for _, approval in plans),
        "approvals": [
            {
                "approval_id": approval["approval_id"],
                "proposal_id": approval["proposal_id"],
                "proposal_sha256": approval["proposal_sha256"],
                "path": str(path),
            }
            for path, approval in plans
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reason", required=True)
    decision = parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = record_batch(
        proposal_directory=args.proposal_directory,
        output_directory=args.output_directory,
        reviewer=args.reviewer,
        decision="approved" if args.approve else "rejected",
        reason=args.reason,
    )
    _write_new(args.output, summary)
    print(json.dumps({key: summary[key] for key in ("reviewer", "decision", "count", "override_count")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
