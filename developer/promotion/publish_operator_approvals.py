#!/usr/bin/env python3
"""Publish every approved proposal in a preflighted private operator batch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from developer.promotion.publish_approved import publish


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite publication output: {path}")
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


def _plans(
    *,
    proposal_directory: Path,
    approval_directory: Path,
    pack_root: Path,
    dispatch_directory: Path,
    public_root: Path,
    release_root: Path,
    receipt_directory: Path,
) -> list[dict[str, Any]]:
    proposal_paths = sorted(proposal_directory.glob("*.json"))
    if not proposal_paths:
        raise ValueError("proposal directory contains no JSON proposals")
    plans: list[dict[str, Any]] = []
    immutable_destinations: set[Path] = set()
    for proposal_path in proposal_paths:
        proposal = _load(proposal_path)
        proposal_id = proposal["proposal_id"]
        approval_id = f"approval-{proposal_id}-approved"
        approval_path = approval_directory / f"{approval_id}.json"
        pack = proposal["pack"]
        operator = proposal["operator"]
        semantic_version = proposal["semantic_version"]
        release_name = f"{pack['pack_id']}-{pack['pack_version']}.tar.gz"
        plan = {
            "proposal": proposal_path,
            "approval": approval_path,
            "pack": pack_root / pack["pack_id"],
            "dispatch": dispatch_directory / f"{operator}-{semantic_version}.json",
            "receipt": receipt_directory / f"{proposal['entry_id']}.json",
            "evidence": public_root / "evidence" / f"{proposal['public_evidence']['evidence_id']}.json",
            "public_approval": public_root / "approvals" / f"{approval_id}.json",
            "release": release_root / release_name,
        }
        for key in ("proposal", "approval", "pack", "dispatch"):
            if not plan[key].exists():
                raise FileNotFoundError(f"missing batch input: {plan[key]}")
        for key in ("receipt", "evidence", "public_approval", "release"):
            destination = plan[key]
            if destination in immutable_destinations or destination.exists():
                raise FileExistsError(f"refusing conflicting batch publication: {destination}")
            immutable_destinations.add(destination)
        plans.append(plan)
    return plans


def publish_batch(
    *,
    proposal_directory: Path,
    approval_directory: Path,
    pack_root: Path,
    dispatch_directory: Path,
    private_key_path: Path,
    trusted_keys_path: Path,
    public_root: Path,
    release_root: Path,
    receipt_directory: Path,
) -> dict[str, Any]:
    plans = _plans(
        proposal_directory=proposal_directory,
        approval_directory=approval_directory,
        pack_root=pack_root,
        dispatch_directory=dispatch_directory,
        public_root=public_root,
        release_root=release_root,
        receipt_directory=receipt_directory,
    )
    receipts: list[dict[str, Any]] = []
    for position, plan in enumerate(plans, start=1):
        receipt = publish(
            proposal_path=plan["proposal"],
            approval_path=plan["approval"],
            pack_path=plan["pack"],
            dispatch_path=plan["dispatch"],
            private_key_path=private_key_path,
            trusted_keys_path=trusted_keys_path,
            public_root=public_root,
            release_root=release_root,
        )
        receipt["batch_position"] = position
        receipt["batch_size"] = len(plans)
        _write_new(plan["receipt"], receipt)
        receipts.append(receipt)
        print(
            json.dumps(
                {
                    "published": position,
                    "total": len(plans),
                    "pack_id": receipt["pack_id"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "count": len(receipts),
        "releases": receipts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-directory", type=Path, required=True)
    parser.add_argument("--approval-directory", type=Path, required=True)
    parser.add_argument("--pack-root", type=Path, required=True)
    parser.add_argument("--dispatch-directory", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--trusted-keys", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--receipt-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = publish_batch(
        proposal_directory=args.proposal_directory,
        approval_directory=args.approval_directory,
        pack_root=args.pack_root,
        dispatch_directory=args.dispatch_directory,
        private_key_path=args.private_key,
        trusted_keys_path=args.trusted_keys,
        public_root=args.public_root,
        release_root=args.release_root,
        receipt_directory=args.receipt_directory,
    )
    _write_new(args.output, summary)
    print(json.dumps({"published": summary["count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
