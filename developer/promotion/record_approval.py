#!/usr/bin/env python3
"""Record an explicit human decision for an immutable promotion proposal."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radeon_kernels.errors import ConfigError
from radeon_kernels.runtime.signing import sha256_file

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(str(path), f"cannot read JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise ConfigError(str(path), "expected a mapping")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ConfigError(path, "expected a lowercase identifier")
    return value


def validate_approval(
    value: Any,
    *,
    path: str = "approval",
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(path, "expected a mapping")
    fields = {
        "schema_version",
        "approval_id",
        "proposal_id",
        "proposal_sha256",
        "decision",
        "reviewer",
        "reviewed_at",
        "reason",
        "override",
    }
    required = fields - {"override"}
    unknown = set(value) - fields
    missing = required - set(value)
    if unknown or missing:
        raise ConfigError(path, f"unknown={sorted(unknown)}, missing={sorted(missing)}")
    if value["schema_version"] != "1.0":
        raise ConfigError(f"{path}.schema_version", "unsupported version")
    _identifier(value["approval_id"], f"{path}.approval_id")
    _identifier(value["proposal_id"], f"{path}.proposal_id")
    _identifier(value["reviewer"], f"{path}.reviewer")
    digest = value["proposal_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ConfigError(f"{path}.proposal_sha256", "expected a lowercase SHA-256")
    if value["decision"] not in {"approved", "rejected"}:
        raise ConfigError(f"{path}.decision", "expected approved or rejected")
    if not isinstance(value["reviewed_at"], str) or not value["reviewed_at"]:
        raise ConfigError(f"{path}.reviewed_at", "expected a timestamp")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ConfigError(f"{path}.reason", "expected a non-empty reason")
    if "override" in value and not isinstance(value["override"], bool):
        raise ConfigError(f"{path}.override", "expected a boolean")
    return value


def record_decision(
    *,
    proposal_path: Path,
    reviewer: str,
    decision: str,
    reason: str,
    override_gates: bool = False,
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    proposal = _load_mapping(proposal_path)
    proposal_id = _identifier(proposal.get("proposal_id"), "proposal.proposal_id")
    recommended = proposal.get("recommended_decision")
    if recommended not in {"promote", "needs_review", "reject"}:
        raise ConfigError("proposal.recommended_decision", "unsupported decision")
    if decision not in {"approved", "rejected"}:
        raise ConfigError("decision", "expected approved or rejected")
    normalized_reviewer = _identifier(reviewer, "reviewer")
    normalized_reason = reason.strip()
    if not normalized_reason:
        raise ConfigError("reason", "expected a non-empty reason")
    if decision == "approved" and recommended != "promote" and not override_gates:
        raise ConfigError(
            "approval",
            "proposal did not pass promotion gates; use an explicit gate override to approve",
        )
    if override_gates and decision != "approved":
        raise ConfigError("approval.override", "gate override is valid only for approval")

    approval = {
        "schema_version": "1.0",
        "approval_id": f"approval-{proposal_id}-{decision}",
        "proposal_id": proposal_id,
        "proposal_sha256": sha256_file(proposal_path),
        "decision": decision,
        "reviewer": normalized_reviewer,
        "reviewed_at": reviewed_at or datetime.now(UTC).isoformat(),
        "reason": normalized_reason,
    }
    if override_gates:
        approval["override"] = True
    validate_approval(approval)
    return approval


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--proposal", type=Path, required=True)
    result.add_argument("--reviewer", required=True)
    result.add_argument("--reason", required=True)
    decision = result.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    result.add_argument("--override-gates", action="store_true")
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    approval = record_decision(
        proposal_path=args.proposal,
        reviewer=args.reviewer,
        decision="approved" if args.approve else "rejected",
        reason=args.reason,
        override_gates=args.override_gates,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite approval: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(approval, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
