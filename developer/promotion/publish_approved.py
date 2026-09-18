#!/usr/bin/env python3
"""Publish a signed kernel pack only after a matching human approval."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from developer.promotion.record_approval import validate_approval
from developer.promotion.signing import sign_file
from radeon_kernels.errors import ConfigError
from radeon_kernels.runtime.pack import KernelPack
from radeon_kernels.runtime.registry import DispatchManifest
from radeon_kernels.runtime.signing import (
    TrustedKeyring,
    sha256_file,
    verify_detached_file,
)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(str(path), f"cannot read JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise ConfigError(str(path), "expected a mapping")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_schema(value: Mapping[str, Any], schema_name: str) -> None:
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / f"{schema_name}.schema.json"
    schema = _load_mapping(schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(value), key=lambda item: tuple(item.absolute_path))
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "document"
        raise ConfigError(f"{schema_name}.{location}", error.message)


def _safe_relative(value: str, path: str) -> PurePosixPath:
    parsed = PurePosixPath(value)
    if not value or parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != value:
        raise ConfigError(path, "expected a normalized relative path")
    return parsed


def _load_signed_or_default(
    path: Path,
    signature_path: Path,
    keyring: TrustedKeyring,
    default: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.exists() and not signature_path.exists():
        return dict(default)
    if not path.is_file() or not signature_path.is_file():
        raise ConfigError(str(path), "signed public document is incomplete")
    verify_detached_file(path, signature_path, keyring, signed_file=path.name)
    return dict(_load_mapping(path))


def _copy_pack_payload(pack: KernelPack, destination: Path) -> None:
    destination.mkdir(parents=True)
    shutil.copy2(pack.root / "manifest.json", destination / "manifest.json")
    for index, item in enumerate(pack.artifacts):
        relative = _safe_relative(item.artifact.artifact, f"pack.artifacts[{index}].artifact")
        source = pack.root.joinpath(*relative.parts)
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _deterministic_archive(pack_root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive,
    ):
        paths = [pack_root, *sorted(pack_root.rglob("*"), key=lambda item: item.as_posix())]
        for path in paths:
            relative = path.relative_to(pack_root.parent).as_posix()
            info = archive.gettarinfo(str(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            if path.is_dir() or path.suffix in {".so", ".hsaco"}:
                info.mode = 0o755
            else:
                info.mode = 0o644
            if path.is_file():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)


def _assert_proposal(
    proposal: Mapping[str, Any],
    approval: Mapping[str, Any],
    proposal_path: Path,
    *,
    allow_override: bool,
) -> None:
    if approval["proposal_id"] != proposal.get("proposal_id"):
        raise ConfigError("approval.proposal_id", "does not match proposal")
    if approval["proposal_sha256"] != sha256_file(proposal_path):
        raise ConfigError("approval.proposal_sha256", "does not match proposal content")
    if approval["decision"] != "approved":
        raise ConfigError("approval.decision", "publication requires an approved decision")
    recommended = proposal.get("recommended_decision")
    override = approval.get("override") is True
    if recommended != "promote" and not (override and allow_override):
        raise ConfigError(
            "proposal.recommended_decision",
            "proposal did not pass gates; publication requires both recorded and CLI override",
        )
    gates = proposal.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise ConfigError("proposal.gates", "expected promotion gates")
    if recommended == "promote" and not all(value is True for value in gates.values()):
        raise ConfigError("proposal.gates", "promote proposal contains a failed gate")


def _assert_pack_and_dispatch(
    proposal: Mapping[str, Any],
    pack: KernelPack,
    dispatch_path: Path,
) -> None:
    pack_proposal = proposal.get("pack")
    if not isinstance(pack_proposal, Mapping):
        raise ConfigError("proposal.pack", "expected a mapping")
    if pack_proposal.get("pack_id") != pack.pack_id:
        raise ConfigError("proposal.pack.pack_id", "does not match candidate pack")
    manifest_path = pack.root / "manifest.json"
    if pack_proposal.get("manifest_sha256") != sha256_file(manifest_path):
        raise ConfigError("proposal.pack.manifest_sha256", "does not match candidate pack")
    dispatch = _load_mapping(dispatch_path)
    DispatchManifest.from_mapping(dispatch, str(dispatch_path))
    if proposal.get("source_evidence_hashes", {}).get("dispatch") != sha256_file(dispatch_path):
        raise ConfigError("proposal.source_evidence_hashes.dispatch", "public dispatch changed")
    entry_id = proposal.get("entry_id")
    entries = [entry for entry in dispatch.get("entries", []) if entry.get("id") == entry_id]
    if len(entries) != 1 or entries[0] != proposal.get("dispatch_entry"):
        raise ConfigError("proposal.dispatch_entry", "does not exactly match public dispatch")


def _commit_files(files: Sequence[tuple[Path, Path]], immutable: set[Path]) -> None:
    for _, destination in files:
        if destination in immutable and destination.exists():
            raise FileExistsError(f"refusing to overwrite immutable publication: {destination}")
    originals = {destination: destination.read_bytes() if destination.exists() else None for _, destination in files}
    committed: list[Path] = []
    try:
        for source, destination in files:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            committed.append(destination)
    except Exception:
        for destination in reversed(committed):
            original = originals[destination]
            if original is None:
                destination.unlink(missing_ok=True)
            else:
                destination.write_bytes(original)
        raise


def publish(
    *,
    proposal_path: Path,
    approval_path: Path,
    pack_path: Path,
    dispatch_path: Path,
    private_key_path: Path,
    trusted_keys_path: Path,
    public_root: Path,
    release_root: Path,
    allow_override: bool = False,
) -> dict[str, Any]:
    proposal = _load_mapping(proposal_path)
    approval = validate_approval(_load_mapping(approval_path), path=str(approval_path))
    _validate_schema(approval, "approval")
    _assert_proposal(proposal, approval, proposal_path, allow_override=allow_override)
    pack = KernelPack.from_directory(pack_path, verify_artifacts=True)
    _assert_pack_and_dispatch(proposal, pack, dispatch_path)
    keyring = TrustedKeyring.from_file(trusted_keys_path)

    operator = proposal.get("operator")
    semantic_version = proposal.get("semantic_version")
    if not isinstance(operator, str) or not isinstance(semantic_version, str):
        raise ConfigError("proposal", "missing operator or semantic version")
    evidence_id = proposal.get("public_evidence", {}).get("evidence_id")
    approval_id = approval["approval_id"]
    if not isinstance(evidence_id, str):
        raise ConfigError("proposal.public_evidence.evidence_id", "expected an identifier")

    public_root = public_root.resolve()
    release_root = release_root.resolve()
    public_root.mkdir(parents=True, exist_ok=True)
    release_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rk-publish-", dir=public_root.parent) as temporary:
        stage = Path(temporary)
        staged_pack = stage / "pack" / pack.pack_id
        _copy_pack_payload(pack, staged_pack)
        manifest_signature_path = staged_pack / "manifest.sig.json"
        manifest_signature = sign_file(
            private_key_path=private_key_path,
            file_path=staged_pack / "manifest.json",
            signed_file="manifest.json",
            output=manifest_signature_path,
        )
        verify_detached_file(
            staged_pack / "manifest.json",
            manifest_signature_path,
            keyring,
            signed_file="manifest.json",
        )
        _validate_schema(manifest_signature, "signature")

        release_name = f"{pack.pack_id}-{pack.pack_version}.tar.gz"
        staged_release = stage / "release" / release_name
        _deterministic_archive(staged_pack, staged_release)

        evidence = dict(proposal["public_evidence"])
        evidence["approval_id"] = approval_id
        evidence["decision"] = "promote"
        compatibility_record = dict(proposal["compatibility_record"])
        compatibility_record["approval_id"] = approval_id
        compatibility_record["status"] = "supported"

        compatibility_path = public_root / "compatibility" / f"{operator}-{semantic_version}.json"
        compatibility_signature_path = compatibility_path.with_suffix(".sig.json")
        compatibility = _load_signed_or_default(
            compatibility_path,
            compatibility_signature_path,
            keyring,
            {"schema_version": "1.0", "records": []},
        )
        records = compatibility.get("records")
        if not isinstance(records, list):
            raise ConfigError(str(compatibility_path), "records must be a list")
        if any(item.get("id") == compatibility_record["id"] for item in records):
            raise ConfigError(str(compatibility_path), "compatibility record already exists")
        records.append(compatibility_record)

        index_path = public_root / "pack-index" / f"{operator}-{semantic_version}.json"
        index_signature_path = index_path.with_suffix(".sig.json")
        index = _load_signed_or_default(
            index_path,
            index_signature_path,
            keyring,
            {
                "schema_version": "1.0",
                "operator": operator,
                "semantic_version": semantic_version,
                "packs": [],
            },
        )
        if index.get("operator") != operator or index.get("semantic_version") != semantic_version:
            raise ConfigError(str(index_path), "index identity does not match proposal")
        packs = index.get("packs")
        if not isinstance(packs, list):
            raise ConfigError(str(index_path), "packs must be a list")
        if any(
            item.get("pack_id") == pack.pack_id and item.get("pack_version") == pack.pack_version
            for item in packs
        ):
            raise ConfigError(str(index_path), "pack version already exists")
        packs.append(
            {
                "pack_id": pack.pack_id,
                "pack_version": pack.pack_version,
                "environment": pack_proposal_environment(proposal),
                "manifest_sha256": sha256_file(staged_pack / "manifest.json"),
                "manifest_signature": manifest_signature,
                "release_file": release_name,
                "release_sha256": sha256_file(staged_release),
                "release_size": staged_release.stat().st_size,
                "artifacts": list(proposal["pack"]["artifacts"]),
                "approval_id": approval_id,
            }
        )
        _validate_schema(evidence, "evidence")
        _validate_schema(compatibility, "compatibility")
        _validate_schema(index, "pack-index")

        staged_documents = {
            stage / "public" / "evidence" / f"{evidence_id}.json": evidence,
            stage / "public" / "approvals" / f"{approval_id}.json": dict(approval),
            stage / "public" / "compatibility" / f"{operator}-{semantic_version}.json": compatibility,
            stage / "public" / "pack-index" / f"{operator}-{semantic_version}.json": index,
        }
        staged_signatures: list[tuple[Path, Path]] = []
        for document_path, document in staged_documents.items():
            _write_json(document_path, document)
            signature_path = document_path.with_suffix(".sig.json")
            signature = sign_file(
                private_key_path=private_key_path,
                file_path=document_path,
                signed_file=document_path.name,
                output=signature_path,
            )
            _validate_schema(signature, "signature")
            verify_detached_file(document_path, signature_path, keyring, signed_file=document_path.name)
            staged_signatures.append((signature_path, signature_path.relative_to(stage / "public")))

        file_moves: list[tuple[Path, Path]] = []
        immutable: set[Path] = set()
        for staged_path in staged_documents:
            relative = staged_path.relative_to(stage / "public")
            destination = public_root / relative
            file_moves.append((staged_path, destination))
            if relative.parts[0] in {"evidence", "approvals"}:
                immutable.add(destination)
        for staged_path, relative in staged_signatures:
            destination = public_root / relative
            file_moves.append((staged_path, destination))
            if relative.parts[0] in {"evidence", "approvals"}:
                immutable.add(destination)
        release_destination = release_root / release_name
        file_moves.append((staged_release, release_destination))
        immutable.add(release_destination)
        _commit_files(file_moves, immutable)

    return {
        "approval_id": approval_id,
        "evidence_id": evidence_id,
        "pack_id": pack.pack_id,
        "pack_version": pack.pack_version,
        "release_file": release_name,
        "release_sha256": sha256_file(release_root / release_name),
    }


def pack_proposal_environment(proposal: Mapping[str, Any]) -> dict[str, Any]:
    pack = proposal.get("pack")
    if not isinstance(pack, Mapping) or not isinstance(pack.get("environment"), Mapping):
        raise ConfigError("proposal.pack.environment", "expected a mapping")
    return dict(pack["environment"])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--proposal", type=Path, required=True)
    result.add_argument("--approval", type=Path, required=True)
    result.add_argument("--pack", type=Path, required=True)
    result.add_argument("--dispatch", type=Path, required=True)
    result.add_argument("--private-key", type=Path, required=True)
    result.add_argument("--trusted-keys", type=Path, required=True)
    result.add_argument("--public-root", type=Path, required=True)
    result.add_argument("--release-root", type=Path, required=True)
    result.add_argument("--allow-override", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    receipt = publish(
        proposal_path=args.proposal,
        approval_path=args.approval,
        pack_path=args.pack,
        dispatch_path=args.dispatch,
        private_key_path=args.private_key,
        trusted_keys_path=args.trusted_keys,
        public_root=args.public_root,
        release_root=args.release_root,
        allow_override=args.allow_override,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
