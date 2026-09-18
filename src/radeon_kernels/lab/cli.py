from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from radeon_kernels.errors import RadeonKernelsError
from radeon_kernels.lab.config import load_targets
from radeon_kernels.lab.planner import build_plan
from radeon_kernels.lab.remote import RemoteStatus, probe_targets
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
from radeon_kernels.runtime.installer import (
    PackInstallError,
    compatible_records,
    fetch_release,
    index_records,
    install_release,
    verify_installed_pack,
)
from radeon_kernels.runtime.packs import default_pack_roots
from radeon_kernels.runtime.resources import builtin_pack_index, builtin_pack_indexes
from radeon_kernels.workspace import load_workspace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rk")
    parser.add_argument(
        "--workspace",
        type=Path,
        help="workspace directory or .radeon-workspace.yaml path",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="validate local workspace configuration")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    doctor.add_argument(
        "--remote",
        action="store_true",
        help="also probe SSH, architecture, runtime, image, and workspace readiness",
    )
    doctor.add_argument(
        "--target",
        action="append",
        default=[],
        dest="targets",
        help="select a configured target; may be repeated",
    )
    doctor.add_argument(
        "--accept-new-host-keys",
        action="store_true",
        help="trust first-seen SSH keys in the private workspace known_hosts file",
    )

    plan = commands.add_parser("plan", help="print the target execution plan")
    plan.add_argument(
        "--target",
        action="append",
        default=[],
        dest="targets",
        help="select a configured target; may be repeated",
    )

    pack = commands.add_parser("pack", help="list, install, and verify signed kernel packs")
    pack_commands = pack.add_subparsers(dest="pack_command", required=True)

    pack_list = pack_commands.add_parser("list", help="list signed public pack releases")
    pack_list.add_argument(
        "--compatible",
        action="store_true",
        help="show only releases compatible with the local ROCm environment",
    )

    pack_install = pack_commands.add_parser("install", help="install a signed kernel pack")
    pack_install.add_argument("--operator", required=True)
    pack_install.add_argument("--semantic-version", required=True)
    pack_install.add_argument("--release-file", help="select a release when several are compatible")
    source = pack_install.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="path to a downloaded release archive")
    source.add_argument("--base-url", help="base URL containing indexed release archives")
    pack_install.add_argument(
        "--destination",
        type=Path,
        help="pack directory; defaults to the first runtime pack root",
    )

    pack_verify = pack_commands.add_parser("verify", help="verify an installed signed pack")
    pack_verify.add_argument("path", type=Path)
    return parser


def _doctor(
    workspace_arg: Path | None,
    as_json: bool,
    remote: bool,
    selected_targets: list[str],
    accept_new_host_keys: bool,
) -> int:
    if accept_new_host_keys and not remote:
        raise RadeonKernelsError("--accept-new-host-keys requires --remote")
    workspace = load_workspace(workspace_arg)
    config = load_targets(workspace.targets_file)
    plan = build_plan(config, selected_targets)
    planned_ids = {target.target_id for target in plan.targets}
    selected = tuple(target for target in config.targets if target.id in planned_ids)
    payload = {
        "status": "valid",
        "workspace": str(workspace.root),
        "targets_file": str(workspace.targets_file),
        "target_count": len(selected),
        "targets": [
            {
                "id": target.id,
                "ssh_host": target.ssh_host,
                "expected_architecture": target.expected_architecture,
                "gpu_ids": list(target.gpu_ids),
            }
            for target in selected
        ],
    }
    exit_code = 0
    if remote:
        if accept_new_host_keys:
            workspace.state_dir.mkdir(parents=True, exist_ok=True)
        probes = probe_targets(
            selected,
            known_hosts_file=workspace.state_dir / "known_hosts",
            accept_new_host_keys=accept_new_host_keys,
        )
        payload["remote"] = [probe.to_dict() for probe in probes]
        if any(probe.status is not RemoteStatus.READY for probe in probes):
            payload["status"] = "invalid"
            exit_code = 1
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"workspace: {payload['workspace']}")
        print(f"targets: {payload['target_count']} ({payload['targets_file']})")
        for target in selected:
            gpu_list = ",".join(str(gpu_id) for gpu_id in target.gpu_ids)
            print(
                f"  {target.id}: ssh={target.ssh_host} "
                f"arch={target.expected_architecture} gpus={gpu_list}"
            )
        if remote:
            for probe in probes:
                print(f"  remote {probe.target_id}: {probe.status.value} - {probe.message}")
    return exit_code


def _plan(workspace_arg: Path | None, targets: list[str]) -> int:
    workspace = load_workspace(workspace_arg)
    config = load_targets(workspace.targets_file)
    plan = build_plan(config, targets)
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


def _record_payload(operator: str, semantic_version: str, record: object) -> dict[str, object]:
    return {
        "operator": operator,
        "semantic_version": semantic_version,
        "pack_id": record.pack_id,
        "pack_version": record.pack_version,
        "release_file": record.release_file,
        "release_sha256": record.release_sha256,
        "release_size": record.release_size,
        "approval_id": record.approval_id,
        "environment": record.environment.to_dict(),
    }


def _pack_list(only_compatible: bool) -> int:
    fingerprint = EnvironmentFingerprint.detect() if only_compatible else None
    releases: list[dict[str, object]] = []
    for index in builtin_pack_indexes():
        operator = str(index["operator"])
        semantic_version = str(index["semantic_version"])
        records = compatible_records(index, fingerprint) if fingerprint else index_records(index)
        releases.extend(
            _record_payload(operator, semantic_version, record) for record in records
        )
    print(json.dumps({"packs": releases}, indent=2, sort_keys=True))
    return 0


def _selected_release(index: object, fingerprint: EnvironmentFingerprint, name: str | None):
    records = compatible_records(index, fingerprint)
    if name is not None:
        records = tuple(record for record in records if record.release_file == name)
    if not records:
        raise PackInstallError("the signed index has no compatible release")
    if len(records) != 1:
        choices = ", ".join(record.release_file for record in records)
        raise PackInstallError(f"multiple compatible releases found; use --release-file: {choices}")
    return records[0]


def _pack_install(args: argparse.Namespace) -> int:
    index = builtin_pack_index(args.operator, args.semantic_version)
    fingerprint = EnvironmentFingerprint.detect()
    record = _selected_release(index, fingerprint, args.release_file)
    destination = args.destination or default_pack_roots()[0]
    if args.archive is not None:
        pack = install_release(
            args.archive,
            index,
            fingerprint,
            destination,
            release_file=record.release_file,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="rk-download-") as temporary:
            archive = fetch_release(record, args.base_url, Path(temporary) / record.release_file)
            pack = install_release(
                archive,
                index,
                fingerprint,
                destination,
                release_file=record.release_file,
            )
    print(
        json.dumps(
            {
                "status": "installed",
                "pack_id": pack.pack_id,
                "pack_version": pack.pack_version,
                "path": str(pack.root),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _pack_verify(path: Path) -> int:
    pack = verify_installed_pack(path)
    print(
        json.dumps(
            {
                "status": "valid",
                "pack_id": pack.pack_id,
                "pack_version": pack.pack_version,
                "path": str(pack.root),
                "artifacts": len(pack.artifacts),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return _doctor(
                args.workspace,
                args.as_json,
                args.remote,
                args.targets,
                args.accept_new_host_keys,
            )
        if args.command == "plan":
            return _plan(args.workspace, args.targets)
        if args.command == "pack":
            if args.pack_command == "list":
                return _pack_list(args.compatible)
            if args.pack_command == "install":
                return _pack_install(args)
            if args.pack_command == "verify":
                return _pack_verify(args.path)
        raise AssertionError(f"unhandled command: {args.command}")
    except RadeonKernelsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
