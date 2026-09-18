#!/usr/bin/env python3
"""Build an immutable private kernel-pack candidate from a verified artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from radeon_kernels.runtime.pack import KernelPack


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-artifact", type=Path, required=True)
    result.add_argument("--expected-sha256", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--pack-id", required=True)
    result.add_argument("--pack-version", required=True)
    result.add_argument("--architecture", required=True)
    result.add_argument("--wavefront-size", type=int, required=True)
    result.add_argument("--rocm-abi", required=True)
    result.add_argument("--python-abi", required=True)
    result.add_argument("--pytorch-version", required=True)
    result.add_argument("--operator", required=True)
    result.add_argument("--semantic-version", required=True)
    result.add_argument("--provider", required=True)
    result.add_argument("--entrypoint", required=True)
    result.add_argument("--format", choices=("python_extension", "hsaco"), required=True)
    result.add_argument("--evidence-id", required=True)
    return result


def build(args: argparse.Namespace) -> KernelPack:
    source = args.source_artifact.resolve(strict=True)
    if not source.is_file():
        raise ValueError("source artifact must be a file")
    actual_sha256 = sha256(source)
    if actual_sha256 != args.expected_sha256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {args.expected_sha256}, got {actual_sha256}"
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite kernel pack: {output}")

    staging = Path(tempfile.mkdtemp(prefix=f".{args.pack_id}-", dir=output.parent))
    try:
        library_directory = staging / "lib"
        library_directory.mkdir()
        relative_artifact = Path("lib") / source.name
        shutil.copy2(source, staging / relative_artifact)
        copied_sha256 = sha256(staging / relative_artifact)
        if copied_sha256 != actual_sha256:
            raise RuntimeError("artifact changed while copying into the kernel pack")

        manifest = {
            "schema_version": "1.0",
            "pack_id": args.pack_id,
            "pack_version": args.pack_version,
            "environment": {
                "architecture": args.architecture,
                "wavefront_size": args.wavefront_size,
                "rocm_abi": args.rocm_abi,
                "python_abi": args.python_abi,
                "pytorch_version": args.pytorch_version,
            },
            "artifacts": [
                {
                    "operator": args.operator,
                    "semantic_version": args.semantic_version,
                    "provider": args.provider,
                    "artifact": relative_artifact.as_posix(),
                    "entrypoint": args.entrypoint,
                    "sha256": copied_sha256,
                    "format": args.format,
                    "evidence_id": args.evidence_id,
                }
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        KernelPack.from_directory(staging, verify_artifacts=True)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return KernelPack.from_directory(output, verify_artifacts=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    pack = build(args)
    print(
        json.dumps(
            {
                "pack_id": pack.pack_id,
                "pack_version": pack.pack_version,
                "root": str(pack.root),
                "architecture": pack.environment.architecture,
                "rocm_abi": pack.environment.rocm_abi,
                "python_abi": pack.environment.python_abi,
                "pytorch_version": pack.environment.pytorch_version,
                "artifacts": [item.artifact.to_dict() for item in pack.artifacts],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
