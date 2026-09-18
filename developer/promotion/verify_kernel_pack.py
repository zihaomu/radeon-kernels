#!/usr/bin/env python3
"""Verify a precompiled GEMM kernel pack without compiling or searching."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
from radeon_kernels.runtime.loader import PrecompiledArtifactLoader
from radeon_kernels.runtime.pack import KernelPack


def shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.lower().split("x"))
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("shape must be MxNxK")
    return parts


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pack", type=Path, required=True)
    result.add_argument("--shape", type=shape, required=True)
    result.add_argument("--variant", type=int, required=True)
    result.add_argument("--seed", type=int, action="append", dest="seeds", required=True)
    result.add_argument("--max-absolute-error", type=float, default=0.001)
    result.add_argument("--min-cosine-similarity", type=float, default=0.9999)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    import torch

    pack = KernelPack.from_directory(args.pack, verify_artifacts=True)
    fingerprint = EnvironmentFingerprint.detect(torch_module=torch)
    if not pack.matches(fingerprint):
        raise RuntimeError(
            f"kernel pack {pack.pack_id!r} does not match runtime {fingerprint.to_dict()}"
        )

    item = pack.get("gemm", "1.0")
    loaded = PrecompiledArtifactLoader(pack.root).load_entrypoint(item.artifact)
    m, n, k = args.shape
    cases = []
    for seed in args.seeds:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        a = torch.empty((m, k), device="cuda", dtype=torch.float16).uniform_(
            -0.125, 0.125, generator=generator
        )
        b = torch.empty((k, n), device="cuda", dtype=torch.float16).uniform_(
            -0.125, 0.125, generator=generator
        )
        output = torch.empty((m, n), device="cuda", dtype=torch.float16)
        reference = torch.mm(a, b)
        returned = loaded.callable(a, b, output, args.variant)
        torch.cuda.synchronize()

        output_float = output.float().reshape(-1)
        reference_float = reference.float().reshape(-1)
        finite = bool(
            torch.isfinite(output_float).all() and torch.isfinite(reference_float).all()
        )
        difference = (output_float - reference_float).abs()
        denominator = torch.linalg.vector_norm(output_float) * torch.linalg.vector_norm(reference_float)
        denominator_value = float(denominator.item())
        cosine = -1.0
        if finite and math.isfinite(denominator_value) and denominator_value > 0:
            cosine = float((torch.dot(output_float, reference_float) / denominator).item())
            cosine = max(-1.0, min(1.0, cosine)) if math.isfinite(cosine) else -1.0
        maximum = float(difference.max().item()) if finite else float("inf")
        passed = bool(
            finite
            and returned.data_ptr() == output.data_ptr()
            and cosine >= args.min_cosine_similarity
            and maximum <= args.max_absolute_error
        )
        cases.append(
            {
                "seed": seed,
                "passed": passed,
                "finite": finite,
                "same_output_storage": returned.data_ptr() == output.data_ptr(),
                "cosine_similarity": cosine,
                "max_absolute_error": maximum,
            }
        )

    result = {
        "schema_version": "1.0",
        "pack_id": pack.pack_id,
        "pack_version": pack.pack_version,
        "fingerprint": fingerprint.to_dict(),
        "artifact": item.artifact.to_dict(),
        "evidence_id": item.evidence_id,
        "shape": list(args.shape),
        "variant": args.variant,
        "compiled_during_verification": False,
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
