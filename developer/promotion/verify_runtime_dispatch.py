#!/usr/bin/env python3
"""Verify public GEMM winner and fallback dispatch without compiling or searching."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


def shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.lower().split("x"))
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("shape must be MxNxK")
    return parts


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--winner-shape", type=shape, required=True)
    result.add_argument("--fallback-shape", type=shape, default=(64, 64, 64))
    result.add_argument("--seed", type=int, default=20260917)
    result.add_argument("--max-absolute-error", type=float, default=0.001)
    result.add_argument("--min-cosine-similarity", type=float, default=0.9999)
    result.add_argument("--output", type=Path, required=True)
    return result


def validate(
    torch: Any,
    rk: Any,
    dimensions: tuple[int, int, int],
    *,
    seed: int,
    expected_provider: str,
    maximum_error: float,
    minimum_cosine: float,
) -> dict[str, Any]:
    m, n, k = dimensions
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.empty((m, k), device="cuda", dtype=torch.float16).uniform_(
        -0.125, 0.125, generator=generator
    )
    b = torch.empty((k, n), device="cuda", dtype=torch.float16).uniform_(
        -0.125, 0.125, generator=generator
    )
    reference = torch.mm(a, b)
    actual = rk.gemm(a, b)
    torch.cuda.synchronize()

    actual_float = actual.float().reshape(-1)
    reference_float = reference.float().reshape(-1)
    finite = bool(
        torch.isfinite(actual_float).all() and torch.isfinite(reference_float).all()
    )
    difference = (actual_float - reference_float).abs()
    denominator = torch.linalg.vector_norm(actual_float) * torch.linalg.vector_norm(reference_float)
    denominator_value = float(denominator.item())
    cosine = -1.0
    if finite and math.isfinite(denominator_value) and denominator_value > 0:
        cosine = float((torch.dot(actual_float, reference_float) / denominator).item())
        cosine = max(-1.0, min(1.0, cosine)) if math.isfinite(cosine) else -1.0
    absolute_error = float(difference.max().item()) if finite else float("inf")
    dispatch = rk.get_last_dispatch()
    selected = dispatch["selected"] if dispatch is not None else None
    passed = bool(
        finite
        and tuple(actual.shape) == (m, n)
        and selected == expected_provider
        and cosine >= minimum_cosine
        and absolute_error <= maximum_error
    )
    return {
        "shape": list(dimensions),
        "expected_provider": expected_provider,
        "selected_provider": selected,
        "finite": finite,
        "cosine_similarity": cosine,
        "max_absolute_error": absolute_error,
        "dispatch": dispatch,
        "passed": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    import torch
    import radeon_kernels as rk

    winner = validate(
        torch,
        rk,
        args.winner_shape,
        seed=args.seed,
        expected_provider="native/gemm_out",
        maximum_error=args.max_absolute_error,
        minimum_cosine=args.min_cosine_similarity,
    )
    fallback = validate(
        torch,
        rk,
        args.fallback_shape,
        seed=args.seed + 1,
        expected_provider="torch/mm",
        maximum_error=0.0,
        minimum_cosine=1.0,
    )
    result = {
        "schema_version": "1.0",
        "compiled_during_verification": False,
        "searched_during_verification": False,
        "winner": winner,
        "fallback": fallback,
        "passed": winner["passed"] and fallback["passed"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
