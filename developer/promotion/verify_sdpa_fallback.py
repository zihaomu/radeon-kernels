#!/usr/bin/env python3
"""Verify SDPA fallback semantics without compilation or search."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _random_tensor(torch: Any, shape: tuple[int, ...], dtype: Any, generator: Any) -> Any:
    return torch.empty(shape, device="cuda", dtype=dtype).uniform_(
        -0.125, 0.125, generator=generator
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-architecture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.utils.cpp_extension

    def compilation_forbidden(*_: Any, **__: Any) -> Any:
        raise RuntimeError("compilation is forbidden during SDPA fallback verification")

    torch.utils.cpp_extension.load = compilation_forbidden
    import radeon_kernels as rk
    from radeon_kernels.ops.sdpa.reference import sdpa_reference
    from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint

    fingerprint = EnvironmentFingerprint.detect(torch_module=torch)
    if fingerprint.architecture != args.expected_architecture:
        raise RuntimeError(
            f"expected {args.expected_architecture}, detected {fingerprint.architecture}"
        )

    cases = (
        (torch.float16, True, 32, 8, 128, 128),
        (torch.bfloat16, False, 16, 16, 96, 64),
    )
    results: list[dict[str, Any]] = []
    for position, (dtype, causal, query_heads, kv_heads, sequence, head_dim) in enumerate(cases):
        generator = torch.Generator(device="cuda").manual_seed(20260918 + position)
        query = _random_tensor(
            torch, (1, query_heads, sequence, head_dim), dtype, generator
        )
        key = _random_tensor(torch, (1, kv_heads, sequence, head_dim), dtype, generator)
        value = _random_tensor(torch, (1, kv_heads, sequence, head_dim), dtype, generator)
        scale = 1.0 / math.sqrt(head_dim)
        expected = sdpa_reference(torch, query, key, value, causal, scale)
        actual = rk.sdpa(query, key, value, is_causal=causal)
        torch.cuda.synchronize()
        tolerance = 0.015 if dtype == torch.float16 else 0.03
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        dispatch = rk.get_last_dispatch()
        if (
            dispatch is None
            or dispatch.get("selected") != "torch/scaled_dot_product_attention"
        ):
            raise AssertionError(f"unexpected SDPA dispatch: {dispatch!r}")
        results.append(
            {
                "dtype": "fp16" if dtype == torch.float16 else "bf16",
                "attention_mode": "causal" if causal else "noncausal",
                "selected": dispatch["selected"],
                "max_absolute_error": float(
                    (actual.float() - expected.float()).abs().max()
                ),
            }
        )

    document = {
        "schema_version": "1.0",
        "environment": fingerprint.to_dict(),
        "compiled_during_verification": False,
        "searched_during_verification": False,
        "results": results,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": True, "cases": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
