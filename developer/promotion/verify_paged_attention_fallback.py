#!/usr/bin/env python3
"""Verify Paged Attention Decode fallback without compilation or search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-architecture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.utils.cpp_extension

    def compilation_forbidden(*_: Any, **__: Any) -> Any:
        raise RuntimeError("compilation is forbidden during Paged Attention verification")

    torch.utils.cpp_extension.load = compilation_forbidden
    import radeon_kernels as rk
    from radeon_kernels.ops.paged_attention.reference import (
        paged_attention_decode_reference,
    )
    from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint

    fingerprint = EnvironmentFingerprint.detect(torch_module=torch)
    if fingerprint.architecture != args.expected_architecture:
        raise RuntimeError(
            f"expected {args.expected_architecture}, detected {fingerprint.architecture}"
        )

    results: list[dict[str, Any]] = []
    for position, dtype in enumerate((torch.float16, torch.bfloat16)):
        generator = torch.Generator(device="cuda").manual_seed(20260918 + position)
        random_tensor = lambda shape: torch.empty(
            shape, device="cuda", dtype=dtype
        ).uniform_(-0.125, 0.125, generator=generator)
        query = random_tensor((2, 8, 64))
        key_cache = random_tensor((8, 16, 2, 64))
        value_cache = random_tensor((8, 16, 2, 64))
        block_tables = torch.tensor([[3, 1, 6, 0], [7, 2, 5, 4]], device="cuda")
        context_lengths = torch.tensor([57, 41], device="cuda")
        expected = paged_attention_decode_reference(
            torch,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
        )
        actual = rk.paged_attention_decode(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        dispatch = rk.get_last_dispatch()
        if dispatch is None or dispatch.get("selected") != "torch/paged_attention":
            raise AssertionError(f"unexpected Paged Attention dispatch: {dispatch!r}")
        results.append(
            {
                "dtype": "fp16" if dtype == torch.float16 else "bf16",
                "selected": dispatch["selected"],
                "max_absolute_error": float((actual.float() - expected.float()).abs().max()),
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
