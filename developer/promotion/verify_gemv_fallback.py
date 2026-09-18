#!/usr/bin/env python3
"""Verify the public GEMV API fallback without compilation or search."""

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
        raise RuntimeError("compilation is forbidden during GEMV fallback verification")

    torch.utils.cpp_extension.load = compilation_forbidden
    import radeon_kernels as rk
    from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint

    fingerprint = EnvironmentFingerprint.detect(torch_module=torch)
    if fingerprint.architecture != args.expected_architecture:
        raise RuntimeError(
            f"expected {args.expected_architecture}, detected {fingerprint.architecture}"
        )

    results: list[dict[str, Any]] = []
    for position, dtype in enumerate((torch.float16, torch.bfloat16)):
        generator = torch.Generator(device="cuda").manual_seed(20260917 + position)
        x = torch.empty((2, 128), device="cuda", dtype=dtype).uniform_(
            -0.125, 0.125, generator=generator
        )
        weight = torch.empty((64, 128), device="cuda", dtype=dtype).uniform_(
            -0.125, 0.125, generator=generator
        )
        expected = torch.mm(x.float(), weight.float().transpose(0, 1)).to(dtype=dtype)
        actual = rk.gemv(x, weight)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        dispatch = rk.get_last_dispatch()
        if dispatch is None or dispatch.get("selected") != "torch/mm":
            raise AssertionError(f"unexpected GEMV dispatch: {dispatch!r}")
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
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "cases": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
