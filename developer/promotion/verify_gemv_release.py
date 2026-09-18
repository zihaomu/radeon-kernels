#!/usr/bin/env python3
"""Verify approved GEMV winners and fallback without compilation or search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WINNERS = (
    {
        "id": "decode-ffn-down-h4096-bf16",
        "m": 1,
        "n": 4096,
        "k": 11008,
        "dtype": "bf16",
        "entry_id": "gemv-gfx1201-decode-ffn-down-h4096-bf16",
        "selected": "native/gemv_wave2",
    },
    {
        "id": "decode-h8192-bf16",
        "m": 1,
        "n": 8192,
        "k": 8192,
        "dtype": "bf16",
        "entry_id": "gemv-gfx1201-decode-h8192-bf16",
        "selected": "native/gemv_wave1",
    },
    {
        "id": "decode-h8192-fp16",
        "m": 1,
        "n": 8192,
        "k": 8192,
        "dtype": "fp16",
        "entry_id": "gemv-gfx1201-decode-h8192-fp16",
        "selected": "native/gemv_wave1",
    },
)


def _inputs(torch: Any, case: dict[str, Any], seed: int) -> tuple[Any, Any]:
    dtype = torch.float16 if case["dtype"] == "fp16" else torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.empty((case["m"], case["k"]), device="cuda", dtype=dtype).uniform_(
        -0.125, 0.125, generator=generator
    )
    weight = torch.empty((case["n"], case["k"]), device="cuda", dtype=dtype).uniform_(
        -0.125, 0.125, generator=generator
    )
    return x, weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-architecture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.utils.cpp_extension

    def compilation_forbidden(*_: Any, **__: Any) -> Any:
        raise RuntimeError("compilation is forbidden during GEMV release verification")

    torch.utils.cpp_extension.load = compilation_forbidden
    import radeon_kernels as rk
    from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint

    fingerprint = EnvironmentFingerprint.detect(torch_module=torch)
    if fingerprint.architecture != args.expected_architecture:
        raise RuntimeError(
            f"expected {args.expected_architecture}, detected {fingerprint.architecture}"
        )

    results: list[dict[str, Any]] = []
    for position, case in enumerate(WINNERS):
        x, weight = _inputs(torch, case, 20260918 + position)
        expected = torch.mm(x.float(), weight.float().transpose(0, 1)).to(dtype=x.dtype)
        actual = rk.gemv(x, weight)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        dispatch = rk.get_last_dispatch()
        if (
            dispatch is None
            or dispatch.get("entry_id") != case["entry_id"]
            or dispatch.get("selected") != case["selected"]
        ):
            raise AssertionError(f"unexpected GEMV winner dispatch: {dispatch!r}")
        results.append(
            {
                "case_id": case["id"],
                "dtype": case["dtype"],
                "entry_id": dispatch["entry_id"],
                "selected": dispatch["selected"],
                "max_absolute_error": float((actual.float() - expected.float()).abs().max()),
            }
        )
        del actual, expected, weight, x
        torch.cuda.empty_cache()

    fallback_case = {"m": 2, "n": 64, "k": 128, "dtype": "fp16"}
    x, weight = _inputs(torch, fallback_case, 20260921)
    expected = torch.mm(x.float(), weight.float().transpose(0, 1)).to(dtype=x.dtype)
    actual = rk.gemv(x, weight)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    dispatch = rk.get_last_dispatch()
    if dispatch is None or dispatch.get("selected") != "torch/mm":
        raise AssertionError(f"unexpected GEMV fallback dispatch: {dispatch!r}")
    fallback = {
        "case_id": "unmatched-m2-n64-k128-fp16",
        "selected": dispatch["selected"],
        "max_absolute_error": float((actual.float() - expected.float()).abs().max()),
    }

    document = {
        "schema_version": "1.0",
        "environment": fingerprint.to_dict(),
        "compiled_during_verification": False,
        "searched_during_verification": False,
        "winners": results,
        "fallback": fallback,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "winners": len(results), "fallbacks": 1}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
