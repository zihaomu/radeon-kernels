#!/usr/bin/env python3
"""Verify paged KV-cache public fallbacks without compilation or search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _maximum_error(torch: Any, actual: tuple[Any, Any], expected: tuple[Any, Any]) -> float:
    maximum = 0.0
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        maximum = max(maximum, float((left.float() - right.float()).abs().max()))
    return maximum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-architecture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.utils.cpp_extension

    def compilation_forbidden(*_: Any, **__: Any) -> Any:
        raise RuntimeError("compilation is forbidden during KV-cache fallback verification")

    torch.utils.cpp_extension.load = compilation_forbidden
    import radeon_kernels as rk
    from radeon_kernels.ops.kv_cache.reference import (
        kv_cache_append_reference,
        kv_cache_copy_reference,
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
        random_tensor = lambda shape: torch.empty(shape, device="cuda", dtype=dtype).uniform_(
            -0.5, 0.5, generator=generator
        )
        key = random_tensor((8, 4, 64))
        value = random_tensor((8, 4, 64))
        key_cache = random_tensor((16, 16, 4, 64))
        value_cache = random_tensor((16, 16, 4, 64))
        slots = torch.tensor([3, 17, 35, 49, 70, 88, 101, 127], device="cuda")
        expected_append = kv_cache_append_reference(
            torch,
            key,
            value,
            key_cache.clone(),
            value_cache.clone(),
            slots,
        )
        actual_append = rk.kv_cache_append(
            key,
            value,
            key_cache.clone(),
            value_cache.clone(),
            slots,
        )
        torch.cuda.synchronize()
        append_dispatch = rk.get_last_dispatch()
        if append_dispatch is None or append_dispatch.get("selected") != "torch/index_copy":
            raise AssertionError(f"unexpected KV append dispatch: {append_dispatch!r}")

        mapping = torch.tensor([[0, 8], [1, 9], [2, 10], [3, 11]], device="cuda")
        expected_copy = kv_cache_copy_reference(
            torch, key_cache.clone(), value_cache.clone(), mapping
        )
        actual_copy = rk.kv_cache_copy(key_cache.clone(), value_cache.clone(), mapping)
        torch.cuda.synchronize()
        copy_dispatch = rk.get_last_dispatch()
        if copy_dispatch is None or copy_dispatch.get("selected") != "torch/index_copy":
            raise AssertionError(f"unexpected KV copy dispatch: {copy_dispatch!r}")
        results.append(
            {
                "dtype": "fp16" if dtype == torch.float16 else "bf16",
                "append": {
                    "selected": append_dispatch["selected"],
                    "max_absolute_error": _maximum_error(
                        torch, actual_append, expected_append
                    ),
                },
                "copy": {
                    "selected": copy_dispatch["selected"],
                    "max_absolute_error": _maximum_error(torch, actual_copy, expected_copy),
                },
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
    print(json.dumps({"passed": True, "cases": len(results) * 2}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
