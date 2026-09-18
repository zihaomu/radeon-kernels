#!/usr/bin/env python3
"""Verify published paged KV-cache winners and fallbacks without compilation or search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OPERATORS = ("kv_cache_append", "kv_cache_copy")


def _scalar(entry: Any, name: str) -> Any:
    constraint = entry.workload.values[name]
    if constraint.kind != "eq":
        raise RuntimeError(f"acceptance requires exact {entry.id}.{name}")
    return constraint.value


def _random_tensor(torch: Any, shape: tuple[int, ...], dtype: Any, generator: Any) -> Any:
    return torch.empty(shape, device="cuda", dtype=dtype).uniform_(
        -0.5, 0.5, generator=generator
    )


def _assert_exact(torch: Any, actual: tuple[Any, Any], expected: tuple[Any, Any]) -> None:
    for left, right in zip(actual, expected):
        if not torch.equal(left, right):
            maximum = float((left.float() - right.float()).abs().max())
            raise AssertionError(f"KV-cache result differs from reference: max error {maximum}")


def _run_entry(
    torch: Any,
    rk: Any,
    operator: str,
    entry: Any,
    seed: int,
) -> dict[str, Any]:
    from radeon_kernels.ops.kv_cache.reference import (
        kv_cache_append_reference,
        kv_cache_copy_reference,
    )

    dtype = getattr(torch, "float16" if _scalar(entry, "dtype") == "fp16" else "bfloat16")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    num_blocks = int(_scalar(entry, "num_blocks"))
    block_size = int(_scalar(entry, "block_size"))
    kv_heads = int(_scalar(entry, "kv_heads"))
    head_dim = int(_scalar(entry, "head_dim"))
    cache_shape = (num_blocks, block_size, kv_heads, head_dim)
    base_key_cache = _random_tensor(torch, cache_shape, dtype, generator)
    base_value_cache = _random_tensor(torch, cache_shape, dtype, generator)

    if operator == "kv_cache_append":
        tokens = int(_scalar(entry, "tokens"))
        token_shape = (tokens, kv_heads, head_dim)
        key = _random_tensor(torch, token_shape, dtype, generator)
        value = _random_tensor(torch, token_shape, dtype, generator)
        mapping = torch.arange(tokens, device="cuda", dtype=torch.int64)
        expected = kv_cache_append_reference(
            torch,
            key,
            value,
            base_key_cache.clone(),
            base_value_cache.clone(),
            mapping,
        )
        actual_inputs = (base_key_cache.clone(), base_value_cache.clone())
        actual = rk.kv_cache_append(key, value, *actual_inputs, mapping)
    else:
        pairs = int(_scalar(entry, "pairs"))
        mapping = torch.stack(
            (
                torch.arange(pairs, device="cuda", dtype=torch.int64),
                torch.arange(pairs, 2 * pairs, device="cuda", dtype=torch.int64),
            ),
            dim=1,
        )
        expected = kv_cache_copy_reference(
            torch,
            base_key_cache.clone(),
            base_value_cache.clone(),
            mapping,
        )
        actual_inputs = (base_key_cache.clone(), base_value_cache.clone())
        actual = rk.kv_cache_copy(*actual_inputs, mapping)

    torch.cuda.synchronize()
    _assert_exact(torch, actual, expected)
    if any(returned.data_ptr() != supplied.data_ptr() for returned, supplied in zip(actual, actual_inputs)):
        raise AssertionError("KV-cache implementation did not return the input cache tensors")
    dispatch = rk.get_last_dispatch()
    expected_selected = f"native/{entry.winner.entrypoint}"
    if (
        dispatch is None
        or dispatch.get("entry_id") != entry.id
        or dispatch.get("selected") != expected_selected
    ):
        raise AssertionError(f"unexpected KV-cache winner dispatch: {dispatch!r}")
    return {
        "operator": operator,
        "entry_id": entry.id,
        "dtype": _scalar(entry, "dtype"),
        "selected": dispatch["selected"],
        "max_absolute_error": 0.0,
        "returns_input_cache": True,
    }


def _run_fallback(torch: Any, rk: Any, operator: str, seed: int) -> dict[str, Any]:
    from radeon_kernels.ops.kv_cache.reference import (
        kv_cache_append_reference,
        kv_cache_copy_reference,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (8, 16, 4, 64)
    key_cache = _random_tensor(torch, shape, torch.float16, generator)
    value_cache = _random_tensor(torch, shape, torch.float16, generator)
    if operator == "kv_cache_append":
        key = _random_tensor(torch, (2, 4, 64), torch.float16, generator)
        value = _random_tensor(torch, (2, 4, 64), torch.float16, generator)
        mapping = torch.tensor([5, 21], device="cuda", dtype=torch.int64)
        expected = kv_cache_append_reference(
            torch, key, value, key_cache.clone(), value_cache.clone(), mapping
        )
        actual = rk.kv_cache_append(
            key, value, key_cache.clone(), value_cache.clone(), mapping
        )
    else:
        mapping = torch.tensor([[0, 4], [1, 5]], device="cuda", dtype=torch.int64)
        expected = kv_cache_copy_reference(
            torch, key_cache.clone(), value_cache.clone(), mapping
        )
        actual = rk.kv_cache_copy(key_cache.clone(), value_cache.clone(), mapping)
    torch.cuda.synchronize()
    _assert_exact(torch, actual, expected)
    dispatch = rk.get_last_dispatch()
    if dispatch is None or dispatch.get("selected") != "torch/index_copy":
        raise AssertionError(f"unexpected KV-cache fallback dispatch: {dispatch!r}")
    return {
        "operator": operator,
        "selected": dispatch["selected"],
        "max_absolute_error": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-architecture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.utils.cpp_extension

    def compilation_forbidden(*_: Any, **__: Any) -> Any:
        raise RuntimeError("compilation is forbidden during KV-cache release acceptance")

    torch.utils.cpp_extension.load = compilation_forbidden
    import radeon_kernels as rk
    from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
    from radeon_kernels.runtime.resources import builtin_dispatch_registry

    fingerprint = EnvironmentFingerprint.detect(torch_module=torch)
    if fingerprint.architecture != args.expected_architecture:
        raise RuntimeError(
            f"expected {args.expected_architecture}, detected {fingerprint.architecture}"
        )

    registry = builtin_dispatch_registry()
    entries = [
        (operator, entry)
        for operator in OPERATORS
        for entry in registry.get(operator, "1.0").entries
        if entry.environment.matches(fingerprint)
    ]
    if not entries:
        raise RuntimeError(f"no published KV-cache winners for {fingerprint.architecture}")
    winners: list[dict[str, Any]] = []
    for position, (operator, entry) in enumerate(entries):
        winners.append(_run_entry(torch, rk, operator, entry, 20260918 + position))
        torch.cuda.empty_cache()
    fallbacks = [
        _run_fallback(torch, rk, operator, 20261000 + position)
        for position, operator in enumerate(OPERATORS)
    ]
    document = {
        "schema_version": "1.0",
        "environment": fingerprint.to_dict(),
        "compiled_during_verification": False,
        "searched_during_verification": False,
        "winners": winners,
        "fallbacks": fallbacks,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"passed": True, "winners": len(winners), "fallbacks": len(fallbacks)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
