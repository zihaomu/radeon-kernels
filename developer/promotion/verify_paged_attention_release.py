#!/usr/bin/env python3
"""Verify published Paged Attention winners and fallbacks without compilation or search."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _scalar(entry: Any, name: str) -> Any:
    constraint = entry.workload.values[name]
    if constraint.kind != "eq":
        raise RuntimeError(f"acceptance requires exact {entry.id}.{name}")
    return constraint.value


def _random_tensor(torch: Any, shape: tuple[int, ...], dtype: Any, generator: Any) -> Any:
    return torch.empty(shape, device="cuda", dtype=dtype).uniform_(
        -0.125, 0.125, generator=generator
    )


def _run_entry(torch: Any, rk: Any, entry: Any, seed: int) -> dict[str, Any]:
    from radeon_kernels.ops.paged_attention.reference import (
        paged_attention_decode_reference,
    )

    dtype = getattr(torch, "float16" if _scalar(entry, "dtype") == "fp16" else "bfloat16")
    index_dtype = getattr(torch, _scalar(entry, "index_dtype"))
    batch = int(_scalar(entry, "batch"))
    query_heads = int(_scalar(entry, "query_heads"))
    kv_heads = int(_scalar(entry, "kv_heads"))
    head_dim = int(_scalar(entry, "head_dim"))
    num_blocks = int(_scalar(entry, "num_blocks"))
    block_size = int(_scalar(entry, "block_size"))
    maximum_blocks = int(_scalar(entry, "max_blocks_per_sequence"))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = _random_tensor(torch, (batch, query_heads, head_dim), dtype, generator)
    cache_shape = (num_blocks, block_size, kv_heads, head_dim)
    key_cache = _random_tensor(torch, cache_shape, dtype, generator)
    value_cache = _random_tensor(torch, cache_shape, dtype, generator)
    required_blocks = batch * maximum_blocks
    if required_blocks > num_blocks:
        raise RuntimeError(f"{entry.id} cannot construct disjoint acceptance block tables")
    block_tables = torch.randperm(
        num_blocks, device="cuda", dtype=torch.int64, generator=generator
    )[:required_blocks].to(dtype=index_dtype).view(batch, maximum_blocks).contiguous()
    context_lengths = torch.full(
        (batch,), maximum_blocks * block_size, device="cuda", dtype=index_dtype
    )
    expected = paged_attention_decode_reference(
        torch, query, key_cache, value_cache, block_tables, context_lengths
    )
    actual = rk.paged_attention_decode(
        query, key_cache, value_cache, block_tables, context_lengths
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    dispatch = rk.get_last_dispatch()
    expected_selected = f"native/{entry.winner.entrypoint}"
    if (
        dispatch is None
        or dispatch.get("entry_id") != entry.id
        or dispatch.get("selected") != expected_selected
    ):
        raise AssertionError(f"unexpected Paged Attention winner dispatch: {dispatch!r}")
    return {
        "entry_id": entry.id,
        "dtype": _scalar(entry, "dtype"),
        "selected": dispatch["selected"],
        "max_absolute_error": float((actual.float() - expected.float()).abs().max()),
    }


def _run_fallback(
    torch: Any, rk: Any, dtype: Any, scale: float | None, seed: int
) -> dict[str, Any]:
    from radeon_kernels.ops.paged_attention.reference import (
        paged_attention_decode_reference,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = _random_tensor(torch, (2, 8, 64), dtype, generator)
    key_cache = _random_tensor(torch, (8, 16, 2, 64), dtype, generator)
    value_cache = _random_tensor(torch, (8, 16, 2, 64), dtype, generator)
    block_tables = torch.tensor([[3, 1, 6, 0], [7, 2, 5, 4]], device="cuda")
    context_lengths = torch.tensor([57, 41], device="cuda")
    expected = paged_attention_decode_reference(
        torch, query, key_cache, value_cache, block_tables, context_lengths, scale
    )
    actual = rk.paged_attention_decode(
        query, key_cache, value_cache, block_tables, context_lengths, scale
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    dispatch = rk.get_last_dispatch()
    if dispatch is None or dispatch.get("selected") != "torch/paged_attention":
        raise AssertionError(f"unexpected Paged Attention fallback dispatch: {dispatch!r}")
    return {
        "dtype": "fp16" if dtype == torch.float16 else "bf16",
        "scale_mode": "default" if scale is None else "explicit",
        "selected": dispatch["selected"],
        "max_absolute_error": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-architecture", required=True)
    parser.add_argument("--expected-winners", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.utils.cpp_extension

    def compilation_forbidden(*_: Any, **__: Any) -> Any:
        raise RuntimeError("compilation is forbidden during Paged Attention release acceptance")

    torch.utils.cpp_extension.load = compilation_forbidden
    import radeon_kernels as rk
    from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
    from radeon_kernels.runtime.resources import builtin_dispatch_registry

    fingerprint = EnvironmentFingerprint.detect(torch_module=torch)
    if fingerprint.architecture != args.expected_architecture:
        raise RuntimeError(
            f"expected {args.expected_architecture}, detected {fingerprint.architecture}"
        )
    entries = [
        entry
        for entry in builtin_dispatch_registry().get("paged_attention_decode", "1.0").entries
        if entry.environment.matches(fingerprint)
    ]
    if len(entries) != args.expected_winners:
        raise RuntimeError(
            f"expected {args.expected_winners} winners for {fingerprint.architecture}, "
            f"found {len(entries)}"
        )
    winners: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        winners.append(_run_entry(torch, rk, entry, 20260918 + position))
        torch.cuda.empty_cache()
    fallbacks = [
        _run_fallback(torch, rk, torch.float16, None, 20261000),
        _run_fallback(torch, rk, torch.bfloat16, 1.0 / math.sqrt(64), 20261001),
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
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"passed": True, "winners": len(winners), "fallbacks": len(fallbacks)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
