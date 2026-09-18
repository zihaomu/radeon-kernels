#!/usr/bin/env python3
"""Probe installed SDPA providers and forced PyTorch attention backends."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable


def _measure(torch: Any, call: Callable[[], Any], repetitions: int = 100) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def _reference(torch: Any, query: Any, key: Any, value: Any, causal: bool) -> Any:
    if query.shape[1] != key.shape[1]:
        group = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(group, dim=1)
        value = value.repeat_interleave(group, dim=1)
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(
        query.shape[-1]
    )
    if causal:
        mask = torch.ones(
            query.shape[2], key.shape[2], device="cuda", dtype=torch.bool
        ).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), value.float()).to(query.dtype)


def _call(torch: Any, query: Any, key: Any, value: Any, causal: bool) -> Any:
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=causal,
        enable_gqa=query.shape[1] != key.shape[1],
    )


def _case(torch: Any, definition: dict[str, Any], backend: Any | None) -> dict[str, Any]:
    from torch.nn.attention import sdpa_kernel

    generator = torch.Generator(device="cuda").manual_seed(20260918)
    dtype = torch.float16
    batch = definition["batch"]
    sequence = definition["sequence"]
    head_dim = definition["head_dim"]
    tensor = lambda heads: torch.empty(
        (batch, heads, sequence, head_dim), device="cuda", dtype=dtype
    ).uniform_(-0.125, 0.125, generator=generator)
    query = tensor(definition["query_heads"])
    key = tensor(definition["kv_heads"])
    value = tensor(definition["kv_heads"])
    expected = _reference(torch, query, key, value, definition["causal"])

    def run() -> Any:
        if backend is None:
            return _call(torch, query, key, value, definition["causal"])
        with sdpa_kernel(backends=[backend]):
            return _call(torch, query, key, value, definition["causal"])

    actual = run()
    torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.015)
    for _ in range(20):
        run()
    torch.cuda.synchronize()
    samples = [_measure(torch, run) for _ in range(20)]
    return {
        "median_ms": statistics.median(samples),
        "coefficient_of_variation": statistics.pstdev(samples)
        / statistics.fmean(samples),
        "max_absolute_error": float((actual.float() - expected.float()).abs().max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-architecture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from torch.nn.attention import SDPBackend

    architecture = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if architecture != args.expected_architecture:
        raise RuntimeError(f"expected {args.expected_architecture}, detected {architecture}")
    packages = {
        name: bool(importlib.util.find_spec(name))
        for name in ("triton", "aiter", "flash_attn", "flash_attn_2_cuda", "xformers")
    }
    definitions = (
        {
            "id": "mha-b1-h16-s128-d64-causal",
            "batch": 1,
            "query_heads": 16,
            "kv_heads": 16,
            "sequence": 128,
            "head_dim": 64,
            "causal": True,
        },
        {
            "id": "gqa-b1-h32x8-s128-d128-causal",
            "batch": 1,
            "query_heads": 32,
            "kv_heads": 8,
            "sequence": 128,
            "head_dim": 128,
            "causal": True,
        },
    )
    backends = {
        "auto": None,
        "flash_attention": SDPBackend.FLASH_ATTENTION,
        "efficient_attention": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    results: dict[str, Any] = {}
    for definition in definitions:
        case_results: dict[str, Any] = {}
        for name, backend in backends.items():
            try:
                case_results[name] = {"available": True, **_case(torch, definition, backend)}
            except Exception as error:
                case_results[name] = {
                    "available": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            finally:
                torch.cuda.empty_cache()
        results[definition["id"]] = case_results
    document = {
        "schema_version": 1,
        "architecture": architecture,
        "torch": str(torch.__version__),
        "rocm": str(torch.version.hip),
        "packages": packages,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
