#!/usr/bin/env python3
"""Verify M5 public winner and fallback paths without compilation or search."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable


OPERATORS = ("rms_norm", "add_rms_norm", "rope", "swiglu", "softmax")


def _scalar(entry: Any, name: str) -> Any:
    constraint = entry.workload.values[name]
    if constraint.kind != "eq":
        raise RuntimeError(f"acceptance requires exact {entry.id}.{name}")
    return constraint.value


def _values(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _compare(torch: Any, actual: Any, expected: Any, *, atol: float, rtol: float) -> float:
    actual_values = _values(actual)
    expected_values = _values(expected)
    if len(actual_values) != len(expected_values):
        raise AssertionError("output count differs from reference")
    maximum = 0.0
    for left, right in zip(actual_values, expected_values):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
        maximum = max(maximum, float((left.float() - right.float()).abs().max()))
    return maximum


def _inputs(torch: Any, operator: str, entry: Any, seed: int, *, fallback: bool) -> tuple[tuple[Any, ...], Callable[[], Any]]:
    dtype = getattr(torch, "float16" if _scalar(entry, "dtype") == "fp16" else "bfloat16")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    random_tensor = lambda shape: torch.empty(shape, device="cuda", dtype=dtype).uniform_(
        -0.5, 0.5, generator=generator
    )
    if operator in {"rms_norm", "add_rms_norm", "swiglu"}:
        tokens = int(_scalar(entry, "tokens")) + int(fallback)
        hidden = int(_scalar(entry, "hidden"))
    if operator == "rms_norm":
        from radeon_kernels.ops.rms_norm.reference import rms_norm_reference

        x, weight = random_tensor((tokens, hidden)), random_tensor((hidden,))
        eps = float(_scalar(entry, "eps"))
        return (x, weight, eps), lambda: rms_norm_reference(torch, x, weight, eps)
    if operator == "add_rms_norm":
        from radeon_kernels.ops.add_rms_norm.reference import add_rms_norm_reference

        x = random_tensor((tokens, hidden))
        residual, weight = random_tensor((tokens, hidden)), random_tensor((hidden,))
        eps = float(_scalar(entry, "eps"))
        return (x, residual, weight, eps), lambda: add_rms_norm_reference(
            torch, x, residual, weight, eps
        )
    if operator == "rope":
        from radeon_kernels.ops.rope.reference import rope_reference

        tokens = int(_scalar(entry, "tokens")) + int(fallback)
        q_heads = int(_scalar(entry, "q_heads"))
        kv_heads = int(_scalar(entry, "kv_heads"))
        head_dim = int(_scalar(entry, "head_dim"))
        q = random_tensor((tokens, q_heads, head_dim))
        k = random_tensor((tokens, kv_heads, head_dim))
        angles = torch.empty((tokens, head_dim), device="cuda", dtype=torch.float32).uniform_(
            -math.pi, math.pi, generator=generator
        )
        cos, sin = angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)
        return (q, k, cos, sin), lambda: rope_reference(torch, q, k, cos, sin)
    if operator == "swiglu":
        from radeon_kernels.ops.swiglu.reference import silu_mul_reference

        gate, up = random_tensor((tokens, hidden)), random_tensor((tokens, hidden))
        return (gate, up), lambda: silu_mul_reference(torch, gate, up)
    if operator == "softmax":
        from radeon_kernels.ops.softmax.reference import softmax_reference

        rows = int(_scalar(entry, "rows")) + int(fallback)
        width = int(_scalar(entry, "width"))
        x = random_tensor((rows, width))
        return (x,), lambda: softmax_reference(torch, x)
    raise ValueError(operator)


def _public_call(rk: Any, operator: str, arguments: tuple[Any, ...]) -> Any:
    if operator == "rms_norm":
        return rk.rms_norm(arguments[0], arguments[1], arguments[2])
    if operator == "add_rms_norm":
        return rk.add_rms_norm(arguments[0], arguments[1], arguments[2], arguments[3])
    if operator == "rope":
        return rk.rope(*arguments)
    if operator == "swiglu":
        return rk.silu_mul(*arguments)
    if operator == "softmax":
        return rk.softmax(*arguments)
    raise ValueError(operator)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-architecture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.utils.cpp_extension

    def compilation_forbidden(*_: Any, **__: Any) -> Any:
        raise RuntimeError("compilation is forbidden during release acceptance")

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
    results: list[dict[str, Any]] = []
    for position, operator in enumerate(OPERATORS):
        manifest = registry.get(operator, "1.0")
        entries = [entry for entry in manifest.entries if entry.environment.matches(fingerprint)]
        if len(entries) != 1:
            raise RuntimeError(f"expected one {operator} winner for {fingerprint.architecture}")
        entry = entries[0]
        operator_results: dict[str, Any] = {"operator": operator, "entry_id": entry.id}
        for fallback in (False, True):
            arguments, reference = _inputs(
                torch, operator, entry, 20260917 + position + int(fallback), fallback=fallback
            )
            expected = reference()
            actual = _public_call(rk, operator, arguments)
            torch.cuda.synchronize()
            tolerance = 0.002 if operator == "softmax" else 0.01
            maximum_error = _compare(
                torch, actual, expected, atol=tolerance, rtol=tolerance
            )
            dispatch = rk.get_last_dispatch()
            expected_selected = (
                f"native/{entry.winner.entrypoint}"
                if not fallback
                else f"torch/{'silu_mul' if operator == 'swiglu' else operator}"
            )
            selected = dispatch["selected"] if dispatch else None
            if selected != expected_selected:
                raise AssertionError(
                    f"{operator} selected {selected!r}, expected {expected_selected!r}"
                )
            operator_results["fallback" if fallback else "winner"] = {
                "selected": selected,
                "max_absolute_error": maximum_error,
                "dispatch": dispatch,
                "output_type": type(actual).__name__,
            }
        results.append(operator_results)
    document = {
        "schema_version": "1.0",
        "environment": fingerprint.to_dict(),
        "compiled_during_verification": False,
        "searched_during_verification": False,
        "operators": results,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "operators": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
