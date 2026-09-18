#!/usr/bin/env python3
"""Reproducible single-GPU GEMM comparison for ROCm PyTorch and Triton."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import triton
import triton.language as tl


DEFAULT_SHAPES = (
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (256, 4096, 4096),
    (4096, 4096, 11008),
    (4096, 11008, 4096),
)


@dataclass(frozen=True)
class Candidate:
    name: str
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


CANDIDATES = (
    Candidate("triton_bm64_bn64_bk32_w4_s2", 64, 64, 32, 4, 2),
    Candidate("triton_bm64_bn128_bk32_w4_s2", 64, 128, 32, 4, 2),
    Candidate("triton_bm128_bn64_bk32_w4_s2", 128, 64, 32, 4, 2),
    Candidate("triton_bm128_bn128_bk32_w8_s2", 128, 128, 32, 8, 2),
    Candidate("triton_bm128_bn128_bk64_w8_s3", 128, 128, 64, 8, 3),
)


@triton.jit
def gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    a_tiles = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_tiles = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for tile_k in range(0, tl.cdiv(k, BLOCK_K)):
        a = tl.load(
            a_tiles,
            mask=(offsets_m[:, None] < m) & (offsets_k[None, :] + tile_k * BLOCK_K < k),
            other=0.0,
        )
        b = tl.load(
            b_tiles,
            mask=(offsets_k[:, None] + tile_k * BLOCK_K < k) & (offsets_n[None, :] < n),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_tiles += BLOCK_K * stride_ak
        b_tiles += BLOCK_K * stride_bk

    c_tiles = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(c_tiles, accumulator, mask=(offsets_m[:, None] < m) & (offsets_n[None, :] < n))


def parse_shape(value: str) -> tuple[int, int, int]:
    try:
        dimensions = tuple(int(item) for item in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be MxNxK") from error
    if len(dimensions) != 3 or any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError("shape must contain three positive dimensions")
    return dimensions


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def device_environment() -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "rocm": getattr(torch.version, "hip", None),
        "device_name": torch.cuda.get_device_name(0),
        "architecture": getattr(properties, "gcnArchName", None),
        "visible_device_count": torch.cuda.device_count(),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
    }


def triton_callable(
    a: torch.Tensor, b: torch.Tensor, output: torch.Tensor, candidate: Candidate
) -> Callable[[], None]:
    m, k = a.shape
    _, n = b.shape
    grid = (triton.cdiv(m, candidate.block_m) * triton.cdiv(n, candidate.block_n),)

    def invoke() -> None:
        gemm_kernel[grid](
            a,
            b,
            output,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_M=candidate.block_m,
            BLOCK_N=candidate.block_n,
            BLOCK_K=candidate.block_k,
            GROUP_M=8,
            num_warps=candidate.num_warps,
            num_stages=candidate.num_stages,
        )

    return invoke


def measure_once(invoke: Callable[[], None], repetitions: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        invoke()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def calibrated_repetitions(invoke: Callable[[], None]) -> int:
    elapsed_ms = measure_once(invoke, 3)
    if elapsed_ms <= 0:
        return 2000
    return max(1, min(2000, math.ceil(20.0 / elapsed_ms)))


def summarize(samples_ms: list[float], m: int, n: int, k: int) -> dict[str, object]:
    median_ms = statistics.median(samples_ms)
    mean_ms = statistics.fmean(samples_ms)
    coefficient_of_variation = statistics.pstdev(samples_ms) / mean_ms if mean_ms else math.inf
    return {
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "p20_ms": percentile(samples_ms, 0.2),
        "p80_ms": percentile(samples_ms, 0.8),
        "coefficient_of_variation": coefficient_of_variation,
        "tflops": 2.0 * m * n * k / (median_ms / 1000.0) / 1.0e12,
        "stable": coefficient_of_variation <= 0.03,
    }


def benchmark_case(
    m: int,
    n: int,
    k: int,
    dtype_name: str,
    warmups: int,
    samples: int,
    seed: int,
) -> dict[str, object]:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.empty((m, k), device="cuda", dtype=dtype).uniform_(-0.125, 0.125, generator=generator)
    b = torch.empty((k, n), device="cuda", dtype=dtype).uniform_(-0.125, 0.125, generator=generator)

    reference = torch.empty((m, n), device="cuda", dtype=dtype)
    torch.mm(a, b, out=reference)
    torch.cuda.synchronize()

    outputs: dict[str, torch.Tensor] = {"torch_mm": torch.empty_like(reference)}
    callables: dict[str, Callable[[], None]] = {
        "torch_mm": lambda: torch.mm(a, b, out=outputs["torch_mm"])
    }
    definitions: dict[str, object] = {"torch_mm": {"kind": "pytorch_rocm"}}
    failures: list[dict[str, str]] = []

    for candidate in CANDIDATES:
        output = torch.empty_like(reference)
        invoke = triton_callable(a, b, output, candidate)
        try:
            invoke()
            torch.cuda.synchronize()
            difference = (output.float() - reference.float()).abs()
            torch.testing.assert_close(output, reference, atol=0.1, rtol=0.02)
            outputs[candidate.name] = output
            callables[candidate.name] = invoke
            definitions[candidate.name] = {"kind": "triton", **asdict(candidate)}
            definitions[candidate.name]["max_absolute_error"] = difference.max().item()
            definitions[candidate.name]["mean_absolute_error"] = difference.mean().item()
        except Exception as error:  # A failed candidate must not abort the experiment matrix.
            failures.append({"candidate": candidate.name, "error": f"{type(error).__name__}: {error}"})

    for invoke in callables.values():
        for _ in range(warmups):
            invoke()
    torch.cuda.synchronize()

    repetitions: dict[str, int] = {}
    for name, invoke in callables.items():
        repetitions[name] = calibrated_repetitions(invoke)

    timings: dict[str, list[float]] = {name: [] for name in callables}
    order_rng = random.Random(seed)
    names = list(callables)
    for _ in range(samples):
        order_rng.shuffle(names)
        for name in names:
            timings[name].append(measure_once(callables[name], repetitions[name]))

    implementations: dict[str, dict[str, object]] = {}
    for name in callables:
        implementations[name] = {
            **definitions[name],
            "correct": True,
            "repetitions_per_sample": repetitions[name],
            **summarize(timings[name], m, n, k),
        }

    winner = min(implementations, key=lambda name: implementations[name]["median_ms"])
    baseline_ms = implementations["torch_mm"]["median_ms"]
    winner_ms = implementations[winner]["median_ms"]
    return {
        "shape": {"m": m, "n": n, "k": k},
        "dtype": dtype_name,
        "layout": "nn",
        "seed": seed,
        "winner": winner,
        "winner_speedup_over_torch": baseline_ms / winner_ms,
        "implementations": implementations,
        "failed_candidates": failures,
    }


def save_result(path: Path, result: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), action="append", dest="dtypes")
    parser.add_argument("--shape", type=parse_shape, action="append", dest="shapes")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("ROCm device is not available through torch.cuda")
    if args.warmups < 1 or args.samples < 2:
        parser.error("warmups must be positive and samples must be at least two")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "target_id": args.target_id,
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": device_environment(),
        "method": {
            "warmups": args.warmups,
            "samples": args.samples,
            "sample_order": "deterministically shuffled per round",
            "timing": "torch.cuda.Event; calibrated repeated launches per sample",
            "correctness_reference": "torch.mm",
            "absolute_tolerance": 0.1,
            "relative_tolerance": 0.02,
            "maximum_cv": 0.03,
        },
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "cases": [],
    }
    save_result(args.output, result)

    dtypes = args.dtypes or ["fp16", "bf16"]
    shapes = args.shapes or list(DEFAULT_SHAPES)
    try:
        for dtype_index, dtype_name in enumerate(dtypes):
            for shape_index, (m, n, k) in enumerate(shapes):
                print(f"benchmarking dtype={dtype_name} shape={m}x{n}x{k}", flush=True)
                result["cases"].append(
                    benchmark_case(
                        m,
                        n,
                        k,
                        dtype_name,
                        args.warmups,
                        args.samples,
                        seed=20260916 + dtype_index * 1000 + shape_index,
                    )
                )
                save_result(args.output, result)
    except Exception as error:
        result["status"] = "FAILED"
        result["error"] = f"{type(error).__name__}: {error}"
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        save_result(args.output, result)
        raise

    result["status"] = "SUCCEEDED"
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_result(args.output, result)
    print(json.dumps({"status": result["status"], "case_count": len(result["cases"])}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
