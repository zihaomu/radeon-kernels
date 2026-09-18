#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from radeon_kernels.ops.gemm.native import (
    VARIANT_NAMES,
    current_architecture,
    load_native_extension,
)


DEFAULT_SHAPES = (
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (256, 4096, 4096),
    (4096, 4096, 11008),
    (4096, 11008, 4096),
)


def parse_shape(value: str) -> tuple[int, int, int]:
    values = tuple(int(item) for item in value.lower().split("x"))
    if len(values) != 3 or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("shape must be MxNxK with positive dimensions")
    return values


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def measure(invoke: Callable[[], object], repetitions: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        invoke()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def repetitions_for(invoke: Callable[[], object], sample_ms: float) -> int:
    elapsed = measure(invoke, 3)
    return max(1, min(10000, math.ceil(sample_ms / elapsed)))


def summarize(samples: list[float], m: int, n: int, k: int) -> dict[str, object]:
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    cv = statistics.pstdev(samples) / mean
    return {
        "samples_ms": samples,
        "median_ms": median,
        "p20_ms": percentile(samples, 0.2),
        "p80_ms": percentile(samples, 0.8),
        "coefficient_of_variation": cv,
        "stable": cv <= 0.03,
        "tflops": 2.0 * m * n * k / (median / 1000.0) / 1.0e12,
    }


def run_case(
    shape: tuple[int, int, int],
    dtype_name: str,
    warmups: int,
    samples: int,
    sample_ms: float,
    variants: list[int],
) -> dict[str, object]:
    m, n, k = shape
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    generator = torch.Generator(device="cuda").manual_seed(20260916)
    a = torch.empty((m, k), device="cuda", dtype=dtype).uniform_(-0.125, 0.125, generator=generator)
    b = torch.empty((k, n), device="cuda", dtype=dtype).uniform_(-0.125, 0.125, generator=generator)
    reference = torch.mm(a, b)
    torch.cuda.synchronize()

    extension = load_native_extension()
    outputs = {"torch_mm": torch.empty_like(reference)}
    callables: dict[str, Callable[[], object]] = {
        "torch_mm": lambda: torch.mm(a, b, out=outputs["torch_mm"])
    }
    correctness: dict[str, dict[str, float | bool]] = {
        "torch_mm": {"correct": True, "max_absolute_error": 0.0}
    }
    failures: list[dict[str, str]] = []
    for variant in variants:
        name = VARIANT_NAMES[variant]
        try:
            output = torch.empty_like(reference)
            extension.gemm_out(a, b, output, variant)
            torch.cuda.synchronize()
            difference = (output.float() - reference.float()).abs()
            torch.testing.assert_close(output, reference, atol=0.1, rtol=0.02)
            outputs[name] = output
            callables[name] = lambda variant=variant, output=output: extension.gemm_out(
                a, b, output, variant
            )
            correctness[name] = {
                "correct": True,
                "max_absolute_error": difference.max().item(),
                "mean_absolute_error": difference.mean().item(),
            }
        except Exception as error:
            failures.append({"candidate": name, "error": f"{type(error).__name__}: {error}"})

    for invoke in callables.values():
        for _ in range(warmups):
            invoke()
    torch.cuda.synchronize()
    repetitions = {
        name: repetitions_for(invoke, sample_ms) for name, invoke in callables.items()
    }
    timings = {name: [] for name in callables}
    order = list(callables)
    rng = random.Random(20260916)
    for _ in range(samples):
        rng.shuffle(order)
        for name in order:
            timings[name].append(measure(callables[name], repetitions[name]))

    implementations = {
        name: {
            **correctness[name],
            "repetitions_per_sample": repetitions[name],
            **summarize(timings[name], m, n, k),
        }
        for name in callables
    }
    winner = min(implementations, key=lambda name: implementations[name]["median_ms"])
    return {
        "dtype": dtype_name,
        "shape": {"m": m, "n": n, "k": k},
        "winner": winner,
        "implementations": implementations,
        "failed_candidates": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), action="append", dest="dtypes")
    parser.add_argument("--shape", type=parse_shape, action="append", dest="shapes")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--sample-ms", type=float, default=20.0)
    parser.add_argument("--variant", type=int, action="append", dest="variants")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    if args.warmups < 1 or args.samples < 2 or args.sample_ms <= 0:
        parser.error("warmups and sample-ms must be positive; samples must be at least two")
    variants = args.variants or list(range(len(VARIANT_NAMES)))
    if any(variant < 0 or variant >= len(VARIANT_NAMES) for variant in variants):
        parser.error(f"variant must be between 0 and {len(VARIANT_NAMES) - 1}")

    extension = load_native_extension(verbose=args.verbose_build)
    result: dict[str, object] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "target_id": args.target_id,
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "architecture": current_architecture(),
            "device_name": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
            "rocm": torch.version.hip,
            "extension_file": extension.__file__,
        },
        "method": {
            "warmups": args.warmups,
            "samples": args.samples,
            "target_sample_ms": args.sample_ms,
            "variants": variants,
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for dtype in args.dtypes or ("fp16", "bf16"):
            for shape in args.shapes or DEFAULT_SHAPES:
                print(f"benchmarking native dtype={dtype} shape={shape}", flush=True)
                result["cases"].append(
                    run_case(
                        shape,
                        dtype,
                        args.warmups,
                        args.samples,
                        args.sample_ms,
                        variants,
                    )
                )
                args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["status"] = "SUCCEEDED"
    except Exception as error:
        result["status"] = "FAILED"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
