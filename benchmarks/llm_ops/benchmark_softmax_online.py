#!/usr/bin/env python3
"""Evaluate Online Softmax against the released three-pass kernel and PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DTYPES = {"fp16": "float16", "bf16": "bfloat16"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summary(samples: list[float]) -> dict[str, Any]:
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    return {
        "samples_ms": samples,
        "median_ms": median,
        "p20_ms": percentile(samples, 0.2),
        "p80_ms": percentile(samples, 0.8),
        "coefficient_of_variation": statistics.pstdev(samples) / mean if mean else math.inf,
    }


def latency_reduction(candidate_ms: float, baseline_ms: float) -> float:
    return (baseline_ms - candidate_ms) / baseline_ms * 100.0


def measure(torch: Any, call: Callable[[], Any], repetitions: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def calibrated_repetitions(torch: Any, call: Callable[[], Any]) -> int:
    elapsed = measure(torch, call, 3)
    if elapsed <= 0:
        return 1000
    return max(1, min(1000, math.ceil(10.0 / elapsed)))


def check_correctness(torch: Any, actual: Any, expected: Any) -> float:
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)
    return float((actual.float() - expected.float()).abs().max())


def make_calls(torch: Any, extension: Any, case: dict[str, Any], dtype: Any, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    value = torch.empty(
        (case["rows"], case["width"]), device="cuda", dtype=dtype
    ).uniform_(-0.5, 0.5, generator=generator)
    return (
        lambda: extension.softmax(value),
        lambda: extension.softmax_three_pass(value),
        lambda: torch.softmax(value.float(), dim=-1).to(dtype=value.dtype),
    )


def evaluate_case(
    torch: Any,
    extension: Any,
    case: dict[str, Any],
    dtype_name: str,
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    dtype = getattr(torch, DTYPES[dtype_name])
    correctness: list[dict[str, Any]] = []
    for seed in (20260917, 20260918, 20260919):
        candidate, legacy, baseline = make_calls(torch, extension, case, dtype, seed)
        expected = baseline()
        correctness.append(
            {
                "seed": seed,
                "passed": True,
                "max_absolute_error": check_correctness(torch, candidate(), expected),
                "legacy_max_absolute_error": check_correctness(torch, legacy(), expected),
                "outputs": 1,
            }
        )

    candidate, legacy, baseline = make_calls(torch, extension, case, dtype, 20260917)
    calls = (candidate, legacy, baseline)
    for _ in range(int(benchmark["warmups"])):
        for call in calls:
            call()
    torch.cuda.synchronize()
    repetitions = max(calibrated_repetitions(torch, call) for call in calls)
    samples: list[list[float]] = [[], [], []]
    for index in range(int(benchmark["samples"])):
        order = (0, 1, 2) if index % 2 == 0 else (2, 1, 0)
        for call_index in order:
            samples[call_index].append(measure(torch, calls[call_index], repetitions))

    candidate_summary, legacy_summary, baseline_summary = map(summary, samples)
    versus_baseline = latency_reduction(
        candidate_summary["median_ms"], baseline_summary["median_ms"]
    )
    versus_legacy = latency_reduction(
        candidate_summary["median_ms"], legacy_summary["median_ms"]
    )
    maximum_cv = float(benchmark["maximum_cv"])
    minimum_improvement = float(benchmark["minimum_improvement_pct"])
    gates = {
        "correctness_passed": all(item["passed"] for item in correctness),
        "candidate_stable": candidate_summary["coefficient_of_variation"] <= maximum_cv,
        "legacy_stable": legacy_summary["coefficient_of_variation"] <= maximum_cv,
        "baseline_stable": baseline_summary["coefficient_of_variation"] <= maximum_cv,
        "minimum_improvement": versus_baseline >= minimum_improvement,
        "online_beats_three_pass": versus_legacy >= minimum_improvement,
    }
    return {
        "case_id": case["id"],
        "operator": "softmax",
        "dtype": dtype_name,
        "shape": {key: value for key, value in case.items() if key != "id"},
        "correctness": correctness,
        "timing": {
            "repetitions_per_sample": repetitions,
            "candidate": candidate_summary,
            "legacy_three_pass": legacy_summary,
            "baseline": baseline_summary,
            "latency_reduction_pct": versus_baseline,
            "latency_reduction_vs_three_pass_pct": versus_legacy,
        },
        "gates": gates,
        "decision": "promote" if all(gates.values()) else "negative_knowledge",
    }


def load_extension(torch: Any, source: Path, build_directory: Path, architecture: str, verbose: bool):
    from torch.utils.cpp_extension import load

    os.environ["PYTORCH_ROCM_ARCH"] = architecture
    os.environ.setdefault("MAX_JOBS", "2")
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name=f"radeon_kernels_softmax_online_{architecture}",
        sources=[str(source)],
        build_directory=str(build_directory),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-ffast-math",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
        ],
        with_cuda=True,
        verbose=verbose,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-architecture", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    import torch

    properties = torch.cuda.get_device_properties(0)
    architecture = getattr(properties, "gcnArchName", "").split(":", 1)[0]
    if architecture != args.expected_architecture:
        raise RuntimeError(f"expected {args.expected_architecture}, detected {architecture}")
    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    if workload.get("operator") != "softmax":
        raise ValueError("the workload must describe softmax")
    extension = load_extension(
        torch, args.source.resolve(), args.build_directory.resolve(), architecture, args.verbose_build
    )
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(extension.__file__), args.artifact)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for case in workload["cases"]:
        for dtype in workload["dtypes"]:
            try:
                print(f"evaluating Online Softmax {case['id']} {dtype}", flush=True)
                results.append(evaluate_case(torch, extension, case, dtype, workload["benchmark"]))
            except Exception as error:
                failures.append(
                    {
                        "operator": "softmax",
                        "case_id": case["id"],
                        "dtype": dtype,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    document = {
        "schema_version": 1,
        "experiment": "online_softmax_vs_three_pass",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "target_id": args.target_id,
        "environment": {
            "architecture": architecture,
            "wavefront_size": int(properties.warp_size),
            "device_name": torch.cuda.get_device_name(0),
            "rocm": str(torch.version.hip),
            "pytorch": str(torch.__version__),
            "python": platform.python_version(),
        },
        "candidate": {
            "algorithm": "online_softmax",
            "source_sha256": sha256_file(args.source),
            "artifact_file": args.artifact.name,
            "artifact_sha256": sha256_file(args.artifact),
            "entrypoints": {"softmax": "softmax", "legacy": "softmax_three_pass"},
        },
        "results": results,
        "failures": failures,
        "summary": {
            "total_cases": len(results) + len(failures),
            "successful_cases": len(results),
            "promotable_cases": sum(item["decision"] == "promote" for item in results),
            "failed_cases": len(failures),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(document["summary"], sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
