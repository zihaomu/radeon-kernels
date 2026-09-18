#!/usr/bin/env python3
"""Build and evaluate wave32 GEMV candidates on one Radeon target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DTYPES = {"fp16": "float16", "bf16": "bfloat16"}
ENTRYPOINTS = (
    "gemv_wave1",
    "gemv_wave2",
    "gemv_wave4",
    "gemv_packed2_wave1",
    "gemv_packed2_wave2",
    "gemv_packed4_wave1",
    "gemv_packed4_wave2",
    "gemv_lds_wave1",
    "gemv_lds_wave2",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summary(samples: list[float], case: dict[str, Any]) -> dict[str, Any]:
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    return {
        "samples_ms": samples,
        "median_ms": median,
        "p20_ms": percentile(samples, 0.2),
        "p80_ms": percentile(samples, 0.8),
        "coefficient_of_variation": statistics.pstdev(samples) / mean if mean else math.inf,
        "effective_tflops": 2.0 * case["m"] * case["n"] * case["k"] / (median * 1.0e9),
    }


def measure(torch: Any, call: Callable[[], Any], repetitions: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def calibrated_repetitions(torch: Any, calls: list[Callable[[], Any]]) -> int:
    fastest = min(measure(torch, call, 3) for call in calls)
    if fastest <= 0:
        return 1000
    return max(1, min(1000, math.ceil(10.0 / fastest)))


def inputs(torch: Any, case: dict[str, Any], dtype: Any, seed: int) -> tuple[Any, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.empty((case["m"], case["k"]), device="cuda", dtype=dtype).uniform_(
        -0.125, 0.125, generator=generator
    )
    weight = torch.empty((case["n"], case["k"]), device="cuda", dtype=dtype).uniform_(
        -0.125, 0.125, generator=generator
    )
    return x, weight


def evaluate_case(
    torch: Any,
    extension: Any,
    case: dict[str, Any],
    dtype_name: str,
    policy: dict[str, Any],
    entrypoints: tuple[str, ...],
) -> dict[str, Any]:
    dtype = getattr(torch, DTYPES[dtype_name])
    correctness: dict[str, list[dict[str, Any]]] = {name: [] for name in entrypoints}
    for seed in (20260917, 20260918, 20260919):
        x, weight = inputs(torch, case, dtype, seed)
        expected = torch.mm(x.float(), weight.float().transpose(0, 1)).to(dtype=dtype)
        for name in entrypoints:
            actual = getattr(extension, name)(x, weight)
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
            correctness[name].append(
                {
                    "seed": seed,
                    "passed": True,
                    "max_absolute_error": float((actual.float() - expected.float()).abs().max()),
                    "outputs": 1,
                }
            )

    x, weight = inputs(torch, case, dtype, 20260917)
    transposed_weight = weight.transpose(0, 1)
    calls: dict[str, Callable[[], Any]] = {
        name: (lambda name=name: getattr(extension, name)(x, weight)) for name in entrypoints
    }
    calls["torch_mm"] = lambda: torch.mm(x, transposed_weight)
    for _ in range(int(policy["warmups"])):
        for call in calls.values():
            call()
    torch.cuda.synchronize()
    repetitions = calibrated_repetitions(torch, list(calls.values()))
    samples = {name: [] for name in calls}
    order = list(calls)
    generator = random.Random(20260917)
    for _ in range(int(policy["samples"])):
        generator.shuffle(order)
        for name in order:
            samples[name].append(measure(torch, calls[name], repetitions))

    timings = {name: summary(values, case) for name, values in samples.items()}
    maximum_cv = float(policy["maximum_cv"])
    stable_candidates = [
        name
        for name in entrypoints
        if timings[name]["coefficient_of_variation"] <= maximum_cv
    ]
    selected = min(stable_candidates or entrypoints, key=lambda name: timings[name]["median_ms"])
    candidate = timings[selected]
    baseline = timings["torch_mm"]
    improvement = (baseline["median_ms"] - candidate["median_ms"]) / baseline["median_ms"] * 100.0
    gates = {
        "correctness_passed": all(item["passed"] for item in correctness[selected]),
        "candidate_stable": candidate["coefficient_of_variation"] <= maximum_cv,
        "baseline_stable": baseline["coefficient_of_variation"] <= maximum_cv,
        "minimum_improvement": improvement >= float(policy["minimum_improvement_pct"]),
    }
    return {
        "case_id": case["id"],
        "operator": "gemv",
        "dtype": dtype_name,
        "shape": {"m": case["m"], "n": case["n"], "k": case["k"], "weight_layout": "nk"},
        "selected_entrypoint": selected,
        "correctness": correctness[selected],
        "variant_correctness": correctness,
        "timing": {
            "repetitions_per_sample": repetitions,
            "candidate": candidate,
            "baseline": baseline,
            "variants": {name: timings[name] for name in entrypoints},
            "latency_reduction_pct": improvement,
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
        name=f"radeon_kernels_gemv_{architecture}",
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
    parser.add_argument("--entrypoint", choices=ENTRYPOINTS, action="append", dest="entrypoints")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    import torch

    properties = torch.cuda.get_device_properties(0)
    architecture = getattr(properties, "gcnArchName", "").split(":", 1)[0]
    if architecture != args.expected_architecture:
        raise RuntimeError(f"expected {args.expected_architecture}, detected {architecture}")
    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    if workload.get("operator") != "gemv":
        raise ValueError("workload must describe gemv")
    extension = load_extension(
        torch, args.source.resolve(), args.build_directory.resolve(), architecture, args.verbose_build
    )
    entrypoints = tuple(args.entrypoints or ENTRYPOINTS)
    if len(set(entrypoints)) != len(entrypoints):
        raise ValueError("entrypoints must not contain duplicates")
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(extension.__file__), args.artifact)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for case in workload["cases"]:
        for dtype in workload["dtypes"]:
            try:
                print(f"evaluating GEMV {case['id']} {dtype}", flush=True)
                results.append(
                    evaluate_case(
                        torch,
                        extension,
                        case,
                        dtype,
                        workload["benchmark"],
                        entrypoints,
                    )
                )
            except Exception as error:
                failures.append(
                    {
                        "operator": "gemv",
                        "case_id": case["id"],
                        "dtype": dtype,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            finally:
                torch.cuda.empty_cache()

    document = {
        "schema_version": 1,
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
        "method": {
            "oracle": "torch_fp32_reference",
            "baseline": "torch_mm",
            "weight_layout": "nk",
            "entrypoints": list(entrypoints),
        },
        "candidate": {
            "source_sha256": sha256_file(args.source),
            "artifact_file": args.artifact.name,
            "artifact_sha256": sha256_file(args.artifact),
            "entrypoints": {name: name for name in entrypoints},
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
