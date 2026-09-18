#!/usr/bin/env python3
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
ENTRYPOINTS = {
    "rms_norm": "rms_norm",
    "add_rms_norm": "add_rms_norm",
    "rope": "rope",
    "swiglu": "silu_mul",
    "softmax": "softmax",
}


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
    cv = statistics.pstdev(samples) / mean if mean else math.inf
    return {
        "samples_ms": samples,
        "median_ms": median,
        "p20_ms": percentile(samples, 0.2),
        "p80_ms": percentile(samples, 0.8),
        "coefficient_of_variation": cv,
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


def calibrated_repetitions(torch: Any, call: Callable[[], Any]) -> int:
    elapsed = measure(torch, call, 3)
    if elapsed <= 0:
        return 1000
    return max(1, min(1000, math.ceil(10.0 / elapsed)))


def rms_reference(torch: Any, x: Any, weight: Any, eps: float) -> Any:
    source = x.float()
    variance = source.square().mean(dim=-1, keepdim=True)
    return (source * torch.rsqrt(variance + eps) * weight.float()).to(dtype=x.dtype)


def add_rms_reference(torch: Any, x: Any, residual: Any, weight: Any, eps: float) -> tuple[Any, Any]:
    residual_out = x + residual
    return rms_reference(torch, residual_out, weight, eps), residual_out


def rotate_half(torch: Any, value: Any) -> Any:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def rope_reference(torch: Any, q: Any, k: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
    cos_fp32 = cos.float().unsqueeze(1)
    sin_fp32 = sin.float().unsqueeze(1)
    q_fp32 = q.float()
    k_fp32 = k.float()
    return (
        (q_fp32 * cos_fp32 + rotate_half(torch, q_fp32) * sin_fp32).to(dtype=q.dtype),
        (k_fp32 * cos_fp32 + rotate_half(torch, k_fp32) * sin_fp32).to(dtype=k.dtype),
    )


def silu_mul_reference(torch: Any, gate: Any, up: Any) -> Any:
    gate_fp32 = gate.float()
    return (gate_fp32 * torch.sigmoid(gate_fp32) * up.float()).to(dtype=gate.dtype)


def softmax_reference(torch: Any, value: Any) -> Any:
    return torch.softmax(value.float(), dim=-1).to(dtype=value.dtype)


def make_calls(
    torch: Any, extension: Any, operator: str, case: dict[str, Any], dtype: Any, seed: int
) -> tuple[Callable[[], Any], Callable[[], Any]]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    random_tensor = lambda shape: torch.empty(shape, device="cuda", dtype=dtype).uniform_(
        -0.5, 0.5, generator=generator
    )
    if operator == "rms_norm":
        x = random_tensor((case["tokens"], case["hidden"]))
        weight = random_tensor((case["hidden"],))
        eps = float(case["eps"])
        return (
            lambda: extension.rms_norm(x, weight, eps),
            lambda: rms_reference(torch, x, weight, eps),
        )
    if operator == "add_rms_norm":
        x = random_tensor((case["tokens"], case["hidden"]))
        residual = random_tensor((case["tokens"], case["hidden"]))
        weight = random_tensor((case["hidden"],))
        eps = float(case["eps"])
        return (
            lambda: extension.add_rms_norm(x, residual, weight, eps),
            lambda: add_rms_reference(torch, x, residual, weight, eps),
        )
    if operator == "rope":
        q = random_tensor((case["tokens"], case["q_heads"], case["head_dim"]))
        k = random_tensor((case["tokens"], case["kv_heads"], case["head_dim"]))
        angles = torch.empty(
            (case["tokens"], case["head_dim"]), device="cuda", dtype=torch.float32
        ).uniform_(-math.pi, math.pi, generator=generator)
        cos = angles.cos().to(dtype=dtype)
        sin = angles.sin().to(dtype=dtype)
        return (
            lambda: extension.rope(q, k, cos, sin),
            lambda: rope_reference(torch, q, k, cos, sin),
        )
    if operator == "swiglu":
        gate = random_tensor((case["tokens"], case["hidden"]))
        up = random_tensor((case["tokens"], case["hidden"]))
        return (
            lambda: extension.silu_mul(gate, up),
            lambda: silu_mul_reference(torch, gate, up),
        )
    if operator == "softmax":
        value = random_tensor((case["rows"], case["width"]))
        return (
            lambda: extension.softmax(value),
            lambda: softmax_reference(torch, value),
        )
    raise ValueError(f"unknown operator: {operator}")


def result_tensors(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def check_correctness(torch: Any, candidate: Any, baseline: Any, atol: float, rtol: float) -> dict[str, Any]:
    candidate_values = result_tensors(candidate)
    baseline_values = result_tensors(baseline)
    if len(candidate_values) != len(baseline_values):
        raise AssertionError("candidate output count differs from reference")
    maximum_error = 0.0
    for actual, expected in zip(candidate_values, baseline_values):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        maximum_error = max(maximum_error, float((actual.float() - expected.float()).abs().max()))
    return {"passed": True, "max_absolute_error": maximum_error, "outputs": len(candidate_values)}


def evaluate_case(
    torch: Any,
    extension: Any,
    operator: str,
    case: dict[str, Any],
    dtype_name: str,
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    dtype = getattr(torch, DTYPES[dtype_name])
    tolerances = {"softmax": (0.002, 0.002)}
    atol, rtol = tolerances.get(operator, (0.01, 0.01))
    correctness: list[dict[str, Any]] = []
    for seed in (20260917, 20260918, 20260919):
        candidate, baseline = make_calls(torch, extension, operator, case, dtype, seed)
        correctness.append({"seed": seed, **check_correctness(torch, candidate(), baseline(), atol, rtol)})
    candidate, baseline = make_calls(torch, extension, operator, case, dtype, 20260917)
    for _ in range(int(benchmark["warmups"])):
        candidate()
        baseline()
    torch.cuda.synchronize()
    repetitions = max(calibrated_repetitions(torch, candidate), calibrated_repetitions(torch, baseline))
    candidate_samples: list[float] = []
    baseline_samples: list[float] = []
    for index in range(int(benchmark["samples"])):
        order = ((candidate, candidate_samples), (baseline, baseline_samples))
        if index % 2:
            order = tuple(reversed(order))
        for call, samples in order:
            samples.append(measure(torch, call, repetitions))
    candidate_summary = summary(candidate_samples)
    baseline_summary = summary(baseline_samples)
    improvement = (
        (baseline_summary["median_ms"] - candidate_summary["median_ms"])
        / baseline_summary["median_ms"]
        * 100.0
    )
    maximum_cv = float(benchmark["maximum_cv"])
    minimum_improvement = float(benchmark["minimum_improvement_pct"])
    gates = {
        "correctness_passed": all(item["passed"] for item in correctness),
        "candidate_stable": candidate_summary["coefficient_of_variation"] <= maximum_cv,
        "baseline_stable": baseline_summary["coefficient_of_variation"] <= maximum_cv,
        "minimum_improvement": improvement >= minimum_improvement,
    }
    return {
        "case_id": case["id"],
        "operator": operator,
        "dtype": dtype_name,
        "shape": {key: value for key, value in case.items() if key != "id"},
        "correctness": correctness,
        "timing": {
            "repetitions_per_sample": repetitions,
            "candidate": candidate_summary,
            "baseline": baseline_summary,
            "latency_reduction_pct": improvement,
        },
        "gates": gates,
        "decision": "promote" if all(gates.values()) else "negative_knowledge",
    }


def load_extension(torch: Any, source: Path, build_directory: Path, architecture: str, verbose: bool) -> Any:
    from torch.utils.cpp_extension import load

    os.environ["PYTORCH_ROCM_ARCH"] = architecture
    os.environ.setdefault("MAX_JOBS", "2")
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name=f"radeon_kernels_llm_ops_{architecture}",
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
    parser.add_argument("--workloads", type=Path, required=True)
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
    workloads = json.loads(args.workloads.read_text(encoding="utf-8"))
    extension = load_extension(
        torch, args.source.resolve(), args.build_directory.resolve(), architecture, args.verbose_build
    )
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(extension.__file__), args.artifact)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for workload in workloads:
        for case in workload["cases"]:
            for dtype in workload["dtypes"]:
                try:
                    print(f"evaluating {workload['operator']} {case['id']} {dtype}", flush=True)
                    results.append(
                        evaluate_case(
                            torch,
                            extension,
                            workload["operator"],
                            case,
                            dtype,
                            workload["benchmark"],
                        )
                    )
                except Exception as error:
                    failures.append(
                        {
                            "operator": workload["operator"],
                            "case_id": case["id"],
                            "dtype": dtype,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
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
        "candidate": {
            "source_sha256": sha256_file(args.source),
            "artifact_file": args.artifact.name,
            "artifact_sha256": sha256_file(args.artifact),
            "entrypoints": ENTRYPOINTS,
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
