#!/usr/bin/env python3
"""Build and evaluate online-softmax SDPA candidates against PyTorch SDPA."""

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

import yaml


DTYPES = {"fp16": "float16", "bf16": "bfloat16"}
ENTRYPOINTS = ("sdpa_wave1", "sdpa_wave4", "sdpa_wave8")
CORRECTNESS_SEEDS = (20260918, 20260919, 20260920)


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


def summary(samples: list[float], operations: int) -> dict[str, Any]:
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    return {
        "samples_ms": samples,
        "median_ms": median,
        "p20_ms": percentile(samples, 0.2),
        "p80_ms": percentile(samples, 0.8),
        "coefficient_of_variation": statistics.pstdev(samples) / mean if mean else math.inf,
        "effective_tflops": operations / (median * 1.0e9),
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


def random_tensor(torch: Any, shape: tuple[int, ...], dtype: Any, generator: Any) -> Any:
    return torch.empty(shape, device="cuda", dtype=dtype).uniform_(
        -0.125, 0.125, generator=generator
    )


def make_inputs(torch: Any, case: dict[str, Any], dtype: Any, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    common = (case["batch"], case["sequence"], case["head_dim"])
    return {
        "query": random_tensor(
            torch,
            (common[0], case["query_heads"], common[1], common[2]),
            dtype,
            generator,
        ),
        "key": random_tensor(
            torch,
            (common[0], case["kv_heads"], common[1], common[2]),
            dtype,
            generator,
        ),
        "value": random_tensor(
            torch,
            (common[0], case["kv_heads"], common[1], common[2]),
            dtype,
            generator,
        ),
        "is_causal": bool(case["causal"]),
        "scale": 1.0 / math.sqrt(case["head_dim"]),
    }


def expanded_kv(torch: Any, query: Any, tensor: Any) -> Any:
    group_size = int(query.shape[1]) // int(tensor.shape[1])
    return tensor if group_size == 1 else tensor.repeat_interleave(group_size, dim=1)


def torch_reference(torch: Any, values: dict[str, Any]) -> Any:
    query = values["query"]
    key = expanded_kv(torch, query, values["key"]).float()
    value = expanded_kv(torch, query, values["value"]).float()
    scores = torch.matmul(query.float(), key.transpose(-2, -1)) * values["scale"]
    if values["is_causal"]:
        sequence = int(query.shape[2])
        mask = torch.ones(
            (sequence, sequence), device="cuda", dtype=torch.bool
        ).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.matmul(probabilities, value).to(query.dtype)


def torch_sdpa(torch: Any, values: dict[str, Any]) -> Any:
    return torch.nn.functional.scaled_dot_product_attention(
        values["query"],
        values["key"],
        values["value"],
        attn_mask=None,
        dropout_p=0.0,
        is_causal=values["is_causal"],
        scale=values["scale"],
        enable_gqa=int(values["query"].shape[1]) != int(values["key"].shape[1]),
    )


def candidate_call(extension: Any, entrypoint: str, values: dict[str, Any]) -> Any:
    return getattr(extension, entrypoint)(
        values["query"],
        values["key"],
        values["value"],
        values["is_causal"],
        values["scale"],
    )


def evaluated_shape(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch": case["batch"],
        "query_heads": case["query_heads"],
        "kv_heads": case["kv_heads"],
        "sequence": case["sequence"],
        "head_dim": case["head_dim"],
        "layout": "batch_head_sequence_dim",
        "attention_mode": "causal" if case["causal"] else "noncausal",
        "scale_mode": "default",
    }


def operation_count(case: dict[str, Any]) -> int:
    sequence = int(case["sequence"])
    pairs = sequence * (sequence + 1) // 2 if case["causal"] else sequence * sequence
    return 4 * int(case["batch"]) * int(case["query_heads"]) * int(case["head_dim"]) * pairs


def correctness_record(actual: Any, expected: Any, seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "passed": True,
        "max_absolute_error": float((actual.float() - expected.float()).abs().max()),
        "outputs": 1,
    }


def evaluate_case(
    torch: Any,
    extension: Any,
    case: dict[str, Any],
    dtype_name: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    dtype = getattr(torch, DTYPES[dtype_name])
    correctness: dict[str, list[dict[str, Any]]] = {
        name: [] for name in (*ENTRYPOINTS, "torch_sdpa")
    }
    last_values: dict[str, Any] | None = None
    tolerance = 0.015 if dtype_name == "fp16" else 0.03
    for seed in CORRECTNESS_SEEDS:
        values = make_inputs(torch, case, dtype, seed)
        expected = torch_reference(torch, values)
        baseline = torch_sdpa(torch, values)
        torch.testing.assert_close(baseline, expected, atol=tolerance, rtol=tolerance)
        correctness["torch_sdpa"].append(correctness_record(baseline, expected, seed))
        for name in ENTRYPOINTS:
            actual = candidate_call(extension, name, values)
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            correctness[name].append(correctness_record(actual, expected, seed))
        last_values = values
    assert last_values is not None

    calls: dict[str, Callable[[], Any]] = {
        name: (lambda name=name: candidate_call(extension, name, last_values))
        for name in ENTRYPOINTS
    }
    calls["torch_sdpa"] = lambda: torch_sdpa(torch, last_values)
    for _ in range(int(policy["warmups"])):
        for call in calls.values():
            call()
    torch.cuda.synchronize()
    repetitions = calibrated_repetitions(torch, list(calls.values()))
    samples = {name: [] for name in calls}
    order = list(calls)
    generator = random.Random(20260918)
    for _ in range(int(policy["samples"])):
        generator.shuffle(order)
        for name in order:
            samples[name].append(measure(torch, calls[name], repetitions))

    operations = operation_count(case)
    timings = {name: summary(values, operations) for name, values in samples.items()}
    maximum_cv = float(policy["maximum_cv"])
    stable_candidates = [
        name
        for name in ENTRYPOINTS
        if timings[name]["coefficient_of_variation"] <= maximum_cv
    ]
    selected = min(stable_candidates or ENTRYPOINTS, key=lambda name: timings[name]["median_ms"])
    candidate = timings[selected]
    baseline = timings["torch_sdpa"]
    improvement = (baseline["median_ms"] - candidate["median_ms"]) / baseline[
        "median_ms"
    ] * 100.0
    gates = {
        "correctness_passed": all(item["passed"] for item in correctness[selected]),
        "baseline_correctness_passed": all(
            item["passed"] for item in correctness["torch_sdpa"]
        ),
        "candidate_stable": candidate["coefficient_of_variation"] <= maximum_cv,
        "baseline_stable": baseline["coefficient_of_variation"] <= maximum_cv,
        "minimum_improvement": improvement >= float(policy["minimum_improvement_pct"]),
    }
    return {
        "case_id": case["id"],
        "operator": "sdpa",
        "dtype": dtype_name,
        "shape": evaluated_shape(case),
        "selected_entrypoint": selected,
        "correctness": correctness[selected],
        "baseline_correctness": correctness["torch_sdpa"],
        "variant_correctness": {name: correctness[name] for name in ENTRYPOINTS},
        "timing": {
            "repetitions_per_sample": repetitions,
            "candidate": candidate,
            "baseline": baseline,
            "variants": {name: timings[name] for name in ENTRYPOINTS},
            "latency_reduction_pct": improvement,
        },
        "gates": gates,
        "decision": "promote" if all(gates.values()) else "negative_knowledge",
    }


def load_extension(
    torch: Any, source: Path, build_directory: Path, architecture: str, verbose: bool
) -> Any:
    from torch.utils.cpp_extension import load

    os.environ["PYTORCH_ROCM_ARCH"] = architecture
    os.environ.setdefault("MAX_JOBS", "2")
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name=f"radeon_kernels_sdpa_{architecture}",
        sources=[str(source)],
        build_directory=str(build_directory),
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
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
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--dtype", choices=tuple(DTYPES), action="append", dest="dtypes")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    import torch

    properties = torch.cuda.get_device_properties(0)
    architecture = getattr(properties, "gcnArchName", "").split(":", 1)[0]
    if architecture != args.expected_architecture:
        raise RuntimeError(f"expected {args.expected_architecture}, detected {architecture}")
    workload = yaml.safe_load(args.workload.read_text(encoding="utf-8"))
    if workload.get("operator") != "sdpa":
        raise ValueError("workload must describe sdpa")
    extension = load_extension(
        torch,
        args.source.resolve(),
        args.build_directory.resolve(),
        architecture,
        args.verbose_build,
    )
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(extension.__file__), args.artifact)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    selected_cases = [
        case
        for case in workload["cases"]
        if args.case_ids is None or case["id"] in args.case_ids
    ]
    if args.case_ids is not None and len(selected_cases) != len(set(args.case_ids)):
        raise ValueError("every requested case must identify one workload case")
    selected_dtypes = args.dtypes or workload["dtypes"]
    if len(set(selected_dtypes)) != len(selected_dtypes):
        raise ValueError("dtypes must not contain duplicates")
    for case in selected_cases:
        for dtype_name in selected_dtypes:
            try:
                print(f"evaluating SDPA {case['id']} {dtype_name}", flush=True)
                results.append(
                    evaluate_case(torch, extension, case, dtype_name, workload["benchmark"])
                )
            except Exception as error:
                failures.append(
                    {
                        "operator": "sdpa",
                        "case_id": case["id"],
                        "dtype": dtype_name,
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
            "oracle": "torch_fp32_explicit_attention",
            "baseline": "torch_scaled_dot_product_attention",
            "layout": "batch_head_sequence_dim",
            "softmax": "online_fp32",
            "entrypoints": list(ENTRYPOINTS),
            "correctness_seeds": list(CORRECTNESS_SEEDS),
        },
        "candidate": {
            "source_sha256": sha256_file(args.source),
            "artifact_file": args.artifact.name,
            "artifact_sha256": sha256_file(args.artifact),
            "entrypoints": {name: name for name in ENTRYPOINTS},
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
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(document["summary"], sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
