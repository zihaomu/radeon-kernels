#!/usr/bin/env python3
"""Build and evaluate paged KV-cache append/copy candidates on one Radeon target."""

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
ENTRYPOINTS = {
    "kv_cache_append": ("append_scalar", "append_vec4", "append_vec8"),
    "kv_cache_copy": ("copy_scalar", "copy_vec4", "copy_vec8"),
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
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summary(samples: list[float], bytes_moved: int) -> dict[str, Any]:
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    return {
        "samples_ms": samples,
        "median_ms": median,
        "p20_ms": percentile(samples, 0.2),
        "p80_ms": percentile(samples, 0.8),
        "coefficient_of_variation": statistics.pstdev(samples) / mean if mean else math.inf,
        "effective_gbps": bytes_moved / (median * 1.0e6),
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
        -0.5, 0.5, generator=generator
    )


def make_inputs(
    torch: Any,
    operator: str,
    case: dict[str, Any],
    dtype: Any,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    cache_shape = (
        case["num_blocks"],
        case["block_size"],
        case["kv_heads"],
        case["head_dim"],
    )
    inputs: dict[str, Any] = {
        "key_cache": random_tensor(torch, cache_shape, dtype, generator),
        "value_cache": random_tensor(torch, cache_shape, dtype, generator),
    }
    if operator == "kv_cache_append":
        token_shape = (case["tokens"], case["kv_heads"], case["head_dim"])
        inputs.update(
            {
                "key": random_tensor(torch, token_shape, dtype, generator),
                "value": random_tensor(torch, token_shape, dtype, generator),
                "mapping": torch.randperm(
                    case["num_blocks"] * case["block_size"],
                    device="cuda",
                    generator=generator,
                    dtype=torch.int64,
                )[: case["tokens"]].contiguous(),
            }
        )
    else:
        pairs = case["pairs"]
        inputs["mapping"] = torch.stack(
            (
                torch.arange(pairs, device="cuda", dtype=torch.int64),
                torch.arange(pairs, 2 * pairs, device="cuda", dtype=torch.int64),
            ),
            dim=1,
        ).contiguous()
    return inputs


def append_reference(torch: Any, values: dict[str, Any]) -> tuple[Any, Any]:
    slots = values["mapping"]
    cache_shape = (-1, values["key_cache"].shape[2], values["key_cache"].shape[3])
    values["key_cache"].view(cache_shape).index_copy_(0, slots, values["key"])
    values["value_cache"].view(cache_shape).index_copy_(0, slots, values["value"])
    return values["key_cache"], values["value_cache"]


def copy_reference(torch: Any, values: dict[str, Any]) -> tuple[Any, Any]:
    del torch
    sources = values["mapping"][:, 0]
    destinations = values["mapping"][:, 1]
    key_source = values["key_cache"].index_select(0, sources)
    value_source = values["value_cache"].index_select(0, sources)
    values["key_cache"].index_copy_(0, destinations, key_source)
    values["value_cache"].index_copy_(0, destinations, value_source)
    return values["key_cache"], values["value_cache"]


def clone_for_call(values: dict[str, Any]) -> dict[str, Any]:
    return {
        name: tensor.clone() if name in {"key_cache", "value_cache"} else tensor
        for name, tensor in values.items()
    }


def candidate_call(extension: Any, operator: str, entrypoint: str, values: dict[str, Any]) -> Any:
    function = getattr(extension, entrypoint)
    if operator == "kv_cache_append":
        return function(
            values["key"],
            values["value"],
            values["key_cache"],
            values["value_cache"],
            values["mapping"],
        )
    return function(values["key_cache"], values["value_cache"], values["mapping"])


def reference_call(torch: Any, operator: str, values: dict[str, Any]) -> Any:
    if operator == "kv_cache_append":
        return append_reference(torch, values)
    return copy_reference(torch, values)


def bytes_moved(case: dict[str, Any], operator: str, element_size: int) -> int:
    units = case["tokens"] if operator == "kv_cache_append" else case["pairs"] * case["block_size"]
    return 4 * units * case["kv_heads"] * case["head_dim"] * element_size


def evaluate_case(
    torch: Any,
    extension: Any,
    operator: str,
    case: dict[str, Any],
    dtype_name: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    dtype = getattr(torch, DTYPES[dtype_name])
    entrypoints = ENTRYPOINTS[operator]
    correctness: dict[str, list[dict[str, Any]]] = {name: [] for name in entrypoints}
    for seed in (20260918, 20260919, 20260920):
        original = make_inputs(torch, operator, case, dtype, seed)
        expected_values = clone_for_call(original)
        expected = reference_call(torch, operator, expected_values)
        for name in entrypoints:
            candidate_values = clone_for_call(original)
            actual = candidate_call(extension, operator, name, candidate_values)
            for actual_cache, expected_cache in zip(actual, expected):
                torch.testing.assert_close(actual_cache, expected_cache, atol=0, rtol=0)
            correctness[name].append(
                {"seed": seed, "passed": True, "max_absolute_error": 0.0, "outputs": 2}
            )

    original = make_inputs(torch, operator, case, dtype, 20260918)
    call_values = {name: clone_for_call(original) for name in entrypoints}
    baseline_values = clone_for_call(original)
    calls: dict[str, Callable[[], Any]] = {
        name: (
            lambda name=name: candidate_call(extension, operator, name, call_values[name])
        )
        for name in entrypoints
    }
    calls["torch_index_copy"] = lambda: reference_call(torch, operator, baseline_values)
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

    moved = bytes_moved(case, operator, 2)
    timings = {name: summary(values, moved) for name, values in samples.items()}
    maximum_cv = float(policy["maximum_cv"])
    stable_candidates = [
        name
        for name in entrypoints
        if timings[name]["coefficient_of_variation"] <= maximum_cv
    ]
    selected = min(stable_candidates or entrypoints, key=lambda name: timings[name]["median_ms"])
    candidate = timings[selected]
    baseline = timings["torch_index_copy"]
    improvement = (baseline["median_ms"] - candidate["median_ms"]) / baseline["median_ms"] * 100.0
    gates = {
        "correctness_passed": all(item["passed"] for item in correctness[selected]),
        "candidate_stable": candidate["coefficient_of_variation"] <= maximum_cv,
        "baseline_stable": baseline["coefficient_of_variation"] <= maximum_cv,
        "minimum_improvement": improvement >= float(policy["minimum_improvement_pct"]),
    }
    return {
        "case_id": case["id"],
        "operator": operator,
        "dtype": dtype_name,
        "shape": {
            key: value
            for key, value in case.items()
            if key != "id"
        }
        | {"index_dtype": "int64", "layout": "block_slot_head_dim"},
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


def load_extension(
    torch: Any,
    source: Path,
    build_directory: Path,
    architecture: str,
    verbose: bool,
) -> Any:
    from torch.utils.cpp_extension import load

    os.environ["PYTORCH_ROCM_ARCH"] = architecture
    os.environ.setdefault("MAX_JOBS", "2")
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name=f"radeon_kernels_kv_cache_{architecture}",
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
    parser.add_argument("--workload", type=Path, action="append", required=True)
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
    workloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in args.workload]
    if {workload["operator"] for workload in workloads} != set(ENTRYPOINTS):
        raise ValueError("workloads must contain kv_cache_append and kv_cache_copy")
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
    for workload in workloads:
        operator = workload["operator"]
        for case in workload["cases"]:
            for dtype in workload["dtypes"]:
                try:
                    print(f"evaluating {operator} {case['id']} {dtype}", flush=True)
                    results.append(
                        evaluate_case(
                            torch,
                            extension,
                            operator,
                            case,
                            dtype,
                            workload["benchmark"],
                        )
                    )
                except Exception as error:
                    failures.append(
                        {
                            "operator": operator,
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
            "oracle": "torch_index_copy",
            "baseline": "torch_index_copy",
            "layout": "block_slot_head_dim",
        },
        "candidate": {
            "source_sha256": sha256_file(args.source),
            "artifact_file": args.artifact.name,
            "artifact_sha256": sha256_file(args.artifact),
            "entrypoints": {
                entrypoint: entrypoint
                for entrypoints in ENTRYPOINTS.values()
                for entrypoint in entrypoints
            },
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
