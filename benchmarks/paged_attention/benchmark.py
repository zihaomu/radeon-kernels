#!/usr/bin/env python3
"""Build and evaluate online-softmax Paged Attention Decode candidates."""

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
ENTRYPOINTS = ("attention_wave1", "attention_wave4", "attention_wave8")


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


def summary(samples: list[float], bytes_read: int) -> dict[str, Any]:
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    return {
        "samples_ms": samples,
        "median_ms": median,
        "p20_ms": percentile(samples, 0.2),
        "p80_ms": percentile(samples, 0.8),
        "coefficient_of_variation": statistics.pstdev(samples) / mean if mean else math.inf,
        "effective_gbps": bytes_read / (median * 1.0e6),
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


def make_inputs(
    torch: Any, case: dict[str, Any], dtype: Any, seed: int
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    maximum_blocks = math.ceil(case["context_length"] / case["block_size"])
    num_blocks = case["batch"] * maximum_blocks
    cache_shape = (
        num_blocks,
        case["block_size"],
        case["kv_heads"],
        case["head_dim"],
    )
    physical_blocks = torch.randperm(
        num_blocks, device="cuda", dtype=torch.int64, generator=generator
    ).to(dtype=torch.int32)
    return {
        "query": random_tensor(
            torch,
            (case["batch"], case["query_heads"], case["head_dim"]),
            dtype,
            generator,
        ),
        "key_cache": random_tensor(torch, cache_shape, dtype, generator),
        "value_cache": random_tensor(torch, cache_shape, dtype, generator),
        "block_tables": physical_blocks.view(case["batch"], maximum_blocks).contiguous(),
        "context_lengths": torch.full(
            (case["batch"],),
            case["context_length"],
            device="cuda",
            dtype=torch.int32,
        ),
        "scale": 1.0 / math.sqrt(case["head_dim"]),
        "num_blocks": num_blocks,
        "max_blocks_per_sequence": maximum_blocks,
    }


def torch_reference(torch: Any, values: dict[str, Any]) -> Any:
    query = values["query"]
    key_cache = values["key_cache"]
    value_cache = values["value_cache"]
    block_tables = values["block_tables"]
    context_lengths = values["context_lengths"]
    batch, query_heads, head_dim = query.shape
    block_size = key_cache.shape[1]
    kv_heads = key_cache.shape[2]
    maximum_context = block_tables.shape[1] * block_size
    positions = torch.arange(maximum_context, device="cuda")
    physical_blocks = block_tables.long().index_select(1, positions // block_size)
    physical_slots = physical_blocks * block_size + (positions % block_size).unsqueeze(0)
    cache_shape = (-1, kv_heads, head_dim)
    keys = key_cache.view(cache_shape).index_select(0, physical_slots.reshape(-1))
    values_cache = value_cache.view(cache_shape).index_select(0, physical_slots.reshape(-1))
    keys = keys.view(batch, maximum_context, kv_heads, head_dim)
    values_cache = values_cache.view(batch, maximum_context, kv_heads, head_dim)
    kv_head_indices = torch.arange(query_heads, device="cuda") // (query_heads // kv_heads)
    keys = keys.index_select(2, kv_head_indices).float()
    values_cache = values_cache.index_select(2, kv_head_indices).float()
    scores = torch.einsum("bhd,bthd->bht", query.float(), keys) * values["scale"]
    valid = positions.unsqueeze(0) < context_lengths.long().unsqueeze(1)
    probabilities = torch.softmax(
        scores.masked_fill(~valid.unsqueeze(1), float("-inf")), dim=-1, dtype=torch.float32
    )
    return torch.einsum("bht,bthd->bhd", probabilities, values_cache).to(query.dtype)


def candidate_call(extension: Any, entrypoint: str, values: dict[str, Any]) -> Any:
    return getattr(extension, entrypoint)(
        values["query"],
        values["key_cache"],
        values["value_cache"],
        values["block_tables"],
        values["context_lengths"],
        values["scale"],
    )


def evaluated_shape(case: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch": case["batch"],
        "query_heads": case["query_heads"],
        "kv_heads": case["kv_heads"],
        "head_dim": case["head_dim"],
        "num_blocks": values["num_blocks"],
        "block_size": case["block_size"],
        "max_blocks_per_sequence": values["max_blocks_per_sequence"],
        "index_dtype": "int32",
        "cache_layout": "block_slot_head_dim",
        "scale_mode": "default",
    }


def bytes_read(case: dict[str, Any]) -> int:
    cache_bytes = (
        case["batch"]
        * case["query_heads"]
        * case["context_length"]
        * case["head_dim"]
        * 2
        * 2
    )
    query_and_output = case["batch"] * case["query_heads"] * case["head_dim"] * 2 * 2
    return cache_bytes + query_and_output


def evaluate_case(
    torch: Any,
    extension: Any,
    case: dict[str, Any],
    dtype_name: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    dtype = getattr(torch, DTYPES[dtype_name])
    correctness: dict[str, list[dict[str, Any]]] = {name: [] for name in ENTRYPOINTS}
    last_values: dict[str, Any] | None = None
    for seed in (20260918, 20260919, 20260920):
        values = make_inputs(torch, case, dtype, seed)
        expected = torch_reference(torch, values)
        for name in ENTRYPOINTS:
            actual = candidate_call(extension, name, values)
            tolerance = 0.015 if dtype_name == "fp16" else 0.03
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            correctness[name].append(
                {
                    "seed": seed,
                    "passed": True,
                    "max_absolute_error": float((actual.float() - expected.float()).abs().max()),
                    "outputs": 1,
                }
            )
        last_values = values
    assert last_values is not None

    calls: dict[str, Callable[[], Any]] = {
        name: (lambda name=name: candidate_call(extension, name, last_values))
        for name in ENTRYPOINTS
    }
    calls["torch_paged_attention"] = lambda: torch_reference(torch, last_values)
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

    traffic = bytes_read(case)
    timings = {name: summary(values, traffic) for name, values in samples.items()}
    maximum_cv = float(policy["maximum_cv"])
    stable_candidates = [
        name
        for name in ENTRYPOINTS
        if timings[name]["coefficient_of_variation"] <= maximum_cv
    ]
    selected = min(
        stable_candidates or ENTRYPOINTS, key=lambda name: timings[name]["median_ms"]
    )
    candidate = timings[selected]
    baseline = timings["torch_paged_attention"]
    improvement = (baseline["median_ms"] - candidate["median_ms"]) / baseline[
        "median_ms"
    ] * 100.0
    gates = {
        "correctness_passed": all(item["passed"] for item in correctness[selected]),
        "candidate_stable": candidate["coefficient_of_variation"] <= maximum_cv,
        "baseline_stable": baseline["coefficient_of_variation"] <= maximum_cv,
        "minimum_improvement": improvement >= float(policy["minimum_improvement_pct"]),
    }
    return {
        "case_id": case["id"],
        "operator": "paged_attention_decode",
        "dtype": dtype_name,
        "shape": evaluated_shape(case, last_values),
        "selected_entrypoint": selected,
        "correctness": correctness[selected],
        "variant_correctness": correctness,
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
        name=f"radeon_kernels_paged_attention_{architecture}",
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
    if workload.get("operator") != "paged_attention_decode":
        raise ValueError("workload must describe paged_attention_decode")
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
                print(f"evaluating Paged Attention {case['id']} {dtype_name}", flush=True)
                results.append(
                    evaluate_case(
                        torch,
                        extension,
                        case,
                        dtype_name,
                        workload["benchmark"],
                    )
                )
            except Exception as error:
                failures.append(
                    {
                        "operator": "paged_attention_decode",
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
            "oracle": "torch_fp32_paged_reference",
            "baseline": "torch_fp32_paged_reference",
            "cache_layout": "block_slot_head_dim",
            "attention_mode": "causal_decode_single_query",
            "softmax": "online_fp32",
            "entrypoints": list(ENTRYPOINTS),
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
