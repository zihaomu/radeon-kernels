#!/usr/bin/env python3
"""Deterministic source-mode adapters for AsmEvo GEMM experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1
DTYPES = {"fp16": "float16", "bf16": "bfloat16"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(document: dict[str, Any]) -> str:
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return document


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_output(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": command, "available": False}
    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def resolve_llvm_toolchain() -> tuple[str, str]:
    """Resolve a matched objdump/readobj pair from one ROCm LLVM toolchain."""
    roots: list[Path] = []
    compiler = shutil.which("amdclang++")
    if compiler is not None:
        roots.append(Path(compiler).resolve().parent)
    roots.extend(
        [
            Path("/opt/rocm/llvm/bin"),
            Path("/opt/rocm/lib/llvm/bin"),
        ]
    )
    roots.extend(
        Path(entry) / "_rocm_sdk_devel/lib/llvm/bin"
        for entry in sys.path
        if entry
    )

    path_objdump = shutil.which("llvm-objdump")
    path_readobj = shutil.which("llvm-readobj")
    if path_objdump is not None and path_readobj is not None:
        objdump_root = Path(path_objdump).resolve().parent
        readobj_root = Path(path_readobj).resolve().parent
        if objdump_root == readobj_root:
            roots.append(objdump_root)

    seen: set[Path] = set()
    for root in roots:
        resolved_root = root.resolve()
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        objdump = resolved_root / "llvm-objdump"
        readobj = resolved_root / "llvm-readobj"
        if all(path.is_file() and os.access(path, os.X_OK) for path in (objdump, readobj)):
            return str(objdump), str(readobj)
    raise FileNotFoundError("a matched llvm-objdump/llvm-readobj toolchain is unavailable")


def executable_identity(path: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    completed = subprocess.run(
        [str(resolved), "--version"], capture_output=True, check=False, text=True
    )
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
        "version_returncode": completed.returncode,
        "version_stdout": completed.stdout,
        "version_stderr": completed.stderr,
    }


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.lower().split("x"))
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("shape must be MxNxK with positive dimensions")
    return parts


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def case_id(dtype: str, shape: tuple[int, int, int], seed: int, variant: int) -> str:
    m, n, k = shape
    return f"{dtype}-m{m}-n{n}-k{k}-seed{seed}-variant{variant}"


def architecture(torch: Any) -> str:
    value = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return value.split(":", 1)[0]


def check_contract(
    path: Path, expected_arch: str, expected_cases: list[str]
) -> tuple[str, dict[str, Any]]:
    contract = load_json(path)
    if contract.get("mode") != "source":
        raise ValueError("AsmEvo worker requires a source-mode contract")
    if contract.get("target_arch") != expected_arch:
        raise ValueError("contract architecture does not match the detected target")
    if contract.get("case_ids") != expected_cases:
        raise ValueError("worker cases do not exactly match the frozen contract")
    return json_sha256(contract), contract


def check_workload(
    contract: dict[str, Any],
    *,
    dtype: str,
    shape: tuple[int, int, int],
    variant: int,
    seeds: list[int],
) -> dict[str, Any]:
    workload = contract.get("workload")
    if not isinstance(workload, dict):
        raise ValueError("contract.workload must be an object")
    expected = {
        "dtype": dtype,
        "shape": list(shape),
        "variant": variant,
        "correctness_seeds": seeds,
    }
    for key, value in expected.items():
        if workload.get(key) != value:
            raise ValueError(f"worker {key} does not match contract.workload.{key}")
    return workload


def import_extension(path: Path, cache: dict[Path, Any]) -> Any:
    resolved = path.resolve()
    if resolved in cache:
        return cache[resolved]
    module_name = resolved.name.split(".", 1)[0]
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension spec: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache[resolved] = module
    return module


def run_environment(args: argparse.Namespace) -> int:
    import torch

    detected = architecture(torch)
    if detected != args.architecture:
        raise RuntimeError(f"expected {args.architecture}, detected {detected}")
    properties = torch.cuda.get_device_properties(0)
    llvm_objdump, llvm_readobj = resolve_llvm_toolchain()
    llvm_toolchain = {
        "llvm_objdump": executable_identity(llvm_objdump),
        "llvm_readobj": executable_identity(llvm_readobj),
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": now(),
        "target_id": args.target_id,
        "target_arch": detected,
        "wavefront_size": 32,
        "image_digest": args.image_digest,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "properties": str(properties),
            "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torch_rocm": torch.version.hip,
            "platform": platform.platform(),
            "hipcc": command_output(["hipcc", "--version"]),
            "amdclang": command_output(["amdclang++", "--version"]),
            "llvm_toolchain": llvm_toolchain,
            "rocminfo": command_output(["rocminfo"]),
            "rocm_smi": command_output(["rocm-smi", "--showdriverversion"]),
        },
        "measurement_policy": {
            "device_timing": "torch.cuda.Event",
            "alternating_order": True,
            "input_distribution": "uniform[-0.125,0.125]",
            "nan_inf_policy": "both oracle and candidate must be finite",
            "subnormal_policy": "native ROCm/PyTorch behavior",
            "signed_zero_policy": "numeric comparison",
            "dtype": args.dtype,
            "shape": {"m": args.shape[0], "n": args.shape[1], "k": args.shape[2]},
            "variant": args.variant,
            "correctness_seeds": args.seeds,
            "warmups": args.warmups,
            "samples": args.samples,
            "repetitions_per_sample": args.repetitions,
            "sample_ms": args.sample_ms,
            "pilot_repetitions": args.pilot_repetitions,
            "timing_seed": args.timing_seed,
            "order_seed": args.order_seed,
            "guard_elements": args.guard_elements,
            "include_torch_in_promotion_timing": False,
        },
        "immutable_inputs": {
            "adapter": {
                "path": str(args.adapter.resolve()),
                "sha256": sha256(args.adapter.resolve()),
            },
            "k0_artifact": {
                "path": str(args.k0_artifact.resolve()),
                "sha256": sha256(args.k0_artifact.resolve()),
            },
            "k0_source": {
                "path": str(args.k0_source.resolve()),
                "sha256": sha256(args.k0_source.resolve()),
            },
            "historical_evidence": {
                "path": str(args.historical_evidence.resolve()),
                "sha256": sha256(args.historical_evidence.resolve()),
            },
        },
    }
    write_json(args.output, document)
    return 0


def run_build(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.cpp_extension import load

    started = now()
    output_dir = args.output_dir.resolve()
    build_dir = output_dir / "build"
    artifact_dir = output_dir / "artifacts"
    build_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    candidate_token = re.sub(r"[^A-Za-z0-9_]", "_", args.candidate_id)
    if not candidate_token or candidate_token[0].isdigit():
        candidate_token = f"c_{candidate_token}"
    module_name = f"radeon_kernels_wmma_{args.architecture}_{candidate_token}"
    artifact = artifact_dir / f"{module_name}.so"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": args.candidate_id,
        "started_at": started,
        "architecture": args.architecture,
        "module_name": module_name,
        "source": str(source),
        "source_sha256": sha256(source),
        "artifact": str(artifact),
        "status": "RUNNING",
    }
    try:
        detected = architecture(torch)
        if detected != args.architecture:
            raise RuntimeError(f"expected {args.architecture}, detected {detected}")
        os.environ["PYTORCH_ROCM_ARCH"] = args.architecture
        os.environ.setdefault("MAX_JOBS", "2")
        extension = load(
            name=module_name,
            sources=[str(source)],
            build_directory=str(build_dir),
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "-ffast-math",
                "-U__HIP_NO_HALF_OPERATORS__",
                "-U__HIP_NO_HALF_CONVERSIONS__",
            ],
            with_cuda=True,
            verbose=True,
        )
        built = Path(extension.__file__).resolve()
        if built != artifact:
            shutil.copy2(built, artifact)
        manifest.update(
            {
                "status": "SUCCEEDED",
                "passed": True,
                "artifact_sha256": sha256(artifact),
                "artifact_size_bytes": artifact.stat().st_size,
                "torch": torch.__version__,
                "rocm": torch.version.hip,
                "hipcc": command_output(["hipcc", "--version"]),
            }
        )
    except Exception as error:
        manifest.update(
            {
                "status": "FAILED",
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        raise
    finally:
        manifest["completed_at"] = now()
        write_json(args.output, manifest)
    return 0


def guarded_tensor(torch: Any, shape: tuple[int, ...], dtype: Any, guard: int, fill: float) -> tuple[Any, Any]:
    elements = math.prod(shape)
    storage = torch.full((elements + 2 * guard,), fill, device="cuda", dtype=dtype)
    tensor = storage[guard : guard + elements].view(shape)
    return tensor, storage


def guards_intact(torch: Any, storage: Any, guard: int, fill: float) -> bool:
    expected = torch.tensor(fill, device=storage.device, dtype=storage.dtype)
    return bool(torch.all(storage[:guard] == expected) and torch.all(storage[-guard:] == expected))


def run_verify(args: argparse.Namespace) -> int:
    import torch

    started = now()
    detected = architecture(torch)
    if detected != args.architecture:
        raise RuntimeError(f"expected {args.architecture}, detected {detected}")
    cases = [case_id(args.dtype, args.shape, seed, args.variant) for seed in args.seeds]
    contract_sha, contract = check_contract(args.contract, detected, cases)
    workload = check_workload(
        contract,
        dtype=args.dtype,
        shape=args.shape,
        variant=args.variant,
        seeds=args.seeds,
    )
    if workload.get("guard_elements") != args.guard_elements or args.guard_elements < 1:
        raise RuntimeError("guard size does not match contract.workload.guard_elements")
    artifact = args.artifact.resolve()
    extension = import_extension(artifact, {})
    dtype = getattr(torch, DTYPES[args.dtype])
    m, n, k = args.shape
    guard = args.guard_elements
    sentinel = 7.0
    results: list[dict[str, Any]] = []
    for seed, frozen_case_id in zip(args.seeds, cases, strict=True):
        result: dict[str, Any] = {
            "case_id": frozen_case_id,
            "runtime_ok": False,
            "float_metrics_applicable": True,
            "integer_exact": False,
            "guards_intact": False,
        }
        try:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            a, a_storage = guarded_tensor(torch, (m, k), dtype, guard, sentinel)
            b, b_storage = guarded_tensor(torch, (k, n), dtype, guard, sentinel)
            output, output_storage = guarded_tensor(torch, (m, n), dtype, guard, sentinel)
            a.uniform_(-0.125, 0.125, generator=generator)
            b.uniform_(-0.125, 0.125, generator=generator)
            a_before = a.clone()
            b_before = b.clone()
            reference = torch.mm(a, b)
            returned = extension.gemm_out(a, b, output, args.variant)
            torch.cuda.synchronize()
            output_float = output.float().reshape(-1)
            reference_float = reference.float().reshape(-1)
            difference = (output_float - reference_float).abs()
            denominator = torch.linalg.vector_norm(output_float) * torch.linalg.vector_norm(reference_float)
            tensor_finite = bool(
                torch.isfinite(output_float).all() and torch.isfinite(reference_float).all()
            )
            denominator_value = float(denominator.item())
            if tensor_finite and math.isfinite(denominator_value) and denominator_value > 0.0:
                cosine_raw = float((torch.dot(output_float, reference_float) / denominator).item())
                cosine_finite = math.isfinite(cosine_raw)
                cosine_value = max(-1.0, min(1.0, cosine_raw)) if cosine_finite else -1.0
            else:
                cosine_finite = False
                cosine_value = -1.0
            finite = tensor_finite and cosine_finite
            exact_state = bool(
                returned.data_ptr() == output.data_ptr()
                and tuple(output.shape) == (m, n)
                and tuple(output.stride()) == (n, 1)
                and torch.equal(a, a_before)
                and torch.equal(b, b_before)
            )
            intact = all(
                guards_intact(torch, storage, guard, sentinel)
                for storage in (a_storage, b_storage, output_storage)
            )
            result.update(
                {
                    "runtime_ok": True,
                    "cosine_similarity": cosine_value,
                    "max_absolute_error": float(difference.max().item()) if finite else 1.0e30,
                    "mean_absolute_error": float(difference.mean().item()) if finite else 1.0e30,
                    "integer_exact": exact_state,
                    "guards_intact": intact,
                    "candidate_finite": finite,
                    "candidate_nan_count": int(torch.isnan(output_float).sum().item()),
                    "candidate_inf_count": int(torch.isinf(output_float).sum().item()),
                }
            )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        results.append(result)
        torch.cuda.empty_cache()
    document = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started,
        "completed_at": now(),
        "artifact": str(artifact),
        "artifact_sha256": sha256(artifact),
        "architecture": detected,
        "contract_sha256": contract_sha,
        "oracle": "torch.mm",
        "dtype": args.dtype,
        "shape": {"m": m, "n": n, "k": k},
        "variant": args.variant,
        "cases": results,
    }
    write_json(args.output, document)
    return 0 if all(case["runtime_ok"] for case in results) else 1


def measure(torch: Any, invoke: Callable[[], Any], repetitions: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        invoke()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / repetitions)


def run_benchmark(args: argparse.Namespace) -> int:
    import torch

    detected = architecture(torch)
    if detected != args.architecture:
        raise RuntimeError(f"expected {args.architecture}, detected {detected}")
    cases = [case_id(args.dtype, args.shape, seed, args.variant) for seed in args.seeds]
    contract_sha, contract = check_contract(args.contract, detected, cases)
    workload = check_workload(
        contract,
        dtype=args.dtype,
        shape=args.shape,
        variant=args.variant,
        seeds=args.seeds,
    )
    frozen_method = {
        "warmups": args.warmups,
        "samples": args.samples,
        "repetitions": args.repetitions,
        "pilot_repetitions": args.pilot_repetitions,
        "timing_seed": args.timing_seed,
        "order_seed": args.order_seed,
        "sample_ms": args.sample_ms,
        "include_torch": args.include_torch,
    }
    for key, value in frozen_method.items():
        if workload.get(key) != value:
            raise RuntimeError(f"benchmark {key} does not match contract.workload.{key}")
    oracle = load_json(args.oracle_evidence)
    candidate_path = args.candidate.resolve()
    if oracle.get("artifact_sha256") != sha256(candidate_path):
        raise RuntimeError("oracle evidence does not belong to the candidate artifact")
    if oracle.get("contract_sha256") != contract_sha:
        raise RuntimeError("oracle evidence does not belong to the frozen contract")
    oracle_cases = oracle.get("cases", [])
    policy = contract["equivalence_policy"]
    if [case.get("case_id") for case in oracle_cases] != cases:
        raise RuntimeError("oracle case order does not match the frozen contract")
    if any(
        case.get("runtime_ok") is not True
        or case.get("candidate_finite") is not True
        or case.get("cosine_similarity", -1.0) < policy["min_cosine_similarity"]
        or case.get("max_absolute_error", float("inf")) > policy["max_absolute_error"]
        or case.get("integer_exact") is not True
        or case.get("guards_intact") is not True
        for case in oracle_cases
    ):
        raise RuntimeError("candidate did not pass equivalence and guard gates; refusing to time it")

    cache: dict[Path, Any] = {}
    paths = {
        "original": args.original.resolve(),
        "parent": args.parent.resolve(),
        "candidate": candidate_path,
    }
    extensions = {name: import_extension(path, cache) for name, path in paths.items()}
    dtype = getattr(torch, DTYPES[args.dtype])
    m, n, k = args.shape
    generator = torch.Generator(device="cuda").manual_seed(args.timing_seed)
    a = torch.empty((m, k), device="cuda", dtype=dtype).uniform_(
        -0.125, 0.125, generator=generator
    )
    b = torch.empty((k, n), device="cuda", dtype=dtype).uniform_(
        -0.125, 0.125, generator=generator
    )
    outputs = {name: torch.empty((m, n), device="cuda", dtype=dtype) for name in paths}
    invokes: dict[str, Callable[[], Any]] = {
        name: (lambda name=name: extensions[name].gemm_out(a, b, outputs[name], args.variant))
        for name in paths
    }
    if args.include_torch:
        outputs["torch_mm"] = torch.empty((m, n), device="cuda", dtype=dtype)
        invokes["torch_mm"] = lambda: torch.mm(a, b, out=outputs["torch_mm"])

    for _ in range(args.warmups):
        for invoke in invokes.values():
            invoke()
    torch.cuda.synchronize()
    pilot = {name: measure(torch, invoke, args.pilot_repetitions) for name, invoke in invokes.items()}
    repetitions = args.repetitions
    samples = {name: [] for name in invokes}
    order_log: list[list[str]] = []
    order = list(invokes)
    rng = random.Random(args.order_seed)
    rng.shuffle(order)
    started = now()
    for sample_index in range(args.samples):
        offset = sample_index % len(order)
        sample_order = order[offset:] + order[:offset]
        order_log.append(sample_order)
        for name in sample_order:
            samples[name].append(measure(torch, invokes[name], repetitions))
    completed = now()

    summaries = {}
    for name, values in samples.items():
        mean = statistics.fmean(values)
        summaries[name] = {
            "median_ms": statistics.median(values),
            "mean_ms": mean,
            "cv": statistics.pstdev(values) / mean,
        }
    document = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started,
        "completed_at": completed,
        "architecture": detected,
        "contract_sha256": contract_sha,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()
        },
        "dtype": args.dtype,
        "shape": {"m": m, "n": n, "k": k},
        "variant": args.variant,
        "method": {
            "warmups": args.warmups,
            "samples": args.samples,
            "target_sample_ms": args.sample_ms,
            "pilot_repetitions": args.pilot_repetitions,
            "repetitions_per_sample": repetitions,
            "timing_seed": args.timing_seed,
            "order_seed": args.order_seed,
        },
        "pilot_ms": pilot,
        "order": order_log,
        "samples_ms": samples,
        "summaries": summaries,
        "timing_valid": True,
    }
    write_json(args.output, document)
    return 0


def run_inspect(args: argparse.Namespace) -> int:
    artifact = args.artifact.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError("inspection directory must be empty")
    llvm_objdump, llvm_readobj = resolve_llvm_toolchain()
    llvm_toolchain = {
        "llvm_objdump": executable_identity(llvm_objdump),
        "llvm_readobj": executable_identity(llvm_readobj),
    }
    inspection_artifact = output_dir / artifact.name
    if inspection_artifact == artifact:
        raise ValueError("inspection output directory must differ from the artifact directory")
    shutil.copyfile(artifact, inspection_artifact)
    extraction_pattern = f"{inspection_artifact.name}.*{args.architecture}*"
    offload = subprocess.run(
        [llvm_objdump, "--offloading", str(inspection_artifact)],
        cwd=output_dir,
        capture_output=True,
        check=False,
        text=True,
    )
    (output_dir / "offloading.log").write_text(
        offload.stdout + offload.stderr, encoding="utf-8"
    )
    extracted = sorted(output_dir.glob(extraction_pattern))
    code_object = extracted[0] if offload.returncode == 0 and len(extracted) == 1 else None
    module_name = artifact.name.split(".", 1)[0]
    dynamic_symbols = command_output(["nm", "-D", str(artifact)])
    pyinit_symbol = f"PyInit_{module_name}"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": str(artifact),
        "artifact_sha256": sha256(artifact),
        "inspection_artifact": str(inspection_artifact),
        "inspection_artifact_sha256": sha256(inspection_artifact),
        "architecture": args.architecture,
        "llvm_toolchain": llvm_toolchain,
        "offload_returncode": offload.returncode,
        "code_object": None,
        "wmma_instruction_count": 0,
        "barrier_instruction_count": 0,
        "barrier_signal_count": 0,
        "barrier_wait_count": 0,
        "global_invalidate_count": 0,
        "pyinit_symbol": pyinit_symbol,
        "pyinit_symbol_present": pyinit_symbol in dynamic_symbols.get("stdout", ""),
        "extracted_object_count": len(extracted),
    }
    if code_object is not None:
        disassembly = subprocess.run(
            [llvm_objdump, "--disassemble", f"--mcpu={args.architecture}", str(code_object)],
            capture_output=True,
            check=False,
            text=True,
        )
        disassembly_path = output_dir / "disassembly.txt"
        disassembly_path.write_text(disassembly.stdout + disassembly.stderr, encoding="utf-8")
        metadata = subprocess.run(
            [llvm_readobj, "--notes", str(code_object)],
            capture_output=True,
            check=False,
            text=True,
        )
        metadata_path = output_dir / "metadata.txt"
        metadata_path.write_text(metadata.stdout + metadata.stderr, encoding="utf-8")
        text = disassembly.stdout
        manifest.update(
            {
                "code_object": str(code_object),
                "code_object_sha256": sha256(code_object),
                "disassembly": str(disassembly_path),
                "disassembly_returncode": disassembly.returncode,
                "metadata": str(metadata_path),
                "metadata_returncode": metadata.returncode,
                "metadata_architecture_confirmed": args.architecture in metadata.stdout,
                "wmma_instruction_count": len(re.findall(r"\bv_wmma_", text)),
                "barrier_instruction_count": len(
                    re.findall(r"\bs_barrier(?:_signal|_wait)?\b", text)
                ),
                "barrier_signal_count": len(re.findall(r"\bs_barrier_signal\b", text)),
                "barrier_wait_count": len(re.findall(r"\bs_barrier_wait\b", text)),
                "global_invalidate_count": len(
                    re.findall(r"\b(?:buffer_gl0_inv|global_inv)\b", text)
                ),
            }
        )
    passed = bool(
        code_object is not None
        and manifest.get("disassembly_returncode") == 0
        and manifest.get("metadata_returncode") == 0
        and manifest.get("metadata_architecture_confirmed") is True
        and manifest.get("pyinit_symbol_present") is True
        and manifest.get("wmma_instruction_count", 0) > 0
    )
    manifest["passed"] = passed
    write_json(args.output, manifest)
    return 0 if passed else 1


def run_evaluate(args: argparse.Namespace) -> int:
    contract = load_json(args.contract)
    build = load_json(args.build_evidence)
    oracle = load_json(args.oracle_evidence)
    timing = load_json(args.timing_evidence)
    original = args.original.resolve()
    parent = args.parent.resolve()
    candidate = args.candidate.resolve()
    expected_hashes = {
        "original": sha256(original),
        "parent": sha256(parent),
        "candidate": sha256(candidate),
    }
    proposal_hash = sha256(args.proposal.resolve())
    if (
        build.get("passed") is not True
        or build.get("artifact_sha256") != expected_hashes["candidate"]
        or build.get("source_sha256") != proposal_hash
        or build.get("candidate_id") != args.candidate_id
        or build.get("architecture") != contract.get("target_arch")
    ):
        raise ValueError("build evidence does not match the successful candidate artifact")
    workload = contract.get("workload", {})
    expected_shape = workload.get("shape", [])
    expected_shape_object = (
        {"m": expected_shape[0], "n": expected_shape[1], "k": expected_shape[2]}
        if isinstance(expected_shape, list) and len(expected_shape) == 3
        else None
    )
    if (
        oracle.get("artifact_sha256") != expected_hashes["candidate"]
        or oracle.get("architecture") != contract.get("target_arch")
        or oracle.get("dtype") != workload.get("dtype")
        or oracle.get("shape") != expected_shape_object
        or oracle.get("variant") != workload.get("variant")
    ):
        raise ValueError("oracle evidence does not match the candidate artifact")
    static = load_json(args.static_evidence)
    if (
        static.get("passed") is not True
        or static.get("artifact_sha256") != expected_hashes["candidate"]
        or static.get("inspection_artifact_sha256") != expected_hashes["candidate"]
        or static.get("architecture") != contract.get("target_arch")
        or static.get("llvm_toolchain") != contract.get("static_toolchain")
        or static.get("offload_returncode") != 0
        or static.get("disassembly_returncode") != 0
        or static.get("metadata_returncode") != 0
        or static.get("metadata_architecture_confirmed") is not True
        or static.get("pyinit_symbol_present") is not True
        or static.get("wmma_instruction_count", 0) <= 0
        or not static.get("code_object")
    ):
        raise ValueError("static evidence does not prove a matching WMMA code object")
    timing_artifacts = timing.get("artifacts", {})
    if set(timing_artifacts) != set(expected_hashes):
        raise ValueError("timing evidence must identify original, parent, and candidate")
    for role, expected_hash in expected_hashes.items():
        identity = timing_artifacts.get(role)
        if not isinstance(identity, dict) or identity.get("sha256") != expected_hash:
            raise ValueError(f"timing evidence has a mismatched {role} artifact")
    if (
        timing.get("architecture") != contract.get("target_arch")
        or timing.get("dtype") != workload.get("dtype")
        or timing.get("shape") != expected_shape_object
        or timing.get("variant") != workload.get("variant")
    ):
        raise ValueError("timing workload does not match the frozen contract")
    method = timing.get("method", {})
    expected_method = {
        "warmups": workload.get("warmups"),
        "samples": workload.get("samples"),
        "target_sample_ms": workload.get("sample_ms"),
        "pilot_repetitions": workload.get("pilot_repetitions"),
        "repetitions_per_sample": workload.get("repetitions"),
        "timing_seed": workload.get("timing_seed"),
        "order_seed": workload.get("order_seed"),
    }
    if method != expected_method:
        raise ValueError("timing method does not match the frozen contract")
    raw_samples = timing.get("samples_ms", {})
    if set(raw_samples) != set(expected_hashes) or any(
        not isinstance(raw_samples[role], list)
        or len(raw_samples[role]) != workload.get("samples")
        for role in expected_hashes
    ):
        raise ValueError("timing sample arrays do not match the frozen contract")
    contract_sha = json_sha256(contract)
    if oracle.get("contract_sha256") != contract_sha or timing.get("contract_sha256") != contract_sha:
        raise ValueError("raw evidence does not match the frozen contract")
    equivalence_cases = []
    for case in oracle.get("cases", []):
        equivalence_cases.append(
            {
                key: case[key]
                for key in (
                    "case_id",
                    "runtime_ok",
                    "float_metrics_applicable",
                    "cosine_similarity",
                    "max_absolute_error",
                    "integer_exact",
                    "guards_intact",
                )
                if key in case
            }
        )
    document = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": args.candidate_id,
        "parent_id": args.parent_id,
        "original_id": "K0",
        "mode": "source",
        "proposal_sha256": proposal_hash,
        "contract": contract,
        "artifacts": {
            "original_sha256": expected_hashes["original"],
            "parent_sha256": expected_hashes["parent"],
            "candidate_sha256": expected_hashes["candidate"],
        },
        "checks": {
            "build": {"passed": True, "evidence": str(args.build_evidence.resolve())},
            "static_consistency": {
                "required": True,
                "passed": True,
                "evidence": str(args.static_evidence.resolve()),
            },
        },
        "equivalence": {"cases": equivalence_cases},
        "timing": {
            "valid": timing.get("timing_valid") is True,
            "original_ms": timing["samples_ms"]["original"],
            "parent_ms": timing["samples_ms"]["parent"],
            "candidate_ms": timing["samples_ms"]["candidate"],
        },
        "raw_evidence": {
            "oracle": str(args.oracle_evidence.resolve()),
            "timing": str(args.timing_evidence.resolve()),
            "static": str(args.static_evidence.resolve()),
        },
        "evidence_files": {
            "build_sha256": sha256(args.build_evidence.resolve()),
            "oracle_sha256": sha256(args.oracle_evidence.resolve()),
            "timing_sha256": sha256(args.timing_evidence.resolve()),
            "static_sha256": sha256(args.static_evidence.resolve()),
        },
    }
    write_json(args.output, document)
    return 0


def add_common_case_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp16")
    parser.add_argument("--shape", type=parse_shape, required=True)
    parser.add_argument("--variant", type=int, required=True)
    parser.add_argument("--seed", type=int, action="append", dest="seeds", required=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    environment = commands.add_parser("environment")
    environment.add_argument("--output", type=Path, required=True)
    environment.add_argument("--target-id", required=True)
    environment.add_argument("--architecture", required=True)
    environment.add_argument("--image-digest", required=True)
    environment.add_argument("--dtype", choices=tuple(DTYPES), default="fp16")
    environment.add_argument("--shape", type=parse_shape, required=True)
    environment.add_argument("--variant", type=int, required=True)
    environment.add_argument("--seed", type=int, action="append", dest="seeds", required=True)
    environment.add_argument("--warmups", type=positive_int, required=True)
    environment.add_argument("--samples", type=positive_int, required=True)
    environment.add_argument("--repetitions", type=positive_int, required=True)
    environment.add_argument("--sample-ms", type=positive_float, default=100.0)
    environment.add_argument("--pilot-repetitions", type=positive_int, default=20)
    environment.add_argument("--timing-seed", type=int, default=20260917)
    environment.add_argument("--order-seed", type=int, default=20260917)
    environment.add_argument("--guard-elements", type=positive_int, default=4096)
    environment.add_argument("--adapter", type=Path, required=True)
    environment.add_argument("--k0-artifact", type=Path, required=True)
    environment.add_argument("--k0-source", type=Path, required=True)
    environment.add_argument("--historical-evidence", type=Path, required=True)
    environment.set_defaults(func=run_environment)

    build = commands.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--architecture", required=True)
    build.add_argument("--candidate-id", required=True)
    build.set_defaults(func=run_build)

    verify = commands.add_parser("verify")
    add_common_case_arguments(verify)
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--guard-elements", type=positive_int, default=4096)
    verify.add_argument("--output", type=Path, required=True)
    verify.set_defaults(func=run_verify)

    benchmark = commands.add_parser("benchmark")
    add_common_case_arguments(benchmark)
    benchmark.add_argument("--original", type=Path, required=True)
    benchmark.add_argument("--parent", type=Path, required=True)
    benchmark.add_argument("--candidate", type=Path, required=True)
    benchmark.add_argument("--oracle-evidence", type=Path, required=True)
    benchmark.add_argument("--warmups", type=positive_int, default=20)
    benchmark.add_argument("--samples", type=positive_int, default=50)
    benchmark.add_argument("--sample-ms", type=positive_float, default=100.0)
    benchmark.add_argument("--pilot-repetitions", type=positive_int, default=20)
    benchmark.add_argument("--repetitions", type=positive_int, required=True)
    benchmark.add_argument("--timing-seed", type=int, default=20260917)
    benchmark.add_argument("--order-seed", type=int, default=20260917)
    benchmark.add_argument("--include-torch", action="store_true")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.set_defaults(func=run_benchmark)

    inspect = commands.add_parser("inspect")
    inspect.add_argument("--artifact", type=Path, required=True)
    inspect.add_argument("--architecture", required=True)
    inspect.add_argument("--output-dir", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.set_defaults(func=run_inspect)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--contract", type=Path, required=True)
    evaluate.add_argument("--candidate-id", required=True)
    evaluate.add_argument("--parent-id", required=True)
    evaluate.add_argument("--proposal", type=Path, required=True)
    evaluate.add_argument("--original", type=Path, required=True)
    evaluate.add_argument("--parent", type=Path, required=True)
    evaluate.add_argument("--candidate", type=Path, required=True)
    evaluate.add_argument("--build-evidence", type=Path, required=True)
    evaluate.add_argument("--oracle-evidence", type=Path, required=True)
    evaluate.add_argument("--timing-evidence", type=Path, required=True)
    evaluate.add_argument("--static-evidence", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(func=run_evaluate)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
