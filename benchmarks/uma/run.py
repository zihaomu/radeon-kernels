#!/usr/bin/env python3
"""Reproduce HIP UMA handoff checks, bandwidth samples, and optional traces.

python3 benchmarks/uma/run.py --trace --output build/uma/results.json
All GPU measurements run sequentially. No root privileges or driver changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("zero_copy.hip")
MODES = ("mapped", "mapped_coherent", "host_noncoherent", "registered", "managed", "device")
SOURCES = (
    "https://docs.kernel.org/gpu/amdgpu/driver-core.html#memory-domains",
    "https://docs.kernel.org/gpu/amdgpu/module-parameters.html",
    "https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/hip_runtime_api/memory_management/host_memory.html",
    "https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/how-to/hip_runtime_api/memory_management/coherence_control.html",
    "https://github.com/ROCm/clr/blob/rocm-7.2.1/hipamd/src/hip_memory.cpp",
)


def command(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=timeout)


def save(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, path)


def parse_probe(proc: subprocess.CompletedProcess[str]) -> dict:
    for line in reversed(proc.stdout.splitlines()):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "passed" in record:
            record["returncode"] = proc.returncode
            if proc.stderr:
                record["stderr"] = proc.stderr
            return record
    return {"passed": False, "returncode": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr}


def csv_records(directory: Path, suffix: str) -> list[dict]:
    rows = []
    for path in sorted(directory.rglob(f"*{suffix}.csv")):
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def trace_probe(binary: Path, mode: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    args = ["rocprofv3", "--sys-trace", "--output-format", "csv",
            "--output-directory", str(directory), "--output-file", mode,
            "--", str(binary), mode, "1024", "3"]
    proc = command(args)
    # Raw traces stay under the ignored build/ directory by default.
    (directory / "stdout.log").write_text(proc.stdout)
    (directory / "stderr.log").write_text(proc.stderr)
    kernels = csv_records(directory, "kernel_trace")
    copies = csv_records(directory, "memory_copy_trace")
    hip = csv_records(directory, "hip_api_trace")
    hsa = csv_records(directory, "hsa_api_trace")
    allocations = csv_records(directory, "memory_allocation_trace")
    agents = csv_records(directory, "agent_info")
    names = [row.get("Function", row.get("Name", "")) for row in hip]
    explicit = [name for name in names if name.startswith("hipMemcpy")]
    hsa_copy_names = [row.get("Function", "") for row in hsa
                      if row.get("Function", "").startswith(("hsa_memory_copy", "hsa_amd_memory_async_copy"))]
    probe = parse_probe(proc)
    captured = proc.returncode == 0 and bool(kernels) and bool(hip) and bool(hsa)
    kernel_ok = len(kernels) == 7 and all("affine_kernel" in row.get("Kernel_Name", "") for row in kernels)
    copy_ok = (bool(copies) and len(explicit) == 4 and bool(hsa_copy_names)) if mode == "device" else (
        not copies and not explicit and not hsa_copy_names)
    return {
        "command": args, "probe": probe,
        "kernel_dispatch_count": len(kernels), "memory_copy_count": len(copies),
        "hip_api_count": len(hip), "hsa_api_count": len(hsa),
        "hip_memcpy_api_count": len(explicit),
        "hsa_copy_api_count": len(hsa_copy_names),
        "copy_directions": dict(Counter(row["Direction"] for row in copies)),
        "memory_copy_records": copies, "memory_allocation_records": allocations,
        "agents": agents,
        "kernel_names": sorted({row.get("Kernel_Name", "") for row in kernels}),
        "trace_captured": captured,
        "expected_behavior_verified": captured and kernel_ok and copy_ok and probe["passed"],
        "raw_directory": str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory),
    }


def compact_summary(result: dict) -> dict:
    """Keep reproducible measurements in the branch without process IDs/addresses."""
    fields = ("mode", "bytes", "samples", "warmups", "allocation_and_registration_ms",
              "median_kernel_ms", "effective_read_write_GB_s", "kernel_ms", "explicit_copy_bytes",
              "errors_first_handoff", "errors_second_handoff", "elements_checked_per_handoff", "passed")
    summary = {key: result[key] for key in (
        "schema_version", "recorded_at_utc", "base_revision", "kernel", "hipcc_version",
        "source_sha256", "binary_sha256", "driver_configuration", "sources", "method",
        "bandwidth_definition", "all_correct", "traces_captured", "trace_expectations_verified") if key in result}
    if result["tests"] and "device" in result["tests"][0]:
        summary["device"] = result["tests"][0]["device"]
    summary["tests"] = []
    for row in result["tests"]:
        record = {key: row[key] for key in fields if key in row}
        record["pointer"] = {key: value for key, value in row.get("pointer", {}).items()
                             if key not in ("host", "gpu")}
        record["snapshots"] = row.get("snapshots", {})
        summary["tests"].append(record)
    summary["traces"] = []
    for trace in result["traces"]:
        record = {key: trace[key] for key in (
            "kernel_dispatch_count", "memory_copy_count", "hip_memcpy_api_count", "hsa_copy_api_count",
            "copy_directions", "kernel_names", "expected_behavior_verified")}
        record["mode"] = trace["probe"].get("mode")
        record["passed"] = trace["probe"]["passed"]
        agents = {f"Agent {row['Logical_Node_Id']}": row for row in trace["agents"]}
        record["target_buffer_allocations"] = [
            {"agent_type": agents.get(row["Agent_Id"], {}).get("Agent_Type"),
             "allocation_bytes": int(row["Allocation_Size"])}
            for row in trace["memory_allocation_records"]
            if int(row["Allocation_Size"]) == 1024**3]
        summary["traces"].append(record)
        gpu = next((row for row in trace["agents"] if row["Agent_Type"] == "GPU"), None)
        if gpu:
            summary["gpu_compute_units"] = int(gpu["Cu_Count"])
        cpu = next((row for row in trace["agents"] if row["Agent_Type"] == "CPU"), None)
        if cpu:
            summary["processor"] = cpu["Name"]
    device = summary.get("device", {})
    largest = max((row.get("bytes", 0) for row in result["tests"] if row["passed"]), default=0)
    summary["interpretation"] = [
        f"All buffer handoff checks passed={result['all_correct']}; BIOS VRAM carve-out={device.get('vram_total_bytes')} bytes.",
        f"Driver GTT budget={device.get('gtt_total_bytes')} bytes; HIP reported allocation total={device.get('hip_total_bytes')} bytes; largest validated buffer={largest} bytes.",
        f"Trace expectations verified={result.get('trace_expectations_verified')}: host/registered paths require GPU dispatches with zero data copies, and the device-buffer control requires copies.",
        "Separate CPU/device buffers plus hipMemcpy still copy data, even when both allocations reside in system RAM on an APU.",
        "Pointer equality alone is insufficient evidence of zero copy; CPU handoff correctness and traces were checked together.",
        "Pinned/registered host pages are not necessarily included in the mem_info_gtt_used TTM counter; RSS and allocation-agent traces also matter.",
        f"GPU pageable-memory access capability={device.get('pageable_memory_access')}; when this is 0, existing CPU buffers should be registered before passing them to a GPU kernel.",
        "CPU and GPU take turns accessing the buffer, with system-scope synchronization; concurrent CPU/GPU writes were not tested.",
        "The timings measure a streaming integer kernel, excluding allocation, CPU initialization, CPU verification, and explicit copies; no GEMM/LLM speed claim is made.",
    ]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/uma/results.json")
    parser.add_argument("--summary-output", type=Path,
                        help="optional compact measurement artifact, omitting process IDs and pointer addresses")
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--large-mib", type=int, default=4096,
                        help="additional host and device capacity probe; 0 disables")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--arch", default="gfx1151")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if not shutil.which("hipcc"):
        parser.error("hipcc is required")
    if args.trace and not shutil.which("rocprofv3"):
        parser.error("--trace requires rocprofv3")
    if not 1 <= args.samples <= 1000:
        parser.error("--samples must be 1..1000")
    if any(not 1 <= size <= 8192 for size in args.sizes_mib) or not 0 <= args.large_mib <= 8192:
        parser.error("probe sizes must be 1..8192 MiB (large-mib may be 0)")
    output = args.output.resolve()
    binary = ROOT / "build/uma/zero_copy"
    binary.parent.mkdir(parents=True, exist_ok=True)
    build_args = ["hipcc", "-O3", "-std=c++17", f"--offload-arch={args.arch}",
                  str(SOURCE), "-o", str(binary)]
    print("Building standalone HIP probe", flush=True)
    built = command(build_args)
    if built.returncode:
        sys.stderr.write(built.stdout + built.stderr)
        return built.returncode
    result = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": command(["git", "branch", "--show-current"]).stdout.strip(),
        "base_revision": command(["git", "rev-parse", "HEAD"]).stdout.strip(),
        "kernel": platform.release(), "hipcc_version": command(["hipcc", "--version"]).stdout,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "build_command": build_args,
        "relevant_environment": {key: value for key, value in os.environ.items()
                                 if key.startswith(("HSA_", "HIP_", "ROCR_"))},
        "sources": SOURCES,
        "method": "uint32 affine update on GPU; full CPU validation at two synchronized handoffs",
        "bandwidth_definition": "logical read+write bytes / median event time; decimal GB/s; not measured DRAM traffic",
        "tests": [], "traces": [],
    }
    for path in (Path("/sys/module/ttm/parameters/pages_limit"), Path("/etc/modprobe.d/ttm.conf")):
        if path.is_file():
            result.setdefault("driver_configuration", {})[str(path)] = path.read_text().strip()
    jobs = [(mode, size) for size in args.sizes_mib for mode in MODES]
    if args.large_mib:
        jobs.extend((mode, args.large_mib) for mode in ("host_noncoherent", "device"))
    for mode, size in jobs:
        record = parse_probe(command([str(binary), mode, str(size), str(args.samples)]))
        record.update(requested_mode=mode, requested_mib=size)
        result["tests"].append(record)
        save(output, result)
        speed = record.get("effective_read_write_GB_s", 0)
        print(f"{mode:20} {size:5} MiB: passed={record['passed']} {speed:7.1f} GB/s", flush=True)
    if args.trace:
        for mode in ("host_noncoherent", "registered", "device"):
            trace = trace_probe(binary, mode, binary.parent / "traces" / mode / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
            result["traces"].append(trace)
            save(output, result)
            print(f"trace {mode}: {trace['kernel_dispatch_count']} kernels, "
                  f"{trace['memory_copy_count']} memory copies, "
                  f"{trace['hip_memcpy_api_count']} HIP memcpy calls", flush=True)
    result["all_correct"] = all(row["passed"] and row["returncode"] == 0 for row in result["tests"])
    result["traces_captured"] = all(row["trace_captured"] for row in result["traces"]) if args.trace else None
    result["trace_expectations_verified"] = all(row["expected_behavior_verified"] for row in result["traces"]) if args.trace else None
    save(output, result)
    if args.summary_output:
        save(args.summary_output.resolve(), compact_summary(result))
    print(f"Results: {output}", flush=True)
    return 0 if result["all_correct"] and result["trace_expectations_verified"] is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
