from __future__ import annotations

from copy import deepcopy


IMAGE = "registry.example.com/rocm-triton@sha256:" + "a" * 64


def target(target_id: str = "lab-a", ssh_host: str = "lab-a-ssh") -> dict:
    return {
        "id": target_id,
        "ssh_host": ssh_host,
        "expected_architecture": "auto",
        "remote_root": f"/data/{target_id}/radeon-kernels-workspace",
        "gpu_ids": [0, 1],
        "gpus_per_job": 1,
        "max_parallel_jobs": 2,
        "busy_policy": "wait",
        "busy_timeout_seconds": 60,
        "container": {
            "runtime": "docker",
            "image": IMAGE,
            "workdir": "/workspace/radeon-kernels",
            "devices": ["/dev/kfd", "/dev/dri"],
            "security_options": ["seccomp=unconfined"],
            "mounts": [],
            "environment_from_host": [],
        },
    }


def targets_payload(*targets: dict) -> dict:
    return {"schema_version": 1, "targets": [deepcopy(item) for item in targets]}

