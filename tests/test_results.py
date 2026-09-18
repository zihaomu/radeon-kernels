from __future__ import annotations

import unittest

from radeon_kernels.errors import ConfigError
from radeon_kernels.lab.results import (
    RunStatus,
    TaskStatus,
    aggregate_run_status,
    sanitize_evidence,
)


class ResultTests(unittest.TestCase):
    def test_run_status_aggregation(self) -> None:
        self.assertEqual(
            aggregate_run_status([TaskStatus.SUCCEEDED]),
            RunStatus.SUCCEEDED,
        )
        self.assertEqual(
            aggregate_run_status([TaskStatus.SKIPPED]),
            RunStatus.PARTIAL,
        )
        self.assertEqual(
            aggregate_run_status([TaskStatus.FAILED]),
            RunStatus.FAILED,
        )
        self.assertEqual(
            aggregate_run_status([TaskStatus.SUCCEEDED, TaskStatus.FAILED]),
            RunStatus.PARTIAL,
        )
        self.assertEqual(
            aggregate_run_status([TaskStatus.SUCCEEDED], needs_review=True),
            RunStatus.NEEDS_REVIEW,
        )

    def private_evidence(self) -> dict:
        return {
            "schema_version": "1.0",
            "source_commit": "abc123",
            "candidate_content_hash": "def456",
            "image_digest": "sha256:" + "a" * 64,
            "operator": "gemm",
            "workload": {
                "dtype": "fp16",
                "layout": "nt",
                "m": 1024,
                "n": 1024,
                "k": 1024,
                "seed": 0,
                "private_path": "/secret/workload",
            },
            "gpu": {
                "architecture": "gfx1100",
                "model": "Example Radeon",
                "compute_units": 96,
                "wavefront_size": 32,
                "memory_capacity_bytes": 1,
                "features": ["matrix"],
                "uuid": "private-uuid",
                "pci_address": "0000:01:00.0",
            },
            "toolchain": {
                "driver": "1.0",
                "runtime": "rocm-1.0",
                "triton": "3.0",
                "registry": "private.example.com",
            },
            "candidate_id": "blocked",
            "configuration": {"block_m": 64, "num_warps": 8},
            "samples_ms": [1.0, 1.1],
            "statistics": {
                "median_ms": 1.05,
                "cv": 0.01,
                "sample_count": 2,
                "private_note": "host-a",
            },
            "ssh_host": "private-host",
            "remote_root": "/private/path",
        }

    def test_sanitizer_uses_nested_allowlists(self) -> None:
        public = sanitize_evidence(self.private_evidence())
        serialized = repr(public)

        self.assertNotIn("private-host", serialized)
        self.assertNotIn("private-uuid", serialized)
        self.assertNotIn("pci_address", serialized)
        self.assertNotIn("private_path", serialized)
        self.assertNotIn("registry", serialized)
        self.assertEqual(public["gpu"]["architecture"], "gfx1100")

    def test_sanitizer_rejects_path_like_candidate_configuration(self) -> None:
        private = self.private_evidence()
        private["configuration"]["cache"] = "/private/cache"

        with self.assertRaisesRegex(ConfigError, "path-like"):
            sanitize_evidence(private)


if __name__ == "__main__":
    unittest.main()

