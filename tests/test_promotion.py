from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from developer.promotion.build_kernel_pack import build
from developer.promotion.publish_approved import publish
from developer.promotion.record_approval import record_decision
from developer.promotion.signing import generate_private_key
from radeon_kernels.errors import ConfigError
from radeon_kernels.runtime.packs import KernelPackRegistry
from radeon_kernels.runtime.signing import (
    TrustedKeyring,
    sha256_file,
    verify_detached_file,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ApprovalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.proposal = Path(self.directory.name) / "proposal.json"

    def write_proposal(self, recommended: str) -> None:
        _write_json(
            self.proposal,
            {
                "proposal_id": "proposal-evidence-test-001",
                "recommended_decision": recommended,
            },
        )

    def test_approval_is_bound_to_exact_proposal_hash(self) -> None:
        self.write_proposal("promote")

        approval = record_decision(
            proposal_path=self.proposal,
            reviewer="release-reviewer",
            decision="approved",
            reason="all gates and sanitized evidence reviewed",
            reviewed_at="2026-09-17T12:00:00+00:00",
        )

        self.assertEqual(approval["proposal_sha256"], sha256_file(self.proposal))
        self.assertNotIn("override", approval)

    def test_needs_review_cannot_be_silently_approved(self) -> None:
        self.write_proposal("needs_review")

        with self.assertRaisesRegex(ConfigError, "explicit gate override"):
            record_decision(
                proposal_path=self.proposal,
                reviewer="release-reviewer",
                decision="approved",
                reason="not stable",
            )

    def test_gate_override_is_explicit_in_approval(self) -> None:
        self.write_proposal("needs_review")

        approval = record_decision(
            proposal_path=self.proposal,
            reviewer="release-reviewer",
            decision="approved",
            reason="accepted risk for a controlled preview",
            override_gates=True,
        )

        self.assertIs(approval["override"], True)


class PublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        source = self.root / "kernel.so"
        source.write_bytes(b"precompiled artifact")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.pack_path = self.root / "candidate-pack"
        self.pack = build(
            SimpleNamespace(
                source_artifact=source,
                expected_sha256=digest,
                output=self.pack_path,
                pack_id="rk-gemm-gfx1151-test",
                pack_version="0.1.0",
                architecture="gfx1151",
                wavefront_size=32,
                rocm_abi="7.15",
                python_abi="cp312",
                pytorch_version="2.14.0+rocm7.15",
                operator="gemm",
                semantic_version="1.0",
                provider="native",
                entrypoint="gemm_out",
                format="python_extension",
                evidence_id="evidence-gemm-test-001",
            )
        )
        (self.pack_path / "verification.json").write_text("private", encoding="utf-8")
        environment = {
            "architecture": "gfx1151",
            "wavefront_size": 32,
            "rocm_abi": "7.15",
            "python_abi": "cp312",
            "pytorch_version": "2.14.0+rocm7.15",
        }
        artifact = {
            "provider": "native",
            "artifact": "lib/kernel.so",
            "entrypoint": "gemm_out",
            "sha256": digest,
            "format": "python_extension",
        }
        self.entry = {
            "id": "gemm-gfx1151-test",
            "priority": 100,
            "environment": environment,
            "workload": {
                "dtype": "fp16",
                "layout": "nn",
                "m": [16, 16],
                "n": [16, 16],
                "k": [16, 16],
                "contiguous": True,
                "requires_grad": False,
            },
            "winner": artifact,
            "launch": {"variant": 1},
            "fallbacks": ["torch"],
            "evidence_id": "evidence-gemm-test-001",
        }
        self.dispatch_path = self.root / "dispatch" / "gemm-1.0.json"
        _write_json(
            self.dispatch_path,
            {
                "schema_version": "1.0",
                "operator": "gemm",
                "semantic_version": "1.0",
                "default_fallbacks": ["torch"],
                "entries": [self.entry],
            },
        )
        pack_artifact = {
            **artifact,
            "operator": "gemm",
            "semantic_version": "1.0",
            "evidence_id": "evidence-gemm-test-001",
        }
        self.proposal_path = self.root / "proposal.json"
        self.proposal = {
            "proposal_id": "proposal-evidence-gemm-test-001",
            "recommended_decision": "promote",
            "gates": {"correctness_passed": True, "timing_stable": True},
            "operator": "gemm",
            "semantic_version": "1.0",
            "entry_id": self.entry["id"],
            "source_evidence_hashes": {"dispatch": sha256_file(self.dispatch_path)},
            "dispatch_entry": self.entry,
            "pack": {
                "pack_id": self.pack.pack_id,
                "pack_version": self.pack.pack_version,
                "manifest_sha256": sha256_file(self.pack_path / "manifest.json"),
                "environment": environment,
                "artifacts": [pack_artifact],
            },
            "public_evidence": {
                "schema_version": "1.0",
                "evidence_id": "evidence-gemm-test-001",
                "operator": "gemm",
                "semantic_version": "1.0",
                "candidate_content_hash": digest,
                "captured_at": "2026-09-17T11:00:00+00:00",
                "environment": environment,
                "workload": {
                    "dtype": "fp16",
                    "layout": "nn",
                    "m": 16,
                    "n": 16,
                    "k": 16,
                    "variant": 1,
                    "contiguous": True,
                    "requires_grad": False,
                },
                "correctness": {
                    "passed": True,
                    "oracle": "torch.mm",
                    "cases": [
                        {
                            "case_id": "fp16-m16-n16-k16-seed1",
                            "runtime_ok": True,
                            "finite": True,
                            "guards_intact": True,
                            "cosine_similarity": 1.0,
                            "max_absolute_error": 0.0,
                        }
                    ],
                },
                "timing": {
                    "baseline": "torch.mm",
                    "candidate_samples_ms": [1.0, 1.0],
                    "baseline_samples_ms": [2.0, 2.0],
                    "candidate_median_ms": 1.0,
                    "baseline_median_ms": 2.0,
                    "candidate_cv": 0.0,
                    "baseline_cv": 0.0,
                    "latency_reduction_pct": 50.0,
                    "confidence_interval_pct": [49.0, 51.0],
                    "stable": True,
                },
                "source_evidence_hashes": {"dispatch": sha256_file(self.dispatch_path)},
                "decision": "promote",
                "guarantee_boundary": "Synthetic exact test environment only.",
            },
            "compatibility_record": {
                "id": "gemm-gfx1151-evidence-gemm-test-001",
                **environment,
                "provider": "native",
                "status": "supported",
                "evidence_id": "evidence-gemm-test-001",
                "reason": "all gates passed",
            },
        }
        _write_json(self.proposal_path, self.proposal)
        self.approval_path = self.root / "approval.json"
        _write_json(
            self.approval_path,
            record_decision(
                proposal_path=self.proposal_path,
                reviewer="release-reviewer",
                decision="approved",
                reason="reviewed exact proposal and artifact lineage",
                reviewed_at="2026-09-17T12:00:00+00:00",
            ),
        )
        self.private_key = self.root / "release-key.json"
        public_record = generate_private_key(self.private_key, "test-release-key")
        self.trusted_keys = self.root / "trusted-keys.json"
        _write_json(self.trusted_keys, {"schema_version": "1.0", "keys": [public_record]})
        self.public_root = self.root / "public"
        self.release_root = self.root / "releases"

    def publish(self) -> dict:
        return publish(
            proposal_path=self.proposal_path,
            approval_path=self.approval_path,
            pack_path=self.pack_path,
            dispatch_path=self.dispatch_path,
            private_key_path=self.private_key,
            trusted_keys_path=self.trusted_keys,
            public_root=self.public_root,
            release_root=self.release_root,
        )

    def test_publishes_minimal_signed_pack_and_public_chain(self) -> None:
        dispatch_sha256 = sha256_file(self.dispatch_path)
        receipt = self.publish()
        archive = self.release_root / receipt["release_file"]

        with tarfile.open(archive, "r:gz") as handle:
            members = {item.name: item for item in handle.getmembers()}
            names = set(members)
            self.assertIn(f"{self.pack.pack_id}/manifest.sig.json", names)
            self.assertIn(f"{self.pack.pack_id}/lib/kernel.so", names)
            self.assertFalse(any("verification.json" in name for name in names))
            self.assertEqual(members[f"{self.pack.pack_id}/manifest.json"].mode, 0o644)
            self.assertEqual(members[f"{self.pack.pack_id}/lib/kernel.so"].mode, 0o755)
            handle.extractall(self.root / "extracted", filter="data")

        keyring = TrustedKeyring.from_file(self.trusted_keys)
        registry = KernelPackRegistry.discover(
            [self.root / "extracted"],
            trusted_keys=keyring,
        )
        self.assertEqual([item.pack_id for item in registry.packs], [self.pack.pack_id])
        self.assertEqual(registry.rejected, ())

        index = self.public_root / "pack-index" / "gemm-1.0.json"
        verify_detached_file(index, index.with_suffix(".sig.json"), keyring)
        index_value = json.loads(index.read_text(encoding="utf-8"))
        self.assertEqual(index_value["packs"][0]["approval_id"], receipt["approval_id"])
        self.assertEqual(
            index_value["packs"][0]["release_sha256"],
            sha256_file(archive),
        )
        self.assertEqual(sha256_file(self.dispatch_path), dispatch_sha256)
        self.assertFalse((self.public_root / "dispatch").exists())

    def test_release_archive_is_reproducible(self) -> None:
        first = self.publish()
        second = publish(
            proposal_path=self.proposal_path,
            approval_path=self.approval_path,
            pack_path=self.pack_path,
            dispatch_path=self.dispatch_path,
            private_key_path=self.private_key,
            trusted_keys_path=self.trusted_keys,
            public_root=self.root / "second-public",
            release_root=self.root / "second-releases",
        )

        self.assertEqual(first["release_sha256"], second["release_sha256"])

    def test_rejects_approval_after_proposal_is_tampered(self) -> None:
        self.proposal["gates"]["late_change"] = True
        _write_json(self.proposal_path, self.proposal)

        with self.assertRaisesRegex(ConfigError, "does not match proposal content"):
            self.publish()


if __name__ == "__main__":
    unittest.main()
