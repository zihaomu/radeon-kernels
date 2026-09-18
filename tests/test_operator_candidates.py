from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from developer.promotion.create_operator_proposal import create_proposal
from developer.promotion.create_operator_proposals import create_all
from developer.promotion.prepare_operator_candidates import prepare
from radeon_kernels.runtime.pack import KernelPack


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class OperatorCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifact = self.root / "gemv.so"
        self.artifact.write_bytes(b"gemv candidate")
        digest = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        self.dispatch = self.root / "gemv-1.0.json"
        self.public_dispatch = {
            "schema_version": "1.0",
            "operator": "gemv",
            "semantic_version": "1.0",
            "default_fallbacks": ["torch"],
            "entries": [],
        }
        write_json(self.dispatch, self.public_dispatch)
        self.evaluation = self.root / "evaluation.json"
        self.evaluation_value = {
            "captured_at": "2026-09-18T00:00:00+00:00",
            "environment": {
                "architecture": "gfx1201",
                "wavefront_size": 32,
                "rocm": "7.2.53211",
                "python": "3.12.3",
                "pytorch": "2.11.0+rocm7.2",
            },
            "candidate": {
                "artifact_sha256": digest,
                "source_sha256": "1" * 64,
                "entrypoints": {"gemv_wave1": "gemv_wave1", "gemv_wave2": "gemv_wave2"},
            },
            "method": {"oracle": "torch_fp32_reference", "baseline": "torch_mm"},
            "results": [
                self.result("decode-a", "fp16", "gemv_wave1", 12.0),
                self.result("decode-b", "bf16", "gemv_wave2", 18.0),
                {
                    **self.result("decode-c", "fp16", "gemv_wave1", 1.0),
                    "decision": "negative_knowledge",
                    "gates": {"correctness_passed": True, "minimum_improvement": False},
                },
            ],
        }
        write_json(self.evaluation, self.evaluation_value)

    @staticmethod
    def result(case_id: str, dtype: str, entrypoint: str, improvement: float) -> dict:
        return {
            "operator": "gemv",
            "case_id": case_id,
            "dtype": dtype,
            "shape": {"m": 1, "n": 4096, "k": 4096, "weight_layout": "nk"},
            "selected_entrypoint": entrypoint,
            "decision": "promote",
            "gates": {"correctness_passed": True, "minimum_improvement": True},
            "correctness": [
                {"seed": 1, "passed": True, "max_absolute_error": 0.0},
                {"seed": 2, "passed": True, "max_absolute_error": 0.0},
            ],
            "timing": {
                "latency_reduction_pct": improvement,
                "candidate": {
                    "samples_ms": [0.8] * 30,
                    "median_ms": 0.8,
                    "coefficient_of_variation": 0.0,
                },
                "baseline": {
                    "samples_ms": [1.0] * 30,
                    "median_ms": 1.0,
                    "coefficient_of_variation": 0.0,
                },
            },
        }

    def run_prepare(self) -> dict:
        return prepare(
            evaluation_path=self.evaluation,
            artifact_path=self.artifact,
            dispatch_path=self.dispatch,
            staged_dispatch_path=self.root / "staged-gemv.json",
            pack_root=self.root / "packs",
            run_label="m6-test",
        )

    def test_stages_every_promotable_result_without_mutating_public_dispatch(self) -> None:
        receipt = self.run_prepare()

        self.assertEqual(json.loads(self.dispatch.read_text(encoding="utf-8")), self.public_dispatch)
        staged = json.loads((self.root / "staged-gemv.json").read_text(encoding="utf-8"))
        self.assertEqual(len(staged["entries"]), 2)
        self.assertEqual(
            [item["entrypoint"] for item in receipt["packs"]],
            ["gemv_wave1", "gemv_wave2"],
        )
        for item in receipt["packs"]:
            pack = KernelPack.from_directory(Path(item["pack_path"]), verify_artifacts=True)
            self.assertEqual(pack.get("gemv", "1.0").artifact.entrypoint, item["entrypoint"])

    def test_rejects_promote_result_with_failed_worker_gate(self) -> None:
        self.evaluation_value["results"][0]["gates"]["minimum_improvement"] = False
        write_json(self.evaluation, self.evaluation_value)

        with self.assertRaisesRegex(ValueError, "did not pass every worker gate"):
            self.run_prepare()
        self.assertFalse((self.root / "staged-gemv.json").exists())
        self.assertFalse((self.root / "packs").exists())

    def test_proposal_preserves_declared_oracle_and_baseline(self) -> None:
        receipt = self.run_prepare()
        item = receipt["packs"][0]

        proposal = create_proposal(
            evaluation_path=self.evaluation,
            pack_path=Path(item["pack_path"]),
            dispatch_path=self.root / "staged-gemv.json",
            entry_id=item["entry_id"],
        )

        self.assertEqual(proposal["recommended_decision"], "promote")
        self.assertEqual(proposal["public_evidence"]["correctness"]["oracle"], "torch_fp32_reference")
        self.assertEqual(proposal["public_evidence"]["timing"]["baseline"], "torch_mm")

    def test_batch_proposals_cover_every_staged_pack_and_refuse_overwrite(self) -> None:
        receipt = self.run_prepare()
        receipt_path = self.root / "receipt.json"
        write_json(receipt_path, receipt)
        output_directory = self.root / "proposals"

        summary = create_all(
            evaluation_path=self.evaluation,
            receipt_path=receipt_path,
            dispatch_path=self.root / "staged-gemv.json",
            output_directory=output_directory,
        )

        self.assertEqual(len(summary["proposals"]), 2)
        self.assertTrue(
            all(item["recommended_decision"] == "promote" for item in summary["proposals"])
        )
        self.assertEqual(len(list(output_directory.glob("*.json"))), 2)
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite proposal"):
            create_all(
                evaluation_path=self.evaluation,
                receipt_path=receipt_path,
                dispatch_path=self.root / "staged-gemv.json",
                output_directory=output_directory,
            )


if __name__ == "__main__":
    unittest.main()
