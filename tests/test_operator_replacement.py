from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from developer.promotion.prepare_operator_replacement import prepare
from radeon_kernels.runtime.pack import KernelPack


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class OperatorReplacementTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifact = self.root / "online-softmax.so"
        self.artifact.write_bytes(b"online softmax artifact")
        self.digest = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        self.entry_id = "softmax-gfx1151-attention-w8192-fp16"
        self.dispatch = self.root / "softmax-1.0.json"
        write_json(
            self.dispatch,
            {
                "schema_version": "1.0",
                "operator": "softmax",
                "semantic_version": "1.0",
                "default_fallbacks": ["torch"],
                "entries": [
                    {
                        "id": self.entry_id,
                        "priority": 100,
                        "environment": {
                            "architecture": "gfx1151",
                            "wavefront_size": 32,
                            "rocm_abi": "7.15",
                            "python_abi": "cp312",
                            "pytorch_version": "2.14.0+rocm7.15",
                        },
                        "workload": {
                            "dtype": "fp16",
                            "rows": 512,
                            "width": 8192,
                            "contiguous": True,
                            "requires_grad": False,
                        },
                        "winner": {
                            "provider": "native",
                            "artifact": "lib/three-pass.so",
                            "entrypoint": "softmax",
                            "sha256": "0" * 64,
                            "format": "python_extension",
                        },
                        "launch": {},
                        "fallbacks": ["torch"],
                        "evidence_id": "m5-three-pass-evidence",
                    }
                ],
            },
        )
        self.evaluation = self.root / "evaluation.json"
        self.evaluation_value = {
            "environment": {
                "architecture": "gfx1151",
                "wavefront_size": 32,
                "rocm": "7.15.0",
                "python": "3.12.4",
                "pytorch": "2.14.0+rocm7.15",
            },
            "candidate": {
                "artifact_sha256": self.digest,
                "entrypoints": {"softmax": "softmax", "legacy": "softmax_three_pass"},
            },
            "results": [
                {
                    "operator": "softmax",
                    "case_id": "attention-w8192",
                    "dtype": "fp16",
                    "shape": {"rows": 512, "width": 8192},
                    "decision": "promote",
                    "gates": {"correctness_passed": True, "online_beats_three_pass": True},
                    "timing": {
                        "latency_reduction_pct": 80.0,
                        "latency_reduction_vs_three_pass_pct": 18.0,
                    },
                }
            ],
        }
        write_json(self.evaluation, self.evaluation_value)

    def run_prepare(self) -> dict:
        return prepare(
            evaluation_path=self.evaluation,
            artifact_path=self.artifact,
            dispatch_path=self.dispatch,
            staged_dispatch_path=self.root / "staged-softmax.json",
            pack_path=self.root / "pack",
            entry_id=self.entry_id,
            pack_id="rk-softmax-gfx1151-online-test",
            pack_version="0.1.0",
            run_label="online-test",
        )

    def test_stages_replacement_without_mutating_public_dispatch(self) -> None:
        receipt = self.run_prepare()

        original = json.loads(self.dispatch.read_text(encoding="utf-8"))
        staged = json.loads((self.root / "staged-softmax.json").read_text(encoding="utf-8"))
        self.assertEqual(original["entries"][0]["winner"]["sha256"], "0" * 64)
        self.assertEqual(staged["entries"][0]["winner"]["sha256"], self.digest)
        self.assertEqual(receipt["latency_reduction_vs_previous_pct"], 18.0)
        pack = KernelPack.from_directory(self.root / "pack", verify_artifacts=True)
        self.assertEqual(pack.artifacts[0].artifact.sha256, self.digest)

    def test_rejects_result_with_failed_worker_gate(self) -> None:
        self.evaluation_value["results"][0]["gates"]["online_beats_three_pass"] = False
        write_json(self.evaluation, self.evaluation_value)

        with self.assertRaisesRegex(ValueError, "did not pass every worker gate"):
            self.run_prepare()


if __name__ == "__main__":
    unittest.main()
