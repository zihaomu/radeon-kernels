from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from radeon_kernels.runtime.signing import TrustedKeyring, verify_detached_file


OPERATORS = ("rms_norm", "add_rms_norm", "rope", "swiglu", "softmax")


class MilestoneFivePublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).parents[1]
        cls.keyring = TrustedKeyring.from_file(cls.root / "keys" / "trusted-keys.json")

    def validate_signed(self, directory: str, filename: str, schema_name: str) -> dict:
        path = self.root / directory / filename
        verify_detached_file(
            path,
            path.with_suffix(".sig.json"),
            self.keyring,
            signed_file=path.name,
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        schema = json.loads(
            (self.root / "schemas" / f"{schema_name}.schema.json").read_text(encoding="utf-8")
        )
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
        return value

    def test_each_operator_index_has_both_signed_architecture_releases(self) -> None:
        for operator in OPERATORS:
            with self.subTest(operator=operator):
                index = self.validate_signed(
                    "pack-index", f"{operator}-1.0.json", "pack-index"
                )
                self.assertEqual(
                    {pack["environment"]["architecture"] for pack in index["packs"]},
                    {"gfx1151", "gfx1201"},
                )
                self.assertTrue(all(len(pack["artifacts"]) == 1 for pack in index["packs"]))

    def test_all_milestone_five_evidence_and_approvals_are_signed(self) -> None:
        evidence_paths = sorted((self.root / "evidence").glob("m5-*.json"))
        approval_paths = sorted(
            path
            for path in (self.root / "approvals").glob("*.json")
            if "proposal-m5-" in path.name and not path.name.endswith(".sig.json")
        )
        evidence_paths = [path for path in evidence_paths if not path.name.endswith(".sig.json")]

        self.assertEqual(len(evidence_paths), 10)
        self.assertEqual(len(approval_paths), 10)
        for path in evidence_paths:
            self.validate_signed("evidence", path.name, "evidence")
        for path in approval_paths:
            self.validate_signed("approvals", path.name, "approval")


if __name__ == "__main__":
    unittest.main()
