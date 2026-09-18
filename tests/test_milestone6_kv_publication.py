from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from radeon_kernels.runtime.signing import TrustedKeyring, verify_detached_file


EXPECTED_COUNTS = {"kv_cache_append": 14, "kv_cache_copy": 19}


class MilestoneSixKvPublicationTests(unittest.TestCase):
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
            (self.root / "schemas" / f"{schema_name}.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
        return value

    def test_signed_indexes_and_compatibility_cover_all_winners(self) -> None:
        for operator, expected_count in EXPECTED_COUNTS.items():
            with self.subTest(operator=operator):
                index = self.validate_signed(
                    "pack-index", f"{operator}-1.0.json", "pack-index"
                )
                compatibility = self.validate_signed(
                    "compatibility", f"{operator}-1.0.json", "compatibility"
                )
                self.assertEqual(len(index["packs"]), expected_count)
                self.assertEqual(len(compatibility["records"]), expected_count)
                self.assertEqual(
                    len({pack["pack_id"] for pack in index["packs"]}), expected_count
                )

    def test_dispatch_evidence_approval_and_index_are_linked(self) -> None:
        for operator, expected_count in EXPECTED_COUNTS.items():
            dispatch = json.loads(
                (self.root / "dispatch" / f"{operator}-1.0.json").read_text(
                    encoding="utf-8"
                )
            )
            index = self.validate_signed(
                "pack-index", f"{operator}-1.0.json", "pack-index"
            )
            indexed = {
                artifact["evidence_id"]: (artifact, pack["approval_id"])
                for pack in index["packs"]
                for artifact in pack["artifacts"]
            }
            self.assertEqual(len(dispatch["entries"]), expected_count)
            self.assertEqual(
                {entry["evidence_id"] for entry in dispatch["entries"]}, set(indexed)
            )
            for entry in dispatch["entries"]:
                evidence_id = entry["evidence_id"]
                artifact, approval_id = indexed[evidence_id]
                evidence = self.validate_signed(
                    "evidence", f"{evidence_id}.json", "evidence"
                )
                approval = self.validate_signed(
                    "approvals", f"{approval_id}.json", "approval"
                )
                self.assertEqual(evidence["approval_id"], approval_id)
                self.assertEqual(approval["decision"], "approved")
                self.assertEqual(
                    entry["winner"],
                    {
                        key: artifact[key]
                        for key in ("artifact", "entrypoint", "format", "provider", "sha256")
                    },
                )


if __name__ == "__main__":
    unittest.main()
