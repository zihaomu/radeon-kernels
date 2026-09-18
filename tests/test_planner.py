from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from radeon_kernels.errors import ConfigError
from radeon_kernels.lab.config import load_targets
from radeon_kernels.lab.planner import build_plan
from tests.helpers import target, targets_payload


class PlannerTests(unittest.TestCase):
    def config(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "targets.yaml"
        payload = targets_payload(
            target("first", "first-ssh"),
            target("second", "second-ssh"),
        )
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return load_targets(path)

    def test_default_plan_contains_exactly_every_listed_target(self) -> None:
        plan = build_plan(self.config())

        self.assertEqual(
            [target.target_id for target in plan.targets],
            ["first", "second"],
        )

    def test_selector_preserves_configuration_order(self) -> None:
        plan = build_plan(self.config(), ["second", "first"])

        self.assertEqual(
            [target.target_id for target in plan.targets],
            ["first", "second"],
        )

    def test_rejects_unconfigured_target(self) -> None:
        with self.assertRaisesRegex(ConfigError, "unknown target.*third"):
            build_plan(self.config(), ["third"])


if __name__ == "__main__":
    unittest.main()

