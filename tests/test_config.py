from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from radeon_kernels.errors import ConfigError
from radeon_kernels.lab.config import load_targets
from tests.helpers import target, targets_payload


class TargetConfigTests(unittest.TestCase):
    def write_config(self, payload: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "targets.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_loads_every_listed_target_in_order(self) -> None:
        path = self.write_config(
            targets_payload(target("first", "first-ssh"), target("second", "second-ssh"))
        )

        config = load_targets(path)

        self.assertEqual([item.id for item in config.targets], ["first", "second"])
        self.assertEqual(config.targets[0].gpu_ids, (0, 1))

    def test_rejects_unknown_fields_with_precise_path(self) -> None:
        item = target()
        item["enabled"] = True
        path = self.write_config(targets_payload(item))

        with self.assertRaisesRegex(ConfigError, r"targets\[0\].*enabled"):
            load_targets(path)

    def test_rejects_duplicate_ssh_aliases(self) -> None:
        path = self.write_config(
            targets_payload(target("first", "same"), target("second", "same"))
        )

        with self.assertRaisesRegex(ConfigError, "ssh_host aliases must be unique"):
            load_targets(path)

    def test_rejects_shell_like_ssh_alias(self) -> None:
        item = target()
        item["ssh_host"] = "-oProxyCommand=bad"
        path = self.write_config(targets_payload(item))

        with self.assertRaisesRegex(ConfigError, "SSH config alias"):
            load_targets(path)

    def test_rejects_root_as_remote_workspace(self) -> None:
        item = target()
        item["remote_root"] = "/"
        path = self.write_config(targets_payload(item))

        with self.assertRaisesRegex(ConfigError, "non-root absolute path"):
            load_targets(path)

    def test_rejects_mutable_image_tag(self) -> None:
        item = target()
        item["container"]["image"] = "rocm/triton:latest"
        path = self.write_config(targets_payload(item))

        with self.assertRaisesRegex(ConfigError, "immutable image reference"):
            load_targets(path)

    def test_rejects_gpu_oversubscription(self) -> None:
        item = target()
        item["gpus_per_job"] = 2
        item["max_parallel_jobs"] = 2
        path = self.write_config(targets_payload(item))

        with self.assertRaisesRegex(ConfigError, "oversubscribe"):
            load_targets(path)


if __name__ == "__main__":
    unittest.main()
