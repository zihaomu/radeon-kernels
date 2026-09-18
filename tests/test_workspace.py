from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from radeon_kernels.errors import ConfigError
from radeon_kernels.workspace import find_workspace_file, load_workspace


class WorkspaceTests(unittest.TestCase):
    def make_workspace(self) -> tuple[Path, Path]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        config_file = root / ".radeon-workspace.yaml"
        payload = {
            "schema_version": 1,
            "project_dir": "./radeon-kernels",
            "private_dir": "./lab-private",
            "inventory": {"targets": "./lab-private/config/targets.yaml"},
            "storage": {
                "runs": "./lab-private/runs",
                "state": "./lab-private/state",
                "cache": "./lab-private/cache",
                "exports": "./lab-private/exports",
            },
        }
        config_file.write_text(yaml.safe_dump(payload), encoding="utf-8")
        nested = root / "radeon-kernels" / "src"
        nested.mkdir(parents=True)
        return root, nested

    def test_discovers_workspace_from_nested_directory(self) -> None:
        root, nested = self.make_workspace()

        found = find_workspace_file(nested)
        workspace = load_workspace(found)

        self.assertEqual(found, root / ".radeon-workspace.yaml")
        self.assertEqual(workspace.root, root)
        self.assertEqual(workspace.targets_file, root / "lab-private/config/targets.yaml")

    def test_rejects_path_outside_workspace(self) -> None:
        root, _ = self.make_workspace()
        path = root / ".radeon-workspace.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["private_dir"] = "../private"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")

        with self.assertRaisesRegex(ConfigError, "inside the workspace root"):
            load_workspace(path)


if __name__ == "__main__":
    unittest.main()

