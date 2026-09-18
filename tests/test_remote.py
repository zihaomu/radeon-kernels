from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from radeon_kernels.lab.config import load_targets
from radeon_kernels.lab.remote import CommandResult, RemoteStatus, probe_target
from tests.helpers import target, targets_payload


class RemoteProbeTests(unittest.TestCase):
    def configured_target(self, expected_architecture: str = "gfx1201"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        item = target()
        item["expected_architecture"] = expected_architecture
        path = Path(directory.name) / "targets.yaml"
        path.write_text(
            yaml.safe_dump(targets_payload(item)),
            encoding="utf-8",
        )
        return load_targets(path).targets[0]

    def test_ready_probe(self) -> None:
        commands: list[list[str]] = []

        def runner(command, timeout):
            commands.append(list(command))
            self.assertEqual(timeout, 7)
            return CommandResult(
                0,
                "user=tester\narchitecture=gfx1201\nruntime=docker\n"
                "image_present=yes\nremote_root_writable=yes\n",
                "",
            )

        result = probe_target(
            self.configured_target(),
            timeout_seconds=7,
            runner=runner,
        )

        self.assertEqual(result.status, RemoteStatus.READY)
        self.assertEqual(result.detected_architecture, "gfx1201")
        self.assertEqual(commands[0][0], "ssh")
        self.assertIn("lab-a-ssh", commands[0])

    def test_workspace_known_hosts_policy_is_explicit(self) -> None:
        commands: list[list[str]] = []

        def runner(command, timeout):
            commands.append(list(command))
            return CommandResult(
                0,
                "user=tester\narchitecture=gfx1201\nruntime=docker\n"
                "image_present=yes\nremote_root_writable=yes\n",
                "",
            )

        probe_target(
            self.configured_target(),
            known_hosts_file=Path("/private/state/known_hosts"),
            accept_new_host_keys=True,
            runner=runner,
        )

        self.assertIn("UserKnownHostsFile=/private/state/known_hosts", commands[0])
        self.assertIn("StrictHostKeyChecking=accept-new", commands[0])

    def test_architecture_mismatch_is_invalid(self) -> None:
        def runner(command, timeout):
            return CommandResult(
                0,
                "user=tester\narchitecture=gfx1100\nruntime=docker\n"
                "image_present=yes\nremote_root_writable=yes\n",
                "",
            )

        result = probe_target(self.configured_target(), runner=runner)

        self.assertEqual(result.status, RemoteStatus.INVALID)
        self.assertIn("expected gfx1201", result.message)

    def test_ssh_failure_is_unreachable(self) -> None:
        def runner(command, timeout):
            return CommandResult(255, "", "host key verification failed")

        result = probe_target(self.configured_target(), runner=runner)

        self.assertEqual(result.status, RemoteStatus.UNREACHABLE)
        self.assertIn("host key", result.message)


if __name__ == "__main__":
    unittest.main()
