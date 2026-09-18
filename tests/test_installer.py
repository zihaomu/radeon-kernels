from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from developer.promotion.publish_approved import _deterministic_archive
from developer.promotion.signing import generate_private_key, sign_file
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
from radeon_kernels.runtime.installer import (
    PackInstallError,
    compatible_records,
    install_release,
    verify_installed_pack,
)
from radeon_kernels.runtime.resources import builtin_pack_indexes
from radeon_kernels.runtime.signing import TrustedKeyring, sha256_file


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class PackInstallerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.pack_id = "rk-rms-norm-gfx1151-test"
        self.pack_root = self.root / "source" / self.pack_id
        artifact = self.pack_root / "lib" / "kernel.so"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"precompiled test artifact")
        self.environment = {
            "architecture": "gfx1151",
            "wavefront_size": 32,
            "rocm_abi": "7.15",
            "python_abi": "cp312",
            "pytorch_version": "2.14.0+rocm7.15",
        }
        self.artifact = {
            "operator": "rms_norm",
            "semantic_version": "1.0",
            "provider": "native",
            "artifact": "lib/kernel.so",
            "entrypoint": "rms_norm",
            "sha256": sha256_file(artifact),
            "format": "python_extension",
            "evidence_id": "evidence-rms-norm-test",
        }
        _write_json(
            self.pack_root / "manifest.json",
            {
                "schema_version": "1.0",
                "pack_id": self.pack_id,
                "pack_version": "0.1.0",
                "environment": self.environment,
                "artifacts": [self.artifact],
            },
        )
        private_key = self.root / "private-key.json"
        public_key = generate_private_key(private_key, "test-release-key")
        self.keyring = TrustedKeyring.from_mapping(
            {"schema_version": "1.0", "keys": [public_key]}
        )
        manifest_signature = sign_file(
            private_key_path=private_key,
            file_path=self.pack_root / "manifest.json",
            signed_file="manifest.json",
            output=self.pack_root / "manifest.sig.json",
        )
        self.archive = self.root / f"{self.pack_id}-0.1.0.tar.gz"
        _deterministic_archive(self.pack_root, self.archive)
        self.index = {
            "schema_version": "1.0",
            "operator": "rms_norm",
            "semantic_version": "1.0",
            "packs": [
                {
                    "pack_id": self.pack_id,
                    "pack_version": "0.1.0",
                    "environment": self.environment,
                    "manifest_sha256": sha256_file(self.pack_root / "manifest.json"),
                    "manifest_signature": manifest_signature,
                    "release_file": self.archive.name,
                    "release_sha256": sha256_file(self.archive),
                    "release_size": self.archive.stat().st_size,
                    "artifacts": [self.artifact],
                    "approval_id": "approval-rms-norm-test",
                }
            ],
        }
        self.fingerprint = EnvironmentFingerprint.from_mapping(self.environment)

    def test_installs_and_reverifies_signed_release(self) -> None:
        destination = self.root / "installed"

        pack = install_release(
            self.archive,
            self.index,
            self.fingerprint,
            destination,
            trusted_keys=self.keyring,
        )

        self.assertEqual(pack.pack_id, self.pack_id)
        verified = verify_installed_pack(pack.root, trusted_keys=self.keyring)
        self.assertEqual(verified.pack_version, "0.1.0")

    def test_rejects_archive_content_changed_after_indexing(self) -> None:
        self.archive.write_bytes(self.archive.read_bytes() + b"tampered")

        with self.assertRaisesRegex(PackInstallError, "size mismatch"):
            install_release(
                self.archive,
                self.index,
                self.fingerprint,
                self.root / "installed",
                trusted_keys=self.keyring,
            )

    def test_rejects_incompatible_environment(self) -> None:
        other = EnvironmentFingerprint(
            architecture="gfx1201",
            wavefront_size=32,
            rocm_abi="7.15",
            python_abi="cp312",
            pytorch_version="2.14.0+rocm7.15",
        )

        self.assertEqual(compatible_records(self.index, other), ())
        with self.assertRaisesRegex(PackInstallError, "no indexed pack matches"):
            install_release(
                self.archive,
                self.index,
                other,
                self.root / "installed",
                trusted_keys=self.keyring,
            )

    def test_refuses_to_overwrite_installed_pack(self) -> None:
        destination = self.root / "installed"
        install_release(
            self.archive,
            self.index,
            self.fingerprint,
            destination,
            trusted_keys=self.keyring,
        )

        with self.assertRaisesRegex(PackInstallError, "refusing to overwrite"):
            install_release(
                self.archive,
                self.index,
                self.fingerprint,
                destination,
                trusted_keys=self.keyring,
            )

    def test_builtin_indexes_are_signature_verified(self) -> None:
        indexes = builtin_pack_indexes()

        self.assertTrue(any(index["operator"] == "gemm" for index in indexes))


if __name__ == "__main__":
    unittest.main()
