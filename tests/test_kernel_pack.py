from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from developer.promotion.build_kernel_pack import build
from radeon_kernels.errors import ConfigError
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
from radeon_kernels.runtime.pack import KernelPack
from radeon_kernels.runtime.packs import KernelPackRegistry


class KernelPackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "radeon_kernels_wmma_gfx1151.so"
        self.source.write_bytes(b"immutable test artifact")
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.output = self.root / "rk-gemm-gfx1151-rocm715"

    def arguments(self, **overrides: object) -> SimpleNamespace:
        values = {
            "source_artifact": self.source,
            "expected_sha256": self.digest,
            "output": self.output,
            "pack_id": "rk-gemm-gfx1151-rocm715",
            "pack_version": "0.1.0",
            "architecture": "gfx1151",
            "wavefront_size": 32,
            "rocm_abi": "7.15",
            "python_abi": "cp312",
            "pytorch_version": "2.14.0+rocm7.15",
            "operator": "gemm",
            "semantic_version": "1.0",
            "provider": "native",
            "entrypoint": "gemm_out",
            "format": "python_extension",
            "evidence_id": "gemm-gfx1151-fp16-001",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_builder_copies_and_verifies_immutable_artifact(self) -> None:
        pack = build(self.arguments())

        self.assertEqual(pack.pack_id, "rk-gemm-gfx1151-rocm715")
        self.assertEqual(pack.get("gemm", "1.0").artifact.sha256, self.digest)
        self.assertEqual(
            (self.output / "lib" / self.source.name).read_bytes(),
            self.source.read_bytes(),
        )

    def test_pack_matches_only_exact_runtime_abi(self) -> None:
        pack = build(self.arguments())
        matching = EnvironmentFingerprint(
            architecture="gfx1151",
            wavefront_size=32,
            rocm_abi="7.15.0",
            python_abi="cp312",
            pytorch_version="2.14.0+rocm7.15",
        )
        different_torch = EnvironmentFingerprint(
            architecture="gfx1151",
            wavefront_size=32,
            rocm_abi="7.15.0",
            python_abi="cp312",
            pytorch_version="2.14.1+rocm7.15",
        )

        self.assertTrue(pack.matches(matching))
        self.assertFalse(pack.matches(different_torch))

    def test_runtime_rejects_unsigned_pack_by_default(self) -> None:
        build(self.arguments())

        registry = KernelPackRegistry.discover([self.output])

        self.assertEqual(registry.packs, ())
        self.assertEqual(len(registry.rejected), 1)
        self.assertIn("manifest.sig.json", registry.rejected[0])

    def test_builder_refuses_wrong_source_hash(self) -> None:
        with self.assertRaisesRegex(ValueError, "source SHA-256 mismatch"):
            build(self.arguments(expected_sha256="0" * 64))

        self.assertFalse(self.output.exists())

    def test_builder_refuses_to_overwrite_pack(self) -> None:
        build(self.arguments())

        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            build(self.arguments())

    def test_pack_rejects_duplicate_operator_versions(self) -> None:
        build(self.arguments())
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        duplicate = dict(manifest["artifacts"][0])
        duplicate["artifact"] = "lib/duplicate.so"
        manifest["artifacts"].append(duplicate)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ConfigError, "operator semantic versions must be unique"):
            KernelPack.from_directory(self.output, verify_artifacts=False)


if __name__ == "__main__":
    unittest.main()
