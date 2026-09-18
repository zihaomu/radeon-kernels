from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from radeon_kernels.errors import ConfigError
from radeon_kernels.runtime.dispatcher import DispatchRequest, Dispatcher
from radeon_kernels.runtime.explain import (
    clear_last_dispatch,
    explain_last_dispatch,
    get_last_dispatch,
)
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint, FingerprintError
from radeon_kernels.runtime.loader import (
    ArtifactIntegrityError,
    ArtifactLoadError,
    PrecompiledArtifactLoader,
    _python_extension_module_name,
)
from radeon_kernels.runtime.registry import (
    ArtifactSpec,
    DispatchManifest,
    DispatchRegistry,
)


def artifact(name: str, digest: str = "a" * 64) -> dict:
    return {
        "provider": "native",
        "artifact": f"lib/{name}.so",
        "entrypoint": "gemm_out",
        "sha256": digest,
    }


def entry(
    identifier: str,
    *,
    priority: int,
    m: list[int],
    dtype: str = "fp16",
    rocm_abi: str = "7.15",
) -> dict:
    return {
        "id": identifier,
        "priority": priority,
        "environment": {
            "architecture": "gfx1151",
            "wavefront_size": 32,
            "rocm_abi": rocm_abi,
            "python_abi": "cp312",
            "pytorch_version": "2.14.0+rocm7.15",
        },
        "workload": {
            "dtype": dtype,
            "layout": "nn",
            "m": m,
            "n": [4096, 4096],
            "k": [4096, 4096],
        },
        "winner": artifact(identifier),
        "launch": {"variant": 8},
        "fallbacks": ["hipblaslt", "torch"],
        "evidence_id": f"evidence-{identifier}",
    }


def manifest(*entries: dict) -> dict:
    return {
        "schema_version": "1.0",
        "operator": "gemm",
        "semantic_version": "1.0",
        "default_fallbacks": ["hipblaslt", "torch"],
        "entries": list(entries),
    }


def fingerprint() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        architecture="gfx1151",
        wavefront_size=32,
        rocm_abi="7.15.0",
        python_abi="cp312",
        pytorch_version="2.14.0+rocm7.15",
    )


class FingerprintTests(unittest.TestCase):
    def test_normalizes_rocm_abi_and_features(self) -> None:
        value = EnvironmentFingerprint(
            architecture="gfx1151",
            wavefront_size=32,
            rocm_abi="7.15.0-rc1",
            python_abi="cp312",
            pytorch_version="2.14",
            features=("wmma", "wmma", "bf16"),
        )

        self.assertEqual(value.rocm_abi, "7.15")
        self.assertEqual(value.features, ("bf16", "wmma"))

    def test_detects_environment_without_importing_torch_eagerly(self) -> None:
        fake_torch = SimpleNamespace(
            __version__="2.14.0",
            version=SimpleNamespace(hip="7.15.0"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda index: SimpleNamespace(
                    gcnArchName="gfx1151:sramecc-:xnack-",
                    warp_size=32,
                ),
            ),
        )

        value = EnvironmentFingerprint.detect(torch_module=fake_torch)

        self.assertEqual(value.architecture, "gfx1151")
        self.assertEqual(value.rocm_abi, "7.15")

    def test_rejects_non_rocm_torch(self) -> None:
        fake_torch = SimpleNamespace(
            __version__="2.14.0",
            version=SimpleNamespace(hip=None),
            cuda=SimpleNamespace(is_available=lambda: True),
        )

        with self.assertRaisesRegex(FingerprintError, "does not report a ROCm"):
            EnvironmentFingerprint.detect(torch_module=fake_torch)


class RegistryTests(unittest.TestCase):
    def test_accepts_fallback_only_manifest(self) -> None:
        value = DispatchManifest.from_mapping(manifest())

        self.assertEqual(value.entries, ())
        self.assertEqual(value.default_fallbacks, ("hipblaslt", "torch"))

    def test_accepts_strict_higher_priority_subset(self) -> None:
        value = DispatchManifest.from_mapping(
            manifest(
                entry("broad", priority=10, m=[1, 512]),
                entry("exact", priority=100, m=[256, 256]),
            )
        )

        self.assertEqual(len(value.entries), 2)

    def test_rejects_same_priority_overlap(self) -> None:
        payload = manifest(
            entry("first", priority=10, m=[1, 512]),
            entry("second", priority=10, m=[256, 256]),
        )

        with self.assertRaisesRegex(ConfigError, "ambiguous overlapping"):
            DispatchManifest.from_mapping(payload)

    def test_rejects_partial_high_priority_shadow(self) -> None:
        payload = manifest(
            entry("broad", priority=10, m=[1, 512]),
            entry("partial", priority=100, m=[256, 768]),
        )

        with self.assertRaisesRegex(ConfigError, "must be a strict subset"):
            DispatchManifest.from_mapping(payload)

    def test_rejects_unknown_fields(self) -> None:
        payload = manifest(entry("exact", priority=100, m=[256, 256]))
        payload["search_on_user_machine"] = True

        with self.assertRaisesRegex(ConfigError, "unknown fields"):
            DispatchManifest.from_mapping(payload)

    def test_rejects_non_scalar_launch_parameters(self) -> None:
        payload = manifest(entry("exact", priority=100, m=[256, 256]))
        payload["entries"][0]["launch"] = {"variant": [8]}

        with self.assertRaisesRegex(ConfigError, "expected a string, number or boolean"):
            DispatchManifest.from_mapping(payload)

    def test_rejects_unknown_schema_version(self) -> None:
        payload = manifest(entry("exact", priority=100, m=[256, 256]))
        payload["schema_version"] = "2.0"

        with self.assertRaisesRegex(ConfigError, "unsupported dispatch schema"):
            DispatchManifest.from_mapping(payload)

    def test_loads_registry_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gemm.json"
            path.write_text(
                json.dumps(manifest(entry("exact", priority=100, m=[256, 256]))),
                encoding="utf-8",
            )

            registry = DispatchRegistry.from_directory(Path(directory))

        self.assertEqual(registry.keys(), (("gemm", "1.0"),))


class DispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_last_dispatch()
        self.registry = DispatchRegistry(
            [
                DispatchManifest.from_mapping(
                    manifest(
                        entry("broad", priority=10, m=[1, 512]),
                        entry("exact", priority=100, m=[256, 256]),
                    )
                )
            ]
        )

    def request(self, *, m: int, dtype: str = "fp16") -> DispatchRequest:
        return DispatchRequest(
            operator="gemm",
            semantic_version="1.0",
            signature={
                "dtype": dtype,
                "layout": "nn",
                "m": m,
                "n": 4096,
                "k": 4096,
            },
        )

    def test_selects_most_specific_matching_entry(self) -> None:
        decision = Dispatcher(self.registry).select(self.request(m=256), fingerprint())

        self.assertTrue(decision.matched)
        self.assertEqual(decision.entry_id, "exact")
        self.assertEqual(decision.winner.artifact, "lib/exact.so")
        self.assertIn("entry 'exact'", explain_last_dispatch())

    def test_selects_broad_region_outside_exact_shape(self) -> None:
        decision = Dispatcher(self.registry).select(self.request(m=128), fingerprint())

        self.assertEqual(decision.entry_id, "broad")

    def test_returns_declared_fallback_without_search(self) -> None:
        decision = Dispatcher(self.registry).select(
            self.request(m=256, dtype="bf16"), fingerprint()
        )

        self.assertFalse(decision.matched)
        self.assertEqual(decision.fallbacks, ("hipblaslt", "torch"))
        self.assertIn("no compatible", decision.reason)
        self.assertEqual(get_last_dispatch()["selected"], None)


class LoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "lib").mkdir()
        self.path = self.root / "lib" / "kernel.so"
        self.path.write_bytes(b"not a real extension")
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.loader = PrecompiledArtifactLoader(self.root)

    def spec(self, *, digest: str | None = None, artifact_format: str = "python_extension") -> ArtifactSpec:
        return ArtifactSpec(
            provider="native",
            artifact="lib/kernel.so",
            entrypoint="gemm_out",
            sha256=digest or self.digest,
            format=artifact_format,
        )

    def test_verifies_artifact_inside_pack(self) -> None:
        self.assertEqual(self.loader.resolve_and_verify(self.spec()), self.path)

    def test_rejects_digest_mismatch(self) -> None:
        with self.assertRaisesRegex(ArtifactIntegrityError, "SHA-256 mismatch"):
            self.loader.resolve_and_verify(self.spec(digest="0" * 64))

    def test_rejects_path_escape_even_for_programmatic_spec(self) -> None:
        outside = self.root.parent / "outside.so"
        outside.write_bytes(b"outside")
        self.addCleanup(outside.unlink)
        spec = ArtifactSpec(
            provider="native",
            artifact="../outside.so",
            entrypoint="gemm_out",
            sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
        )

        with self.assertRaisesRegex(ArtifactIntegrityError, "escapes"):
            self.loader.resolve_and_verify(spec)

    def test_hsaco_requires_provider_specific_launcher(self) -> None:
        with self.assertRaisesRegex(ArtifactLoadError, "provider-specific"):
            self.loader.load_module(self.spec(artifact_format="hsaco"))

    def test_renamed_extension_uses_its_exported_python_module_name(self) -> None:
        self.path.write_bytes(
            b"binary-prefix\x00PyInit_compiled_kernel\x00binary-suffix"
        )

        self.assertEqual(_python_extension_module_name(self.path), "compiled_kernel")

    def test_extension_rejects_ambiguous_python_module_exports(self) -> None:
        self.path.write_bytes(b"PyInit_first\x00PyInit_second\x00")

        with self.assertRaisesRegex(ArtifactLoadError, "multiple PyInit symbols"):
            _python_extension_module_name(self.path)


class SchemaTests(unittest.TestCase):
    def test_schema_directory_is_a_python_resource_package(self) -> None:
        root = Path(__file__).parents[1] / "schemas"
        self.assertTrue((root / "__init__.py").is_file())

    def test_all_public_schemas_are_valid_json_with_unique_ids(self) -> None:
        root = Path(__file__).parents[1] / "schemas"
        documents = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]

        self.assertEqual(len(documents), 9)
        self.assertEqual(len({document["$id"] for document in documents}), 9)
        self.assertTrue(all(document["additionalProperties"] is False for document in documents))


if __name__ == "__main__":
    unittest.main()
