from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from radeon_kernels.ops.gemm.api import GemmRuntime
from radeon_kernels.runtime.explain import clear_last_dispatch, get_last_dispatch
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
from radeon_kernels.runtime.loader import PrecompiledArtifactLoader
from radeon_kernels.runtime.packs import KernelPackRegistry, default_pack_roots
from radeon_kernels.runtime.registry import DispatchManifest, DispatchRegistry
from radeon_kernels.runtime.resources import builtin_dispatch_registry


@dataclass(frozen=True)
class FakeDevice:
    index: int = 0


class FakeTensor:
    _next_pointer = 1000

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        dtype: str = "float16",
        device: FakeDevice = FakeDevice(),
        contiguous: bool = True,
        requires_grad: bool = False,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.is_cuda = True
        self.requires_grad = requires_grad
        self._contiguous = contiguous
        self._pointer = FakeTensor._next_pointer
        FakeTensor._next_pointer += 1

    def dim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def new_empty(self, shape: tuple[int, int]) -> FakeTensor:
        return FakeTensor(shape, dtype=self.dtype, device=self.device)

    def data_ptr(self) -> int:
        return self._pointer


class FakeTorch:
    Tensor = FakeTensor
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __init__(self) -> None:
        self.cuda = SimpleNamespace(current_device=lambda: 0)
        self.mm_calls = 0

    def mm(self, a: FakeTensor, b: FakeTensor, *, out: FakeTensor | None = None) -> FakeTensor:
        self.mm_calls += 1
        return out if out is not None else a.new_empty((a.shape[0], b.shape[1]))


def fingerprint() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        architecture="gfx1151",
        wavefront_size=32,
        rocm_abi="7.15",
        python_abi="cp312",
        pytorch_version="2.14.0+rocm7.15",
    )


def manifest(digest: str) -> DispatchManifest:
    return DispatchManifest.from_mapping(
        {
            "schema_version": "1.0",
            "operator": "gemm",
            "semantic_version": "1.0",
            "default_fallbacks": ["torch"],
            "entries": [
                {
                    "id": "test-gemm-winner",
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
                        "layout": "nn",
                        "m": [2, 2],
                        "n": [4, 4],
                        "k": [3, 3],
                        "contiguous": True,
                        "requires_grad": False,
                    },
                    "winner": {
                        "provider": "native",
                        "artifact": "lib/test_kernel.so",
                        "entrypoint": "gemm_out",
                        "sha256": digest,
                        "format": "python_extension",
                    },
                    "launch": {"variant": 8},
                    "fallbacks": ["torch"],
                    "evidence_id": "test-evidence",
                }
            ],
        }
    )


def write_pack(root: Path, digest: str, artifact: bytes) -> Path:
    pack = root / "test-pack"
    library = pack / "lib"
    library.mkdir(parents=True)
    (library / "test_kernel.so").write_bytes(artifact)
    (pack / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "pack_id": "test-pack",
                "pack_version": "0.1.0",
                "environment": {
                    "architecture": "gfx1151",
                    "wavefront_size": 32,
                    "rocm_abi": "7.15",
                    "python_abi": "cp312",
                    "pytorch_version": "2.14.0+rocm7.15",
                },
                "artifacts": [
                    {
                        "operator": "gemm",
                        "semantic_version": "1.0",
                        "provider": "native",
                        "artifact": "lib/test_kernel.so",
                        "entrypoint": "gemm_out",
                        "sha256": digest,
                        "format": "python_extension",
                        "evidence_id": "test-evidence",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return pack


class FakeLoader:
    variants: list[int] = []

    def __init__(self, root: Path) -> None:
        self._verifier = PrecompiledArtifactLoader(root)

    def load_entrypoint(self, artifact: object) -> SimpleNamespace:
        self._verifier.resolve_and_verify(artifact)

        def call(a: FakeTensor, b: FakeTensor, output: FakeTensor, variant: int) -> FakeTensor:
            self.variants.append(variant)
            return output

        return SimpleNamespace(callable=call)


class GemmDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_last_dispatch()
        FakeLoader.variants.clear()
        self.torch = FakeTorch()
        self.a = FakeTensor((2, 3))
        self.b = FakeTensor((3, 4))
        self.artifact = b"precompiled kernel"
        self.digest = hashlib.sha256(self.artifact).hexdigest()
        self.registry = DispatchRegistry([manifest(self.digest)])
        self.detector = lambda **_: fingerprint()

    def runtime(self, root: Path) -> GemmRuntime:
        return GemmRuntime(
            registry=self.registry,
            pack_roots=(root,),
            loader_factory=FakeLoader,
            fingerprint_detector=self.detector,
            require_pack_signatures=False,
        )

    def test_executes_exact_precompiled_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pack(root, self.digest, self.artifact)

            result = self.runtime(root).execute(self.a, self.b, torch_module=self.torch)

        self.assertEqual(result.shape, (2, 4))
        self.assertEqual(FakeLoader.variants, [8])
        self.assertEqual(self.torch.mm_calls, 0)
        self.assertEqual(get_last_dispatch()["selected"], "native/gemm_out")
        self.assertIn("loaded pack test-pack 0.1.0", get_last_dispatch()["reason"])

    def test_missing_pack_executes_declared_torch_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.runtime(Path(directory)).execute(
                self.a, self.b, torch_module=self.torch
            )

        self.assertEqual(result.shape, (2, 4))
        self.assertEqual(self.torch.mm_calls, 1)
        self.assertEqual(get_last_dispatch()["selected"], "torch/mm")
        self.assertIn("winner unavailable", get_last_dispatch()["reason"])

    def test_bf16_uses_declared_fallback(self) -> None:
        a = FakeTensor((2, 3), dtype="bfloat16")
        b = FakeTensor((3, 4), dtype="bfloat16")
        with tempfile.TemporaryDirectory() as directory:
            self.runtime(Path(directory)).execute(a, b, torch_module=self.torch)

        self.assertEqual(self.torch.mm_calls, 1)
        self.assertEqual(get_last_dispatch()["selected"], "torch/mm")

    def test_rocm_abi_mismatch_uses_declared_fallback(self) -> None:
        incompatible = EnvironmentFingerprint(
            architecture="gfx1151",
            wavefront_size=32,
            rocm_abi="7.14",
            python_abi="cp312",
            pytorch_version="2.14.0+rocm7.15",
        )
        with tempfile.TemporaryDirectory() as directory:
            runtime = GemmRuntime(
                registry=self.registry,
                pack_roots=(Path(directory),),
                loader_factory=FakeLoader,
                fingerprint_detector=lambda **_: incompatible,
                require_pack_signatures=False,
            )
            runtime.execute(self.a, self.b, torch_module=self.torch)

        self.assertEqual(self.torch.mm_calls, 1)
        self.assertEqual(get_last_dispatch()["selected"], "torch/mm")

    def test_corrupted_winner_falls_back_after_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pack = write_pack(root, self.digest, self.artifact)
            (pack / "lib" / "test_kernel.so").write_bytes(b"tampered")

            self.runtime(root).execute(self.a, self.b, torch_module=self.torch)

        self.assertEqual(self.torch.mm_calls, 1)
        self.assertIn("SHA-256 mismatch", get_last_dispatch()["reason"])

    def test_unmatched_shape_uses_fallback_without_loading_pack(self) -> None:
        different = FakeTensor((1, 3))
        with tempfile.TemporaryDirectory() as directory:
            result = self.runtime(Path(directory)).execute(
                different, self.b, torch_module=self.torch
            )

        self.assertEqual(result.shape, (1, 4))
        self.assertEqual(FakeLoader.variants, [])
        self.assertIn("no compatible", get_last_dispatch()["reason"])

    def test_autograd_input_does_not_use_inference_only_winner(self) -> None:
        requires_grad = FakeTensor((2, 3), requires_grad=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pack(root, self.digest, self.artifact)

            self.runtime(root).execute(requires_grad, self.b, torch_module=self.torch)

        self.assertEqual(FakeLoader.variants, [])
        self.assertEqual(get_last_dispatch()["selected"], "torch/mm")

    def test_winner_writes_to_caller_output(self) -> None:
        output = FakeTensor((2, 4))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pack(root, self.digest, self.artifact)

            result = self.runtime(root).execute(
                self.a,
                self.b,
                output=output,
                torch_module=self.torch,
            )

        self.assertIs(result, output)
        self.assertEqual(self.torch.mm_calls, 0)

    def test_rejects_mismatched_output_before_dispatch(self) -> None:
        output = FakeTensor((2, 5))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "output must have shape"):
                self.runtime(Path(directory)).execute(
                    self.a,
                    self.b,
                    output=output,
                    torch_module=self.torch,
                )

        self.assertIsNone(get_last_dispatch())


class PublishedDispatchTests(unittest.TestCase):
    def test_builtin_manifest_contains_both_verified_winners(self) -> None:
        manifest_value = builtin_dispatch_registry().get("gemm", "1.0")

        self.assertEqual(
            {entry.environment.architecture for entry in manifest_value.entries},
            {"gfx1151", "gfx1201"},
        )
        self.assertEqual(
            {entry.launch["variant"] for entry in manifest_value.entries},
            {8, 9},
        )
        self.assertTrue(all(entry.fallbacks == ("torch",) for entry in manifest_value.entries))

    def test_pack_roots_put_explicit_paths_before_defaults(self) -> None:
        roots = default_pack_roots(
            environ={"RADEON_KERNELS_PACK_PATH": f"/first{os.pathsep}/second"},
            home=Path("/home/tester"),
            prefix=Path("/runtime"),
        )

        self.assertEqual(roots[0:2], (Path("/first"), Path("/second")))
        self.assertEqual(roots[-2], Path("/home/tester/.local/share/radeon-kernels/packs"))
        self.assertEqual(roots[-1], Path("/runtime/share/radeon-kernels/packs"))

    def test_pack_discovery_ignores_invalid_manifests_but_reports_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "invalid"
            invalid.mkdir()
            (invalid / "manifest.json").write_text("{}", encoding="utf-8")

            registry = KernelPackRegistry.discover((root,), require_signatures=False)

        self.assertEqual(registry.packs, ())
        self.assertEqual(len(registry.rejected), 1)


if __name__ == "__main__":
    unittest.main()
