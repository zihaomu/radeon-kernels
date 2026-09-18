from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from radeon_kernels.ops._runtime import OperatorRuntime
from radeon_kernels.runtime.explain import clear_last_dispatch, get_last_dispatch
from radeon_kernels.runtime.fingerprint import EnvironmentFingerprint
from radeon_kernels.runtime.registry import DispatchManifest, DispatchRegistry


def _fingerprint() -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        architecture="gfx1151",
        wavefront_size=32,
        rocm_abi="7.15",
        python_abi="cp312",
        pytorch_version="2.14.0+rocm7.15",
    )


def _manifest(*, winner: bool) -> DispatchManifest:
    entries = []
    if winner:
        entries.append(
            {
                "id": "softmax-test-winner",
                "priority": 100,
                "environment": {
                    "architecture": "gfx1151",
                    "wavefront_size": 32,
                    "rocm_abi": "7.15",
                    "python_abi": "cp312",
                    "pytorch_version": "2.14.0+rocm7.15",
                },
                "workload": {"dtype": "fp16", "rows": [2, 2], "width": [4, 4]},
                "winner": {
                    "provider": "native",
                    "artifact": "lib/softmax.so",
                    "entrypoint": "softmax",
                    "sha256": "a" * 64,
                    "format": "python_extension",
                },
                "launch": {},
                "fallbacks": ["torch"],
                "evidence_id": "evidence-softmax-test",
            }
        )
    return DispatchManifest.from_mapping(
        {
            "schema_version": "1.0",
            "operator": "softmax",
            "semantic_version": "1.0",
            "default_fallbacks": ["torch"],
            "entries": entries,
        }
    )


class _PackRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = 0

    def resolve(self, **_: object) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(pack=SimpleNamespace(root=self.root, pack_id="test-pack", pack_version="0.1.0"))


class _Loader:
    def __init__(self, _: Path) -> None:
        pass

    def load_entrypoint(self, _: object) -> SimpleNamespace:
        return SimpleNamespace(callable=lambda value: f"native:{value}")


class OperatorRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_last_dispatch()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def runtime(self, *, winner: bool) -> tuple[OperatorRuntime, _PackRegistry]:
        packs = _PackRegistry(self.root)
        runtime = OperatorRuntime(
            registry=DispatchRegistry([_manifest(winner=winner)]),
            pack_registry=packs,
            loader_factory=_Loader,
            fingerprint_detector=lambda **_: _fingerprint(),
        )
        return runtime, packs

    def test_fallback_only_manifest_executes_declared_fallback(self) -> None:
        runtime, packs = self.runtime(winner=False)

        result = runtime.execute(
            operator="softmax",
            semantic_version="1.0",
            signature={"dtype": "fp16", "rows": 2, "width": 4},
            device_index=0,
            torch_module=object(),
            native_arguments=("input",),
            fallback=lambda: "torch-result",
            fallback_selected="torch/softmax",
        )

        self.assertEqual(result, "torch-result")
        self.assertEqual(packs.calls, 0)
        self.assertEqual(get_last_dispatch()["selected"], "torch/softmax")

    def test_matching_entry_loads_precompiled_winner(self) -> None:
        runtime, packs = self.runtime(winner=True)

        result = runtime.execute(
            operator="softmax",
            semantic_version="1.0",
            signature={"dtype": "fp16", "rows": 2, "width": 4},
            device_index=0,
            torch_module=object(),
            native_arguments=("input",),
            fallback=lambda: "torch-result",
            fallback_selected="torch/softmax",
        )

        self.assertEqual(result, "native:input")
        self.assertEqual(packs.calls, 1)
        self.assertEqual(get_last_dispatch()["selected"], "native/softmax")


if __name__ == "__main__":
    unittest.main()
