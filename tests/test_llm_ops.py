from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import yaml
from jsonschema import Draft202012Validator

import radeon_kernels as rk
from radeon_kernels.ops.add_rms_norm.api import _execute as execute_add_rms_norm
from radeon_kernels.ops.gemv.api import _execute as execute_gemv
from radeon_kernels.ops.kv_cache.api import (
    _execute_append as execute_kv_cache_append,
    _execute_copy as execute_kv_cache_copy,
)
from radeon_kernels.ops.paged_attention.api import (
    _execute as execute_paged_attention_decode,
)
from radeon_kernels.ops.rms_norm.api import _execute as execute_rms_norm
from radeon_kernels.ops.rope.api import _execute as execute_rope
from radeon_kernels.ops.sdpa.api import _execute as execute_sdpa
from radeon_kernels.ops.softmax.api import _execute as execute_softmax
from radeon_kernels.ops.swiglu.api import _execute as execute_swiglu
from radeon_kernels.runtime.resources import builtin_dispatch_registry


@dataclass(frozen=True)
class FakeDevice:
    index: int = 0


class FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
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

    def dim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self._contiguous


class FakeTorch:
    Tensor = FakeTensor
    float16 = "float16"
    bfloat16 = "bfloat16"
    int32 = "int32"
    int64 = "int64"
    cuda = SimpleNamespace(current_device=lambda: 0)


class CaptureRuntime:
    def __init__(self, result: object = "result") -> None:
        self.result = result
        self.call: dict[str, object] | None = None

    def execute(self, **values: object) -> object:
        self.call = values
        return self.result


class LlmOperatorApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.torch = FakeTorch()

    def test_rms_norm_builds_stable_dispatch_signature(self) -> None:
        runtime = CaptureRuntime()
        x = FakeTensor((128, 4096), requires_grad=True)
        weight = FakeTensor((4096,))

        result = execute_rms_norm(
            x, weight, 1e-6, torch_module=self.torch, runtime=runtime
        )

        self.assertEqual(result, "result")
        self.assertEqual(runtime.call["operator"], "rms_norm")
        self.assertEqual(
            runtime.call["signature"],
            {
                "dtype": "fp16",
                "tokens": 128,
                "hidden": 4096,
                "eps": 1e-6,
                "contiguous": True,
                "requires_grad": True,
            },
        )

    def test_add_rms_norm_preserves_two_output_contract(self) -> None:
        runtime = CaptureRuntime(["normalized", "residual"])
        tensors = (FakeTensor((32, 8192)), FakeTensor((32, 8192)), FakeTensor((8192,)))

        result = execute_add_rms_norm(
            *tensors, 1e-6, torch_module=self.torch, runtime=runtime
        )

        self.assertEqual(result, ("normalized", "residual"))
        self.assertEqual(runtime.call["operator"], "add_rms_norm")
        self.assertEqual(runtime.call["native_arguments"][-1], 1e-6)

    def test_rope_supports_grouped_query_attention(self) -> None:
        runtime = CaptureRuntime(["q", "k"])

        result = execute_rope(
            FakeTensor((512, 32, 128)),
            FakeTensor((512, 8, 128)),
            FakeTensor((512, 128)),
            FakeTensor((512, 128)),
            torch_module=self.torch,
            runtime=runtime,
        )

        self.assertEqual(result, ("q", "k"))
        signature = runtime.call["signature"]
        self.assertEqual(signature["q_heads"], 32)
        self.assertEqual(signature["kv_heads"], 8)
        self.assertEqual(signature["mode"], "split_half")

    def test_silu_mul_maps_to_versioned_swiglu_operator(self) -> None:
        runtime = CaptureRuntime()

        execute_swiglu(
            FakeTensor((1, 11008)),
            FakeTensor((1, 11008)),
            torch_module=self.torch,
            runtime=runtime,
        )

        self.assertEqual(runtime.call["operator"], "swiglu")
        self.assertEqual(runtime.call["fallback_selected"], "torch/silu_mul")

    def test_softmax_records_rows_and_width(self) -> None:
        runtime = CaptureRuntime()

        execute_softmax(
            FakeTensor((64, 65536), contiguous=False),
            torch_module=self.torch,
            runtime=runtime,
        )

        self.assertEqual(
            runtime.call["signature"],
            {
                "dtype": "fp16",
                "rows": 64,
                "width": 65536,
                "contiguous": False,
                "requires_grad": False,
            },
        )

    def test_gemv_uses_nk_weight_layout_and_skinny_signature(self) -> None:
        runtime = CaptureRuntime()

        execute_gemv(
            FakeTensor((8, 4096)),
            FakeTensor((11008, 4096)),
            torch_module=self.torch,
            runtime=runtime,
        )

        self.assertEqual(runtime.call["operator"], "gemv")
        self.assertEqual(
            runtime.call["signature"],
            {
                "dtype": "fp16",
                "m": 8,
                "n": 11008,
                "k": 4096,
                "weight_layout": "nk",
                "contiguous": True,
                "requires_grad": False,
            },
        )
        self.assertEqual(runtime.call["fallback_selected"], "torch/mm")

    def test_kv_cache_append_uses_physical_slot_signature(self) -> None:
        key_cache = FakeTensor((256, 16, 8, 128))
        value_cache = FakeTensor((256, 16, 8, 128))
        runtime = CaptureRuntime((key_cache, value_cache))

        result = execute_kv_cache_append(
            FakeTensor((32, 8, 128)),
            FakeTensor((32, 8, 128)),
            key_cache,
            value_cache,
            FakeTensor((32,), dtype="int64"),
            torch_module=self.torch,
            runtime=runtime,
        )

        self.assertEqual(result, (key_cache, value_cache))
        self.assertEqual(runtime.call["operator"], "kv_cache_append")
        self.assertEqual(
            runtime.call["signature"],
            {
                "dtype": "fp16",
                "tokens": 32,
                "num_blocks": 256,
                "block_size": 16,
                "kv_heads": 8,
                "head_dim": 128,
                "index_dtype": "int64",
                "layout": "block_slot_head_dim",
                "contiguous": True,
                "requires_grad": False,
            },
        )

    def test_kv_cache_copy_uses_source_destination_pairs(self) -> None:
        key_cache = FakeTensor((128, 32, 32, 128), dtype="bfloat16")
        value_cache = FakeTensor((128, 32, 32, 128), dtype="bfloat16")
        runtime = CaptureRuntime((key_cache, value_cache))

        execute_kv_cache_copy(
            key_cache,
            value_cache,
            FakeTensor((8, 2), dtype="int32"),
            torch_module=self.torch,
            runtime=runtime,
        )

        self.assertEqual(runtime.call["operator"], "kv_cache_copy")
        self.assertEqual(runtime.call["signature"]["pairs"], 8)
        self.assertEqual(runtime.call["signature"]["index_dtype"], "int32")
        self.assertEqual(runtime.call["signature"]["layout"], "block_slot_head_dim")

    def test_paged_attention_decode_builds_gqa_signature(self) -> None:
        runtime = CaptureRuntime("decoded")

        result = execute_paged_attention_decode(
            FakeTensor((8, 32, 128)),
            FakeTensor((512, 16, 8, 128)),
            FakeTensor((512, 16, 8, 128)),
            FakeTensor((8, 64), dtype="int32"),
            FakeTensor((8,), dtype="int32"),
            torch_module=self.torch,
            runtime=runtime,
        )

        self.assertEqual(result, "decoded")
        self.assertEqual(runtime.call["operator"], "paged_attention_decode")
        self.assertEqual(
            runtime.call["signature"],
            {
                "dtype": "fp16",
                "batch": 8,
                "query_heads": 32,
                "kv_heads": 8,
                "head_dim": 128,
                "num_blocks": 512,
                "block_size": 16,
                "max_blocks_per_sequence": 64,
                "index_dtype": "int32",
                "cache_layout": "block_slot_head_dim",
                "scale_mode": "default",
                "contiguous": True,
                "requires_grad": False,
            },
        )
        self.assertAlmostEqual(runtime.call["native_arguments"][-1], 128**-0.5)
        self.assertEqual(runtime.call["fallback_selected"], "torch/paged_attention")

    def test_sdpa_builds_causal_gqa_signature(self) -> None:
        runtime = CaptureRuntime("attended")

        result = execute_sdpa(
            FakeTensor((2, 32, 256, 128)),
            FakeTensor((2, 8, 256, 128)),
            FakeTensor((2, 8, 256, 128)),
            True,
            torch_module=self.torch,
            runtime=runtime,
        )

        self.assertEqual(result, "attended")
        self.assertEqual(runtime.call["operator"], "sdpa")
        self.assertEqual(
            runtime.call["signature"],
            {
                "dtype": "fp16",
                "batch": 2,
                "query_heads": 32,
                "kv_heads": 8,
                "sequence": 256,
                "head_dim": 128,
                "layout": "batch_head_sequence_dim",
                "attention_mode": "causal",
                "scale_mode": "default",
                "contiguous": True,
                "requires_grad": False,
            },
        )
        self.assertAlmostEqual(runtime.call["native_arguments"][-1], 128**-0.5)
        self.assertEqual(
            runtime.call["fallback_selected"],
            "torch/scaled_dot_product_attention",
        )

    def test_rejects_invalid_shapes_and_epsilon(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden dimension"):
            execute_rms_norm(
                FakeTensor((2, 8)),
                FakeTensor((4,)),
                1e-6,
                torch_module=self.torch,
                runtime=CaptureRuntime(),
            )
        with self.assertRaisesRegex(ValueError, "must be even"):
            execute_rope(
                FakeTensor((2, 4, 63)),
                FakeTensor((2, 4, 63)),
                FakeTensor((2, 63)),
                FakeTensor((2, 63)),
                torch_module=self.torch,
                runtime=CaptureRuntime(),
            )
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            execute_rms_norm(
                FakeTensor((2, 8)),
                FakeTensor((8,)),
                0,
                torch_module=self.torch,
                runtime=CaptureRuntime(),
            )
        with self.assertRaisesRegex(ValueError, "share the K dimension"):
            execute_gemv(
                FakeTensor((1, 4096)),
                FakeTensor((4096, 8192)),
                torch_module=self.torch,
                runtime=CaptureRuntime(),
            )
        with self.assertRaisesRegex(ValueError, "M must be in"):
            execute_gemv(
                FakeTensor((33, 4096)),
                FakeTensor((4096, 4096)),
                torch_module=self.torch,
                runtime=CaptureRuntime(),
            )
        with self.assertRaisesRegex(ValueError, "one slot per token"):
            execute_kv_cache_append(
                FakeTensor((2, 8, 128)),
                FakeTensor((2, 8, 128)),
                FakeTensor((16, 16, 8, 128)),
                FakeTensor((16, 16, 8, 128)),
                FakeTensor((1,), dtype="int64"),
                torch_module=self.torch,
                runtime=CaptureRuntime(),
            )
        with self.assertRaisesRegex(TypeError, "index tensors"):
            execute_kv_cache_copy(
                FakeTensor((16, 16, 8, 128)),
                FakeTensor((16, 16, 8, 128)),
                FakeTensor((1, 2)),
                torch_module=self.torch,
                runtime=CaptureRuntime(),
            )
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            execute_kv_cache_append(
                FakeTensor((2, 8, 128), contiguous=False),
                FakeTensor((2, 8, 128)),
                FakeTensor((16, 16, 8, 128)),
                FakeTensor((16, 16, 8, 128)),
                FakeTensor((2,), dtype="int64"),
                torch_module=self.torch,
                runtime=CaptureRuntime(),
            )
        with self.assertRaisesRegex(ValueError, "divisible by kv_heads"):
            execute_paged_attention_decode(
                FakeTensor((2, 10, 128)),
                FakeTensor((32, 16, 4, 128)),
                FakeTensor((32, 16, 4, 128)),
                FakeTensor((2, 8), dtype="int32"),
                FakeTensor((2,), dtype="int32"),
                torch_module=self.torch,
                runtime=CaptureRuntime(),
            )
        with self.assertRaisesRegex(ValueError, "one of 64, 128, 256"):
            execute_paged_attention_decode(
                FakeTensor((2, 8, 80)),
                FakeTensor((32, 16, 4, 80)),
                FakeTensor((32, 16, 4, 80)),
                FakeTensor((2, 8), dtype="int32"),
                FakeTensor((2,), dtype="int32"),
                torch_module=self.torch,
                runtime=CaptureRuntime(),
            )
        with self.assertRaisesRegex(ValueError, "equal query and KV sequence"):
            execute_sdpa(
                FakeTensor((1, 8, 64, 128)),
                FakeTensor((1, 2, 128, 128)),
                FakeTensor((1, 2, 128, 128)),
                torch_module=self.torch,
                runtime=CaptureRuntime(),
            )

    def test_public_package_exports_do_not_import_torch(self) -> None:
        self.assertTrue(callable(rk.rms_norm))
        self.assertTrue(callable(rk.add_rms_norm))
        self.assertTrue(callable(rk.rope))
        self.assertTrue(callable(rk.silu_mul))
        self.assertTrue(callable(rk.softmax))
        self.assertTrue(callable(rk.gemv))
        self.assertTrue(callable(rk.kv_cache_append))
        self.assertTrue(callable(rk.kv_cache_copy))
        self.assertTrue(callable(rk.paged_attention_decode))
        self.assertTrue(callable(rk.sdpa))


class LlmOperatorArtifactTests(unittest.TestCase):
    def test_contracts_validate_against_public_schema(self) -> None:
        root = Path(__file__).parents[1]
        schema = json.loads((root / "schemas" / "operator-contract.schema.json").read_text())
        validator = Draft202012Validator(schema)
        contracts = sorted((root / "contracts").glob("*.json"))

        self.assertEqual(len(contracts), 10)
        for contract in contracts:
            with self.subTest(contract=contract.name):
                validator.validate(json.loads(contract.read_text()))

    def test_dispatch_registry_includes_all_milestone_five_operators(self) -> None:
        keys = set(builtin_dispatch_registry().keys())

        self.assertTrue(
            {
                ("rms_norm", "1.0"),
                ("add_rms_norm", "1.0"),
                ("rope", "1.0"),
                ("swiglu", "1.0"),
                ("softmax", "1.0"),
                ("gemv", "1.0"),
                ("kv_cache_append", "1.0"),
                ("kv_cache_copy", "1.0"),
                ("paged_attention_decode", "1.0"),
                ("sdpa", "1.0"),
            }.issubset(keys)
        )

    def test_workload_documents_cover_every_operator_and_dtype(self) -> None:
        root = Path(__file__).parents[1] / "benchmarks" / "workloads"
        documents = [yaml.safe_load(path.read_text()) for path in sorted(root.glob("*.yaml"))]

        self.assertEqual(
            {document["operator"] for document in documents},
            {
                "rms_norm",
                "add_rms_norm",
                "rope",
                "swiglu",
                "softmax",
                "gemv",
                "kv_cache_append",
                "kv_cache_copy",
                "paged_attention_decode",
                "sdpa",
            },
        )
        self.assertTrue(all(document["dtypes"] == ["fp16", "bf16"] for document in documents))
        self.assertTrue(all(len(document["cases"]) >= 4 for document in documents))


if __name__ == "__main__":
    unittest.main()
