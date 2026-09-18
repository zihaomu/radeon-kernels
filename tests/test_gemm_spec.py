from __future__ import annotations

import unittest

from radeon_kernels.ops.gemm import DType, GemmWorkload, Layout, ShapeRange


class GemmSpecTests(unittest.TestCase):
    def test_workload_defaults_match_methodology(self) -> None:
        workload = GemmWorkload(
            id="gemm-fp16-nt",
            dtype=DType.FP16,
            layout=Layout.NT,
            m=ShapeRange(1, 4096),
            n=ShapeRange(1, 4096),
            k=ShapeRange(1, 4096),
        )

        self.assertEqual(workload.warmups, 10)
        self.assertEqual(workload.samples, 30)
        self.assertEqual(workload.minimum_improvement_pct, 3.0)

    def test_rejects_invalid_shape_range(self) -> None:
        with self.assertRaises(ValueError):
            ShapeRange(16, 8)


if __name__ == "__main__":
    unittest.main()

