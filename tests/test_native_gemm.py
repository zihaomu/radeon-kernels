from __future__ import annotations

import unittest

from radeon_kernels.ops.gemm.native import VARIANT_NAMES, gemm


class NativeGemmTests(unittest.TestCase):
    def test_variant_names_are_stable_and_unique(self) -> None:
        self.assertEqual(len(VARIANT_NAMES), 11)
        self.assertEqual(len(set(VARIANT_NAMES)), len(VARIANT_NAMES))

    def test_rejects_invalid_variant_before_loading_torch(self) -> None:
        with self.assertRaisesRegex(ValueError, "variant must be between"):
            gemm(None, None, variant=-1)


if __name__ == "__main__":
    unittest.main()
