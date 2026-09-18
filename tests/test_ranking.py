from __future__ import annotations

import unittest

from radeon_kernels.search.ranking import RankingDecision, rank_challenger


class RankingTests(unittest.TestCase):
    def test_promotes_stable_clear_improvement(self) -> None:
        current = [10.0 + (index % 3) * 0.01 for index in range(30)]
        challenger = [9.0 + (index % 3) * 0.01 for index in range(30)]

        result = rank_challenger(current, challenger, seed=11)

        self.assertEqual(result.decision, RankingDecision.PROMOTE)
        self.assertGreater(result.confidence_lower_pct, 3.0)

    def test_keeps_clearly_slower_current_challenger(self) -> None:
        current = [9.0 + (index % 3) * 0.01 for index in range(30)]
        challenger = [10.0 + (index % 3) * 0.01 for index in range(30)]

        result = rank_challenger(current, challenger, seed=11)

        self.assertEqual(result.decision, RankingDecision.KEEP_CURRENT)

    def test_noisy_samples_require_review(self) -> None:
        current = [5.0 if index % 2 else 15.0 for index in range(30)]
        challenger = [4.5 if index % 2 else 14.0 for index in range(30)]

        result = rank_challenger(current, challenger, seed=11)

        self.assertEqual(result.decision, RankingDecision.NEEDS_REVIEW)


if __name__ == "__main__":
    unittest.main()

