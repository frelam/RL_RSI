from __future__ import annotations

import math
import unittest

from rsi.core.scoring import pass_at_k, score_text


class ScoringTests(unittest.TestCase):
    def test_pass_at_k_boundaries(self) -> None:
        self.assertEqual(pass_at_k(1, 1, 1), 1.0)
        self.assertEqual(pass_at_k(1, 0, 1), 0.0)
        self.assertEqual(pass_at_k(10, 10, 5), 1.0)
        self.assertEqual(pass_at_k(10, 0, 5), 0.0)
        self.assertAlmostEqual(pass_at_k(10, 5, 1), 0.5)
        self.assertEqual(pass_at_k(0, 0, 1), 0.0)

    def test_scorers(self) -> None:
        self.assertEqual(score_text("The answer is 42.", "42", "numeric"), 1.0)
        self.assertEqual(score_text("42", "42.5", "numeric"), 0.0)
        self.assertEqual(score_text("a b c", "b", "contains"), 1.0)
        self.assertEqual(score_text("ABC", "abc", "exact"), 1.0)
        self.assertEqual(score_text("v1.2.3", r"v\d+\.\d+", "regex"), 1.0)
        self.assertEqual(score_text("x", None, "contains"), 0.0)
        with self.assertRaises(ValueError):
            score_text("x", "x", "unknown")


if __name__ == "__main__":
    unittest.main()
