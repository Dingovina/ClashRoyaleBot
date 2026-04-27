from __future__ import annotations

import unittest

from src.runtime.domain.match_exit import MatchExitTracker


class MatchExitTrackerTests(unittest.TestCase):
    def test_streak_resets_after_high_probability(self) -> None:
        t = MatchExitTracker()
        self.assertFalse(
            t.observe_probability(prob=0.1, end_threshold=0.5, confirm_ticks=3, did_check=True)
        )
        self.assertFalse(
            t.observe_probability(prob=0.1, end_threshold=0.5, confirm_ticks=3, did_check=True)
        )
        self.assertTrue(
            t.observe_probability(prob=0.1, end_threshold=0.5, confirm_ticks=3, did_check=True)
        )
        self.assertFalse(t.observe_probability(prob=0.9, end_threshold=0.5, confirm_ticks=3, did_check=True))
        self.assertEqual(t.consecutive_below, 0)

    def test_skipped_probe_does_not_change_streak(self) -> None:
        t = MatchExitTracker()
        t.observe_probability(prob=0.1, end_threshold=0.5, confirm_ticks=3, did_check=True)
        t.observe_probability(prob=0.1, end_threshold=0.5, confirm_ticks=3, did_check=False)
        self.assertEqual(t.consecutive_below, 1)


if __name__ == "__main__":
    unittest.main()
