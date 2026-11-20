"""Tests for skintokens-2 picker baselines in `tools/bench/score_artists.py`."""

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(TOOLS_DIR, "bench")
sys.path.insert(0, BENCH_DIR)

import score_artists  # noqa: E402


class PickerBaselinesTest(unittest.TestCase):
    def test_baselines_report_all_four_agreements(self):
        picker = {
            f"animal_{i}": {
                "truth_winner": "predicted" if i < 9 else "body",
                "picked": "predicted" if i < 9 else ("body" if i < 11 else "predicted"),
                "agrees": i < 11,
                "old_agrees": i < 5,
            }
            for i in range(12)
        }
        loo = {f"animal_{i}": {"agrees": i < 10} for i in range(12)}

        b = score_artists.picker_baselines(picker, loo=loo, seed=7)
        self.assertEqual(b["total"], 12)
        self.assertEqual(b["picker"], 11)
        self.assertEqual(b["leave_one_out"], 10)
        self.assertEqual(b["old_score"], 5)
        self.assertEqual(b["always_predicted"], 9)
        self.assertEqual(b["always_body"], 3)
        self.assertAlmostEqual(b["coin_flip_expected"], 6.0)
        self.assertEqual(b["always_predicted"] + b["always_body"], b["total"])
        self.assertGreaterEqual(b["coin_flip_seeded"], 0)
        self.assertLessEqual(b["coin_flip_seeded"], 12)
        # Same seed is deterministic.
        self.assertEqual(
            b["coin_flip_seeded"],
            score_artists.picker_baselines(picker, loo=loo, seed=7)["coin_flip_seeded"],
        )

        line = score_artists.format_picker_baselines(b)
        for token in [
            "11 of 12",
            "always-predicted: 9 of 12",
            "always-body: 3 of 12",
            "coin-flip: 6.0 of 12",
        ]:
            self.assertIn(token, line)

    def test_all_predicted_truth_surfaces_always_predicted_ceiling(self):
        picker = {
            f"a{i}": {
                "truth_winner": "predicted",
                "picked": "predicted",
                "agrees": True,
                "old_agrees": False,
            }
            for i in range(8)
        }
        b = score_artists.picker_baselines(picker, seed=1)
        self.assertEqual(b["picker"], 8)
        self.assertEqual(b["always_predicted"], 8)
        self.assertEqual(b["always_body"], 0)
        self.assertAlmostEqual(b["coin_flip_expected"], 4.0)

    def test_empty_picker_handles_zero_animals(self):
        b = score_artists.picker_baselines({})
        self.assertEqual(b["total"], 0)
        self.assertEqual(b["picker"], 0)
        self.assertEqual(b["always_predicted"], 0)
        self.assertEqual(b["always_body"], 0)
        self.assertEqual(b["coin_flip_expected"], 0.0)
        self.assertEqual(b["coin_flip_seeded"], 0)


if __name__ == "__main__":
    unittest.main()
