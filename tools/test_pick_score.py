#!/usr/bin/env python3
"""Unit tests for `tools/pick_score.py`: `python3 -m unittest discover -s tools -p 'test_*.py'`."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pick_score import (  # noqa: E402
    fair_score,
    old_score,
    pick_candidate,
    rig_score,
    tears_pct,
    trusted,
)

CLEAN_ANATOMY = {
    "pass": True,
    "findings": [],
    "limbs": [
        {"from": "f.L", "side": "+x", "partner": "f.R", "joints": 3},
        {"from": "f.R", "side": "-x", "partner": "f.L", "joints": 3},
        {"from": "h.L", "side": "+x", "partner": "h.R", "joints": 4},
        {"from": "h.R", "side": "-x", "partner": "h.L", "joints": 4},
    ],
}

SWAPPED_ANATOMY = {
    "pass": False,
    "findings": [{"code": "B_SWAPPED", "severity": "fail"}],
    "limbs": CLEAN_ANATOMY["limbs"],
}


class PickScoreTest(unittest.TestCase):
    def test_tears_counts_only_failing_tear_verts(self):
        bend = {
            "verts": 1398,
            "findings": [
                {"code": "D_STRETCH", "severity": "fail", "verts": 40},
                {"code": "D_VOLUME", "severity": "fail", "verts": 24},
                {"code": "D_NOISE", "severity": "fail", "verts": 90},
                {"code": "D_STRETCH", "severity": "warn", "verts": 10},
            ],
        }
        self.assertEqual(tears_pct(bend), round(100.0 * 64 / 1398, 2))

    def test_hand_made_rigs_score_over_90_and_beat_crude_and_planted(self):
        # Even on low-poly fox (where artist tears 4.83% vs crude 4.48%),
        # artist blends joints (blended_share=0.28) while crude is rigid (0.0).
        artist = rig_score(
            {"score": 97.3, "pass": True, "blended_share": 0.28},
            4.83,
            CLEAN_ANATOMY,
        )
        crude = rig_score(
            {"score": 97.3, "pass": True, "blended_share": 0.0},
            4.48,
            CLEAN_ANATOMY,
        )
        planted = rig_score(
            {"score": 97.3, "pass": True, "blended_share": 0.28},
            22.0,
            SWAPPED_ANATOMY,
        )
        self.assertGreaterEqual(artist, 90.0)
        self.assertGreater(artist, crude + 5.0)
        self.assertLess(planted, 65.0)
        self.assertLess(old_score(11.6, 15.1, False), 0.0)

    def test_picker_filters_bad_seeds_and_avoids_circular_body_bias(self):
        # Fox / cow pattern: body gets 100 on its own lines, predicted gets 96
        # with clean anatomy and 2 limb pairs -> keeps predicted (learned base).
        body = {
            "skeleton": {"score": 100.0, "pass": True, "blended_share": 0.35},
            "tears_pct": 0.0,
            "anatomy": CLEAN_ANATOMY,
        }
        bad_seed = {
            "seed": 3,
            "rigged": True,
            "skeleton": {"score": 98.0, "pass": True, "blended_share": 0.35},
            "tears_pct": 0.0,
            "anatomy": SWAPPED_ANATOMY,
        }
        good_seed = {
            "seed": 2,
            "rigged": True,
            "skeleton": {"score": 95.5, "pass": True, "blended_share": 0.35},
            "tears_pct": 0.2,
            "anatomy": CLEAN_ANATOMY,
        }
        res = pick_candidate(body, [bad_seed, good_seed])
        self.assertEqual(res["which"], "predicted")
        self.assertEqual(res["best_predicted"]["seed"], 2)
        self.assertTrue(res["trusted"])

    def test_picker_rescues_with_body_when_predicted_fails_anatomy_or_cuts(self):
        body = {
            "skeleton": {"score": 96.0, "pass": True, "blended_share": 0.30},
            "tears_pct": 0.4,
            "anatomy": CLEAN_ANATOMY,
        }
        broken_pred = {
            "seed": 1,
            "rigged": True,
            "cut_at_token_limit": True,
            "skeleton": {"score": 94.0, "pass": True, "blended_share": 0.30},
            "tears_pct": 0.5,
            "anatomy": CLEAN_ANATOMY,
        }
        res = pick_candidate(body, [broken_pred])
        self.assertEqual(res["which"], "body")
        self.assertTrue(res["trusted"])

    def test_trust_margin_and_anatomy_gate(self):
        self.assertFalse(trusted(89.9))
        self.assertTrue(trusted(93.0))
        self.assertFalse(trusted(95.0, anatomy_ok=False))
        self.assertFalse(trusted(96.0, 94.5))
        self.assertEqual(fair_score(100.0, 0.0, True, CLEAN_ANATOMY, 0.3), 100.0)


if __name__ == "__main__":
    unittest.main()
