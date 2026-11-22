#!/usr/bin/env python3
"""Unit tests for skintokens-7 humanoid-route five-finger hands helper."""

import os
import sys
import unittest

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import humanoid_hands as hh  # noqa: E402


class HumanoidHandsTest(unittest.TestCase):
    def test_count_finger_chains_distinguishes_five_vs_mitten(self):
        self.assertEqual(hh.count_finger_chains(["DEF-hand.L", "DEF-hand.R"]), {"L": 0, "R": 0})
        five = [
            f"DEF-{c}.01.{s}"
            for s in ("L", "R")
            for c in ("thumb", "f_index", "f_middle", "f_ring", "f_pinky")
        ]
        self.assertEqual(hh.count_finger_chains(five), {"L": 5, "R": 5})

    def test_parse_grip_gate_output_extracts_closed_and_planted(self):
        sample = """
  five_finger    closed/cleave     : score=0.966 palm=1.2cm wrap=1.000 curl=33.8deg skin=True
  five_finger    closed/backcut    : score=0.966 palm=1.2cm wrap=1.000 curl=33.8deg skin=True
  five_finger    closed/sweep      : score=0.966 palm=1.2cm wrap=1.000 curl=33.8deg skin=True
  five_finger    closed/thrust     : score=0.966 palm=1.2cm wrap=1.000 curl=33.8deg skin=True
  five_finger    planted_open      : score=0.000 palm=0.7cm wrap=0.792 curl=0.0deg skin=True
  five_finger    planted_fingertip : score=0.000 palm=10.6cm wrap=0.000 curl=33.8deg skin=True
  five_finger    planted_10cm_off  : score=0.000 palm=8.6cm wrap=0.000 curl=33.8deg skin=True
"""
        parsed = hh.parse_grip_gate_output(sample)
        self.assertEqual(len(parsed["closed"]), 4)
        self.assertTrue(all(v >= 0.90 for v in parsed["closed"].values()))
        self.assertEqual(len(parsed["planted"]), 3)
        self.assertTrue(all(v < 0.30 for v in parsed["planted"].values()))

    def test_extract_hand_findings_and_parity_check(self):
        base = {
            "headline": [
                {
                    "name": "game clips",
                    "findings": [
                        {"severity": "fail", "code": "D_STRETCH", "part": "right thigh", "area_cm2": 120.0},
                        {"severity": "warn", "code": "D_LIMB_THROUGH", "part": "arm (right) into right fingers", "area_cm2": 40.0},
                    ],
                }
            ]
        }
        better = {
            "headline": [
                {
                    "name": "game clips",
                    "findings": [
                        {"severity": "fail", "code": "D_STRETCH", "part": "right thigh", "area_cm2": 120.0},
                    ],
                }
            ]
        }
        worse_fail = {
            "headline": [
                {
                    "name": "poses",
                    "findings": [
                        {"severity": "fail", "code": "D_BLEED", "part": "left hand", "area_cm2": 15.0},
                    ],
                }
            ]
        }
        b_m = hh.extract_hand_findings(base)
        g_m = hh.extract_hand_findings(better)
        w_m = hh.extract_hand_findings(worse_fail)
        self.assertEqual((b_m["fails"], b_m["total"], b_m["area_cm2"]), (0, 1, 40.0))
        self.assertTrue(hh.hand_findings_no_worse(b_m, g_m))
        self.assertFalse(hh.hand_findings_no_worse(b_m, w_m))

    def test_missing_input_returns_usage_code_2(self):
        code = hh.main(["--input", "/nonexistent/missing.glb", "--out", "/tmp/out.glb"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
