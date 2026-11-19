#!/usr/bin/env python3
"""Unit tests for `tools/creature_blind.py`."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from creature_blind import (  # noqa: E402
    a_on_left,
    presentation_order,
    score_blind_run,
    write_blind_pack,
    write_png_rgb,
)


class CreatureBlindTest(unittest.TestCase):
    def test_seeded_coin_and_order_are_reproducible_and_mixed(self):
        coins = [a_on_left(20261116, k) for k in range(1, 12)]
        self.assertEqual(coins, [a_on_left(20261116, k) for k in range(1, 12)])
        self.assertTrue(any(coins) and not all(coins))
        order = presentation_order(6, 20261116)
        self.assertEqual(order, presentation_order(6, 20261116))
        self.assertEqual(sorted(order), [0, 1, 2, 3, 4, 5])

    def test_pack_keeps_key_outside_and_verifies_zero_rmse_control(self):
        tmp = tempfile.mkdtemp(prefix="test_cblind_")
        try:
            p1 = os.path.join(tmp, "p1.png")
            p2 = os.path.join(tmp, "p2.png")
            write_png_rgb(p1, 32, 24, bytearray([80, 120, 160] * (32 * 24)))
            write_png_rgb(p2, 32, 24, bytearray([180, 70, 60] * (32 * 24)))

            items = [
                {
                    "category": "claim",
                    "subject": "fox",
                    "a_name": "predicted",
                    "b_name": "body",
                    "a_pngs": [p1, p1],
                    "b_pngs": [p2, p2],
                    "truth_winner": "predicted",
                    "pick_winner": "predicted",
                    "old_winner": "body",
                },
                {
                    "category": "control",
                    "subject": "fox",
                    "a_name": "same_a",
                    "b_name": "same_b",
                    "a_pngs": [p1, p1],
                    "b_pngs": [p1, p1],
                    "expected": "cant_tell",
                },
                {
                    "category": "planted",
                    "subject": "fox",
                    "a_name": "clean",
                    "b_name": "planted_swapped_limbs",
                    "a_pngs": [p1, p1],
                    "b_pngs": [p2, p2],
                    "expected": "clean",
                    "planted_metric_ok": True,
                },
            ]
            judged = os.path.join(tmp, "pack")
            key_path = write_blind_pack(items, judged, 20261116)
            self.assertEqual(key_path, judged + ".key.json")
            self.assertTrue(os.path.exists(key_path))
            self.assertFalse(any("key" in f for f in os.listdir(judged)))

            with open(key_path) as kf:
                key = json.load(kf)
            answers = {}
            for p in key["pairs"]:
                if p["category"] == "control":
                    self.assertEqual(p["panel_rmse"], 0.0)
                    answers[str(p["pair"])] = "can't tell"
                elif p["category"] == "planted":
                    answers[str(p["pair"])] = "left" if p["left_rig"] == "clean" else "right"
                else:
                    answers[str(p["pair"])] = "left" if p["left_rig"] == "predicted" else "right"

            v = score_blind_run(key, answers)
            self.assertTrue(v["pass"], v)
            self.assertEqual(v["pick_agrees_judge"], 1)
            self.assertEqual(len(v["old_score_retired_pairs"]), 1)
            self.assertEqual(v["score_retired_pairs"], [])

            # Hallucinating a difference on the identical control discards the run.
            ctrl_pair = next(str(p["pair"]) for p in key["pairs"] if p["category"] == "control")
            bad_ctrl = dict(answers, **{ctrl_pair: "left"})
            self.assertFalse(score_blind_run(key, bad_ctrl)["pass"])

            # Missing the planted bad case fails the run.
            plant_pair = next(str(p["pair"]) for p in key["pairs"] if p["category"] == "planted")
            bad_plant = dict(answers, **{plant_pair: "cant_tell"})
            self.assertFalse(score_blind_run(key, bad_plant)["pass"])

            # List-style `{"answers": [{"pair": N, "better": ..., "why": ...}]}` works identically.
            list_answers = {
                "answers": [
                    {"pair": int(k), "better": v.upper(), "why": f"reason {k}"}
                    for k, v in answers.items()
                ]
            }
            v_list = score_blind_run(key, list_answers)
            self.assertTrue(v_list["pass"], v_list)
            self.assertTrue(all("why" in p for p in v_list["pairs"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)



class PackRefusesStrangeFoldersTest(unittest.TestCase):
    """skintokens-10: `write_blind_pack` replaces only a blind pack it made (its pairs.json
    inside, its key beside); any other folder named by --out is refused, untouched. Until
    then it deleted whatever --out named (`shutil.rmtree`)."""

    def _items(self, tmp):
        p1 = os.path.join(tmp, "p1.png")
        write_png_rgb(p1, 8, 6, bytearray([80, 120, 160] * (8 * 6)))
        return [{"category": "control", "subject": "fox", "a_name": "same_a", "b_name": "same_b",
                 "a_pngs": [p1], "b_pngs": [p1], "expected": "cant_tell"}]

    def test_refuses_a_folder_that_is_not_a_pack(self):
        tmp = tempfile.mkdtemp(prefix="test_cblind_refuse_")
        try:
            items = self._items(tmp)
            precious = os.path.join(tmp, "work")
            os.makedirs(precious)
            with open(os.path.join(precious, "notes.txt"), "w") as f:
                f.write("not a blind pack")
            with self.assertRaises((ValueError, RuntimeError, OSError, SystemExit)):
                write_blind_pack(items, precious, 1)
            self.assertTrue(os.path.exists(os.path.join(precious, "notes.txt")), "the folder was deleted")
            # A pack's own folder without its key beside it is refused too.
            half = os.path.join(tmp, "half")
            os.makedirs(half)
            with open(os.path.join(half, "pairs.json"), "w") as f:
                f.write("{}")
            with self.assertRaises((ValueError, RuntimeError, OSError, SystemExit)):
                write_blind_pack(items, half, 1)
            self.assertTrue(os.path.exists(os.path.join(half, "pairs.json")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_replaces_its_own_pack(self):
        tmp = tempfile.mkdtemp(prefix="test_cblind_again_")
        try:
            items = self._items(tmp)
            pack = os.path.join(tmp, "pack")
            write_blind_pack(items, pack, 1)
            key = write_blind_pack(items, pack, 2)  # a second run over its own pack: fine
            self.assertTrue(os.path.exists(key))
            # An empty folder (nothing to lose) is fine too.
            empty = os.path.join(tmp, "empty")
            os.makedirs(empty)
            write_blind_pack(items, empty, 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
