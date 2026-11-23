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
    score_blind_run,
    write_blind_pack,
    write_png_rgb,
)
from test_blind_gk import write_fake_gk  # noqa: E402


class CreatureBlindTest(unittest.TestCase):
    def setUp(self):
        self._gk_dir = tempfile.mkdtemp(prefix="cblind_gk_")
        self._gk_log = os.path.join(self._gk_dir, "gk.log")
        self._fake_gk = os.path.join(self._gk_dir, "gk")
        write_fake_gk(self._fake_gk, self._gk_log)
        self._old_gk = os.environ.get("GK_BIN")
        os.environ["GK_BIN"] = self._fake_gk

    def tearDown(self):
        if self._old_gk is None:
            os.environ.pop("GK_BIN", None)
        else:
            os.environ["GK_BIN"] = self._old_gk
        shutil.rmtree(self._gk_dir, ignore_errors=True)

    def test_seeded_order_delegated_to_gk_and_reproducible(self):
        tmp = tempfile.mkdtemp(prefix="test_cblind_seed_")
        try:
            p1 = os.path.join(tmp, "p1.png")
            p2 = os.path.join(tmp, "p2.png")
            write_png_rgb(p1, 16, 12, bytearray([80, 120, 160] * (16 * 12)))
            write_png_rgb(p2, 16, 12, bytearray([180, 70, 60] * (16 * 12)))
            items = [
                {"category": "claim", "subject": "fox", "a_name": "A", "b_name": "B", "a_pngs": [p1], "b_pngs": [p2]},
                {"category": "control", "subject": "fox", "a_name": "same_a", "b_name": "same_b", "a_pngs": [p1], "b_pngs": [p1], "expected": "cant_tell"},
            ]
            k1_path = write_blind_pack(items, os.path.join(tmp, "pack1"), 20261116)
            k2_path = write_blind_pack(items, os.path.join(tmp, "pack2"), 20261116)
            with open(k1_path) as f1, open(k2_path) as f2:
                self.assertEqual(json.load(f1)["pairs"], json.load(f2)["pairs"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

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
    """skintokens-10: `write_blind_pack` replaces only a blind pack it made (its judge.json
    or pairs.json inside, its key beside); any other folder named by --out is refused, untouched."""

    def setUp(self):
        self._gk_dir = tempfile.mkdtemp(prefix="cblind_refuse_gk_")
        self._fake_gk = os.path.join(self._gk_dir, "gk")
        write_fake_gk(self._fake_gk, os.path.join(self._gk_dir, "gk.log"))
        self._old_gk = os.environ.get("GK_BIN")
        os.environ["GK_BIN"] = self._fake_gk

    def tearDown(self):
        if self._old_gk is None:
            os.environ.pop("GK_BIN", None)
        else:
            os.environ["GK_BIN"] = self._old_gk
        shutil.rmtree(self._gk_dir, ignore_errors=True)

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
            with open(os.path.join(half, "judge.json"), "w") as f:
                f.write("{}")
            with self.assertRaises((ValueError, RuntimeError, OSError, SystemExit)):
                write_blind_pack(items, half, 1)
            self.assertTrue(os.path.exists(os.path.join(half, "judge.json")))

            # Unrelated judge.json + .key.json without blind-pack version/pairs markers are refused too.
            unrelated = os.path.join(tmp, "unrelated")
            os.makedirs(unrelated)
            with open(os.path.join(unrelated, "judge.json"), "w") as f:
                f.write('{"unrelated": true}')
            with open(unrelated + ".key.json", "w") as f:
                f.write('{"unrelated": true}')
            with open(os.path.join(unrelated, "keep.txt"), "w") as f:
                f.write("keep")
            with self.assertRaises((ValueError, RuntimeError, OSError, SystemExit)):
                write_blind_pack(items, unrelated, 1)
            self.assertTrue(os.path.exists(os.path.join(unrelated, "keep.txt")))

            # Passing "/" must not resolve to os.getcwd() via rstrip("/").
            pack = os.path.join(tmp, "cwd_pack")
            write_blind_pack(items, pack, 1)
            old_cwd = os.getcwd()
            try:
                os.chdir(pack)
                with self.assertRaises((ValueError, RuntimeError, OSError, PermissionError, SystemExit)):
                    write_blind_pack(items, "/", 2)
                self.assertTrue(os.path.exists(os.path.join(pack, "judge.json")))
            finally:
                os.chdir(old_cwd)
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


class CameraFovTest(unittest.TestCase):
    def test_default_follow_fov_is_75_never_45(self):
        # Static guard: neither creature_blind.py nor render_gamecam.py may hardcode 45 deg FOV.
        tools_dir = os.path.dirname(os.path.abspath(__file__))
        for rel in ("creature_blind.py", os.path.join("bench", "render_gamecam.py")):
            with open(os.path.join(tools_dir, rel), encoding="utf-8") as f:
                src = f.read()
            self.assertNotIn("math.radians(45", src, f"{rel} still hardcodes 45 deg FOV")
            self.assertNotIn("45 deg", src, f"{rel} docs still claim 45 deg FOV")

        import creature_blind  # noqa: E402
        self.assertTrue(hasattr(creature_blind, "resolve_follow_fov_deg"), "missing resolve_follow_fov_deg")
        old_hlh = os.environ.pop("HLH_METRICS", None)
        old_met = os.environ.pop("METRICS_FILE", None)
        try:
            self.assertAlmostEqual(creature_blind.resolve_follow_fov_deg(), 75.0)
            with tempfile.TemporaryDirectory() as tmp:
                mpath = os.path.join(tmp, "metrics.ron")
                with open(mpath, "w", encoding="utf-8") as f:
                    f.write('// comment\n{\n  "camera": {"distance": 3.0, "focus": 1.5, "pitch": -18.3, "fov": 68.0},\n}\n')
                self.assertAlmostEqual(creature_blind.resolve_follow_fov_deg(metrics_path=mpath), 68.0)
                self.assertAlmostEqual(creature_blind.resolve_follow_fov_deg(metrics_path=mpath, fov_deg=72.0), 72.0)

            with tempfile.TemporaryDirectory() as tmp:
                import toolpaths
                orig_run = toolpaths.run
                seen_cmds = []

                def fake_run(cmd, timeout=None):
                    seen_cmds.append(list(cmd))
                    if "--measure-only" in cmd:
                        spec = {"cam_spec": [-0.5, 0.5, -0.5, 0.5, 2.0, 0.0], "sampled_times": [0.0], "clip": "Walk"}
                        return {"code": 0, "out": "RENDER_GAMECAM_SPEC " + json.dumps(spec) + "\n", "err": ""}
                    prefix = cmd[cmd.index("--out") + 1]
                    png = f"{prefix}_1.png"
                    write_png_rgb(png, 4, 4, bytearray([128, 128, 128] * 16))
                    return {"code": 0, "out": "", "err": ""}

                toolpaths.run = fake_run
                try:
                    a_pngs, b_pngs, info = creature_blind.render_pair_spec(
                        "blender", "render_gamecam.py", "a.glb", "b.glb", "Walk", "0.0", None, tmp, "tag",
                    )
                    self.assertEqual(len(a_pngs), 1)
                    self.assertEqual(len(b_pngs), 1)
                    self.assertAlmostEqual(info["fov_deg"], 75.0)
                    self.assertTrue(all("--python-exit-code" in c for c in seen_cmds))
                finally:
                    toolpaths.run = orig_run
        finally:
            if old_hlh is not None:
                os.environ["HLH_METRICS"] = old_hlh
            if old_met is not None:
                os.environ["METRICS_FILE"] = old_met


if __name__ == "__main__":
    unittest.main()
