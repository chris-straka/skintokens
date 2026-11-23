#!/usr/bin/env python3
"""Unit tests for `tools/pick_score.py`: `python3 -m unittest discover -s tools -p 'test_*.py'`."""
import os
import stat
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pick_score import (  # noqa: E402
    CheckDidNotRun,
    anatomy_summary,
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

    def test_subtle_falloff_anatomy_fails_rig_score_and_picker(self):
        falloff_anatomy = {
            "pass": False,
            "findings": [{"code": "B_FALLOFF", "severity": "fail"}],
            "limbs": CLEAN_ANATOMY["limbs"],
        }
        an = anatomy_summary(falloff_anatomy)
        self.assertFalse(an["pass"])
        self.assertIn("B_FALLOFF", an["codes"])
        score = rig_score({"score": 96.0, "pass": True, "blended_share": 0.30}, 0.5, falloff_anatomy)
        self.assertLess(score, 80.0)
        body = {"skeleton": {"score": 96.0, "pass": True, "blended_share": 0.30}, "tears_pct": 0.5, "anatomy": CLEAN_ANATOMY}
        bad = {"seed": 1, "rigged": True, "skeleton": {"score": 97.0, "pass": True, "blended_share": 0.30}, "tears_pct": 0.2, "anatomy": falloff_anatomy}
        res = pick_candidate(body, [bad])
        self.assertEqual(res["which"], "body")



HERE = os.path.dirname(os.path.abspath(__file__))


def fake_tool(folder, name, code, writes=None):
    """A stand-in binary: exits `code`; with `writes`, first writes that JSON to the path
    after `--json` in its arguments."""
    path = os.path.join(folder, name)
    body = "#!/bin/sh\n"
    if writes is not None:
        body += 'while [ $# -gt 0 ]; do if [ "$1" = --json ]; then printf \'%s\' \'' + writes + '\' > "$2"; fi; shift; done\n'
    body += f"exit {code}\n"
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


class MissingChecksFailTest(unittest.TestCase):
    """skintokens-3: a check that did not run is a failure, never a pass."""

    def test_missing_anatomy_never_passes(self):
        for missing in (None, {}):
            an = anatomy_summary(missing)
            self.assertFalse(an["present"])
            self.assertFalse(an["pass"], f"anatomy_summary({missing!r}) passes a check that never ran")

    def test_picker_never_trusts_a_rig_whose_anatomy_did_not_run(self):
        body = {"skeleton": {"score": 99.0, "pass": True, "blended_share": 0.35}, "tears_pct": 0.0, "anatomy": None}
        pred = {"seed": 1, "rigged": True, "skeleton": {"score": 95.0, "pass": True, "blended_share": 0.35},
                "tears_pct": 0.3, "anatomy": CLEAN_ANATOMY}
        res = pick_candidate(body, [pred])
        self.assertEqual(res["which"], "predicted", res["reason"])
        self.assertTrue(res["trusted"])
        # Neither checked: whichever it keeps, it does not trust it.
        res = pick_candidate(body, [dict(pred, anatomy=None)])
        self.assertFalse(res["trusted"], res)
        # Body alone, unchecked.
        self.assertFalse(pick_candidate(body, [])["trusted"])

    def test_route_stops_when_the_anatomy_check_is_missing(self):
        # The route runs from a copy with no sibling builds, so nothing can stand in for the
        # missing creature-bench; Blender is a stand-in that refuses (the route must stop first).
        with tempfile.TemporaryDirectory() as d:
            tools = os.path.join(d, "repo", "tools")
            os.makedirs(tools)
            for f in ("creature_route.py", "pick_score.py"):
                with open(os.path.join(HERE, f)) as src, open(os.path.join(tools, f), "w") as dst:
                    dst.write(src.read())
            env = dict(os.environ, PATH="/usr/bin:/bin", CREATURE_RIG_BIN=fake_tool(d, "creature-rig", 0),
                       CREATURE_BENCH_BIN=os.path.join(d, "no-creature-bench"), BLENDER=fake_tool(d, "blender", 9))
            model = os.path.join(d, "m.glb")
            with open(model, "wb"):
                pass
            r = subprocess.run([sys.executable, os.path.join(tools, "creature_route.py"), model, os.path.join(d, "out"),
                                "--blender", env["BLENDER"]],
                               capture_output=True, text=True, env=env, timeout=120)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("creature-bench", r.stderr)

    def _judges(self):
        sys.path.insert(0, os.path.join(HERE, "bench"))
        import score_artists  # noqa: E402
        import creature_blind  # noqa: E402
        return [("score_artists.judge", score_artists.judge), ("creature_blind.judge_rig_metrics", creature_blind.judge_rig_metrics)]

    def test_judges_raise_when_a_check_did_not_run(self):
        skel = '{"score": 95.0, "pass": true}'
        anat = '{"pass": true, "findings": [], "limbs": []}'
        bend = '{"score": 90.0, "findings": [], "verts": 100}'
        for label, judge in self._judges():
            with tempfile.TemporaryDirectory() as d:
                glb = os.path.join(d, "r.glb")
                with open(glb, "wb"):
                    pass
                wf = os.path.join(d, "weights")
                with open(wf, "w") as f:
                    f.write("#!/bin/sh\necho '" + bend + "'\nexit 1\n")  # a failing rig: exit 1 is a verdict
                os.chmod(wf, 0o755)
                cr = fake_tool(d, "creature-rig", 0, skel)
                good_cb = fake_tool(d, "cb-good", 0, anat)
                # The control: every tool ran (weights' exit 1 is its verdict, not an error).
                judge(cr, wf, good_cb, glb, d, "ok")
                # Each tool crashing (exit 3, nothing written) must raise, never score.
                for bad in ("creature-rig", "creature-bench", "weights"):
                    crash = fake_tool(d, "crash-" + bad, 3)
                    args = {"creature-rig": (crash, wf, good_cb), "creature-bench": (cr, wf, crash),
                            "weights": (cr, crash, good_cb)}[bad]
                    with self.assertRaises(CheckDidNotRun, msg=f"{label}: {bad} crashed and the judge scored anyway"):
                        judge(*args, glb, d, "t_" + bad)


if __name__ == "__main__":
    unittest.main()
