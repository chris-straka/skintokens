#!/usr/bin/env python3
"""Shared 3-tool rig judge (`creature-rig check`, `weights check`, `creature-bench anatomy`)."""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
import pick_score  # noqa: E402
import toolpaths  # noqa: E402


def judge(cr, wf, cb, glb, work, tag, cut_at_limit=False):
    """Run skeleton, bend, and anatomy checks on `glb`; raise `CheckDidNotRun` on tool crash."""
    rom = os.path.join(work, tag + ".rom.ron")
    sj = os.path.join(work, tag + ".skel.json")
    aj = os.path.join(work, tag + ".anatomy.json")

    def call(cmd):
        r = toolpaths.run(cmd, timeout=1800)
        return r["code"], r["out"], r["err"]

    sk = pick_score.checked_json("creature-rig check", call([cr, "check", glb, "--json", sj, "--rom", rom]), sj)
    bend = pick_score.checked_json("weights check", call([wf, "check", glb, "--json", "--poses", rom]))
    an = pick_score.checked_json("creature-bench anatomy", call([cb, "anatomy", glb, "--json", aj]), aj)
    tears = pick_score.tears_pct(bend)
    score = pick_score.rig_score(sk, bend, an)
    oscore = pick_score.old_score(bend.get("score", 0.0), sk.get("score", 0.0), bool(sk.get("pass")))
    an_sum = pick_score.anatomy_summary(an)
    return {
        "skeleton": sk.get("score", 0.0),
        "skeleton_pass": bool(sk.get("pass")),
        "blended_share": sk.get("blended_share"),
        "anatomy_pass": an_sum["pass"],
        "anatomy_fails": an_sum["codes"],
        "articulated_pairs": an_sum["articulated_pairs"],
        "bend": bend.get("score", 0.0),
        "tears_pct": tears,
        "old": oscore,
        "old_score": oscore,
        "rig_score": score,
        "anatomy": an,
        "skeleton_raw": sk,
        "cut_at_token_limit": cut_at_limit,
    }
