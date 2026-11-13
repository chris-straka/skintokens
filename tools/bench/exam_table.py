#!/usr/bin/env python3
"""The exam's results as a table (stdlib only).

    python3 tools/bench/exam_table.py BENCH [names...] > exam.md

Reads BENCH/exam/<name>/ as tools/bench/exam.sh leaves it: the joint
finder's training numbers (run.json), motionforge's exam scores per seed
(exam/exam.json) and creature-bench's joint-by-joint comparisons of the
finder's answer, the outline follower's and a rig that does not move.
"""
import json
import os
import statistics
import sys


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    bench = sys.argv[1]
    root = os.path.join(bench, "exam")
    names = sys.argv[2:] or sorted(os.listdir(root))
    out = ["# The video-to-motion exam on the route's skeletons", "",
           "For each creature: a joint finder trained on 3,000 renders of that creature, then three made-up motions "
           "(every joint moving at random within its limits) filmed by the two cameras of the start picture, read back and "
           "compared joint by joint with the motion that was made. No video model is involved: the films are our own renders, "
           "made worse the way a video model does (blur, light, colour, noise).", "",
           "The rows, on the same three motions:", "",
           "- **finder**: the joint finder on both pictures, the two rays crossed, solved onto the skeleton (motionforge).",
           "- **outline**: the side picture alone, the rig turned until its outline lies on the film's (`creature-bench follow`, what part B did with the paid clips).",
           "- **still**: the rig standing in its rest pose, for scale.", "",
           "Joint error is the distance from each joint to where the made-up motion put it, averaged over joints and frames, "
           "in millimetres and as a share of the creature's height.", "",
           "| Creature | Height | Finder (held-out pictures) | Finder: joints | Outline: joints | Still: joints | Finder's worst joints |",
           "|---|---|---|---|---|---|---|"]
    details = []
    for n in names:
        d = os.path.join(root, n)
        run = load(os.path.join(d, "run.json")) or {}
        last = (run.get("log") or [{}])[-1]
        val = last.get("val_degraded") or last.get("val") or {}
        exam = load(os.path.join(d, "exam", "exam.json")) or {}
        rows = {"finder": [], "outline": [], "rest": []}
        worst = {}
        height = None
        overlaps = []
        for s in (1, 2, 3):
            e = os.path.join(d, "exam", n, f"seed{s}")
            for k in rows:
                j = load(os.path.join(e, f"joints_{k}.json"))
                if j:
                    rows[k].append(j)
                    height = j.get("height_m", height)
            f = load(os.path.join(e, "follow.json"))
            if f:
                overlaps.append(f.get("overlap_mean"))
            j = load(os.path.join(e, "joints_finder.json"))
            if j:
                for x in j["joints"]:
                    worst.setdefault(x["joint"], []).append(x["mean_mm"])

        def cell(k):
            xs = rows[k]
            if not xs:
                return "not run"
            mm = statistics.mean(x["mean_mm"] for x in xs)
            pct = statistics.mean(x["mean_pct_of_height"] for x in xs)
            return f"{mm:.0f} mm ({pct:.1f}%)"
        w = sorted(((statistics.mean(v), k) for k, v in worst.items()), reverse=True)[:3]
        train = f"{val.get('mean_px', float('nan')):.1f} px mean, {100 * val.get('pck_10px', 0):.0f}% within 10 px" if val else "not trained"
        out.append(f"| {n} | {height:.2f} m | {train} | {cell('finder')} | {cell('outline')} | {cell('rest')} | "
                   + ", ".join(f"{k.replace('DEF-', '')} {v:.0f} mm" for v, k in w) + " |" if height else f"| {n} | | {train} | not run | | | |")
        c = (exam.get("creatures") or {}).get(n)
        if c:
            seeds = c.get("seeds", {})
            agree = all(v.get("solve", {}).get("consistent") for v in seeds.values())
            details.append(f"- **{n}**: motionforge's own scores over {len(seeds)} motions: joints {c['mean_mm']:.0f} mm, inner bones "
                           f"{c['mean_deg_inner']:.1f} degrees off, worst joint {c['max_mm']:.0f} mm; the two views agreed on "
                           f"{'every' if agree else 'not every'} motion. Outline overlap reached by the follower: "
                           + ", ".join(f"{x:.2f}" for x in overlaps if x is not None) + ".")
    out += ["", "## Per creature", ""] + details + [""]
    print("\n".join(out))


main()
