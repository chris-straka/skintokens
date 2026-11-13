#!/usr/bin/env python3
"""The creature rig benchmark, part A: rig routes scored against
hand-made animals (stdlib only).

    python3 tools/creature_bench.py BENCH [names...] [--routes new,old]
        [--out DIR] [--creature-bench BIN] [--weights BIN] [--no-score] [--pictures]

BENCH holds `animals/<name>/` with `truth.glb` and `truth.json` (from
`tools/bench/prep_truth.py`: the artist's rig and clips) and one folder
per route with `<name>.glb` in it (from `tools/bench/run_routes.sh`: the
route's rig of the bare mesh). `animals.tsv` there gives each animal's
kind (name, kind, source file).

For each animal and each route, plus two controls (the artist's own rig,
and the artist's skeleton with every vertex on its nearest bone):

1. `creature-bench score`: where the joints are, and how far the skin
   lands from the artist's when both rigs play the artist's clips.
2. `weights check` (weightforge) on the route's rig carrying those same
   frames, with no pose set: tears = vertices in failing stretch, volume
   and bleed findings, as a share of all vertices.

Writes `<out>/table.md`, `<out>/summary.json` and, per animal,
`animals/<name>/score/<route>.json`, `.clips.glb` and `.tears.json`.
`--pictures` adds `<out>/<name>.png` (the artist's rig and each route's,
painted, at the frame where the first route is farthest from the
artist), `<out>/<name>.mp4` (a loop of one clip) and `<out>/all.jpg`.
What each number means: `docs/creature-bench.md`.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys

TEARS = ("D_STRETCH", "D_VOLUME", "D_BLEED")
CONTROLS = ["artist", "crude"]
LABEL = {"artist": "the artist's rig", "crude": "artist's skeleton, crude weights", "new": "today's route", "old": "old route",
         "new2": "today's route, mirror-plane reading"}


def find(name, explicit, env, fallbacks):
    for p in [explicit, os.environ.get(env), shutil.which(name)] + fallbacks:
        if p and os.path.exists(p):
            return p
    raise SystemExit(f"{name} not found: pass --{name}, set {env}, or put it on PATH")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def tears(wf, glb, empty, out):
    """The bend checker on the rig's clips alone."""
    code, text, err = run([wf, "check", glb, "--json", "--poses", empty])
    try:
        d = json.loads(text)
    except ValueError:
        return {"error": (err or text).strip()[-200:]}
    with open(out, "w") as f:
        f.write(text)
    fails = [x for x in d.get("findings", []) if x.get("severity") == "fail"]
    v = max(d.get("verts", 0), 1)
    torn = sum(x["verts"] for x in fails if x["code"] in TEARS)
    return {
        "tears_pct": round(min(100.0, 100.0 * torn / v), 2),
        "noise_pct": round(min(100.0, 100.0 * sum(x["verts"] for x in fails if x["code"] == "D_NOISE") / v), 2),
        "frames_checked": d.get("poses"),
        "findings": [f"{x['code']} {x['region']} ({x['verts']})" for x in fails if x["code"] in TEARS],
    }


def score_animal(cb, wf, d, name, routes, empty):
    truth = os.path.join(d, "truth.glb")
    sd = os.path.join(d, "score")
    os.makedirs(sd, exist_ok=True)
    rigid = os.path.join(sd, "crude.glb")
    code, out, err = run([cb, "rigid", truth, rigid])
    if code != 0:
        raise SystemExit(f"{name}: {err.strip()}")
    rows = {}
    for r in CONTROLS + routes:
        glb = {"artist": truth, "crude": rigid}.get(r) or os.path.join(d, r, name + ".glb")
        if not os.path.exists(glb):
            rows[r] = {"error": "no rig"}
            continue
        js, clips = os.path.join(sd, r + ".json"), os.path.join(sd, r + ".clips.glb")
        cmd = [cb, "score", truth, glb, "--json", js, "--clips", clips, "--label", f"{name} {r}"]
        roles = os.path.join(d, "roles.json")
        if os.path.exists(roles):
            cmd += ["--roles", roles]
        code, out, err = run(cmd)
        if code != 0:
            rows[r] = {"error": err.strip()[-200:]}
            continue
        with open(js) as f:
            rep = json.load(f)
        s, k = rep["skeleton"], rep["skin"]
        row = {
            "joints_pct": s["joint_error_pct"], "joints_max_pct": s["joint_error_max_pct"],
            "limb_joints": s["limb_joints"], "without_a_joint": s["limb_joints_without_a_joint"],
            "missing": s["missing_chains"], "extra_chains": s["extra_chains"], "bones": s["bones"]["route"],
            "skin_mean_pct": k["mean_pct"], "skin_p95_pct": k["p95_pct"],
            "carried_mean_pct": rep["skin_rotations_carried"]["mean_pct"], "carried_p95_pct": rep["skin_rotations_carried"]["p95_pct"],
            "weights_only_mean_pct": rep["skin_weights_only"]["mean_pct"], "weights_only_p95_pct": rep["skin_weights_only"]["p95_pct"],
            "worst": k["worst_frame"], "where": [w["under"] for w in k["where"][:3]],
            "feet_off_pct": rep["feet"]["off_mean_pct"], "feet_sink_pct": rep["feet"]["sink_max_pct"],
            "mesh_off_pct": rep["align"]["mesh_off_mean_pct"], "frames": rep["frames"],
        }
        row.update(tears(wf, clips, empty, os.path.join(sd, r + ".tears.json")))
        rows[r] = row
        print(f"  {name:13s} {r:7s} joints {row['joints_pct']:5.2f}%  missing {len(row['missing'])}  skin {row['skin_mean_pct']:5.2f}% / {row['skin_p95_pct']:5.2f}%"
              f"  carried {row['carried_mean_pct']:5.2f}%  weights only {row['weights_only_mean_pct']:5.2f}%  tears {row.get('tears_pct', '?')}%  feet {row['feet_off_pct']}%")
    return rows


def short(label):
    """`leg (FrontLeg.L to FrontLowLeg.L, 3 bones)` -> `leg FrontLeg.L`"""
    role, rest = label.split(" (", 1)
    return role + " " + rest.split(" to ")[0].rstrip(")")


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def table(summary, routes, kinds):
    cols = CONTROLS[1:] + routes
    lines = [
        "# Creature rig benchmark, part A: routes against hand-made animals",
        "",
        "Each animal is a hand-made rig with its own clips (walk, run, attack, idle, death). Its mesh was stripped bare,",
        "each route rigged it, and both rigs then played the artist's clips. All lengths are shares of the body's length.",
        "",
        "- **joints**: mean distance from the artist's leg, neck, head, jaw, tail and wing joints to the route's joint there.",
        "- **missing**: chains the artist has that no bone of the route's lies along.",
        "- **skin**: how far the skin lands from where the artist's rig puts it, mean / 95th percentile over the vertices, averaged over frames and clips.",
        "- **weights only**: the same with every bone carried bodily by the artist's bone, so joint places don't count: what the weights alone cost.",
        "- **tears**: vertices in failing stretch, volume and bleed findings of the bend checker on those frames. The artist's own rig is the first column of numbers to beat.",
        "- **feet**: while the artist's foot is on the ground, how far the route's foot is from it.",
        "",
        "| Animal | Rig | Joints | Missing | Skin mean / 95th | Weights only | Tears | Feet |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, rows in summary.items():
        a = rows.get("artist", {})
        lines.append(f"| **{name}** ({kinds.get(name, '')}) | {LABEL['artist']} | 0 | | 0 | 0 | {a.get('tears_pct', '?')}% | 0 |")
        for r in cols:
            x = rows.get(r, {})
            if "error" in x or not x:
                lines.append(f"| | {LABEL.get(r, r)} | no rig | | | | | |")
                continue
            miss = ", ".join(short(m) for m in x["missing"]) or "none"
            lines.append(f"| | {LABEL.get(r, r)} | {x['joints_pct']}% | {miss} | {x['skin_mean_pct']}% / {x['skin_p95_pct']}% | "
                         f"{x['weights_only_mean_pct']}% | {x.get('tears_pct', '?')}% | {x['feet_off_pct']}% |")
    lines += ["", "## Medians over the set", "", "| Rig | Joints | Animals with a missing chain | Skin mean | Skin 95th | Weights only | Tears | Feet |", "|---|---|---|---|---|---|---|---|"]
    meds = {}
    for r in ["artist"] + cols:
        xs = [rows[r] for rows in summary.values() if r in rows and "error" not in rows[r]]
        if not xs:
            continue
        m = {k: med([x.get(k) for x in xs]) for k in ("joints_pct", "skin_mean_pct", "skin_p95_pct", "weights_only_mean_pct", "tears_pct", "feet_off_pct")}
        m["with_missing"] = sum(1 for x in xs if x.get("missing"))
        m["animals"] = len(xs)
        meds[r] = m
        lines.append(f"| {LABEL.get(r, r)} | {m['joints_pct']}% | {m['with_missing']} of {m['animals']} | {m['skin_mean_pct']}% | {m['skin_p95_pct']}% | "
                     f"{m['weights_only_mean_pct']}% | {m['tears_pct']}% | {m['feet_off_pct']}% |")
    if len(routes) >= 2:
        a, b = routes[0], routes[1]
        both = [(n, rows[a], rows[b]) for n, rows in summary.items() if "error" not in rows.get(a, {"error": 1}) and "error" not in rows.get(b, {"error": 1})]
        better = [n for n, x, y in both if x["skin_mean_pct"] < y["skin_mean_pct"] - 0.05]
        worse = [n for n, x, y in both if x["skin_mean_pct"] > y["skin_mean_pct"] + 0.05]
        lines += ["", f"Skin, {LABEL.get(a, a)} against {LABEL.get(b, b)}: closer to the artist on {len(better)} of {len(both)} ({', '.join(better) or 'none'}), "
                  f"farther on {len(worse)} ({', '.join(worse) or 'none'})."]
    return "\n".join(lines) + "\n", meds


def pictures(bench, out, summary, routes, blender, repo):
    """Per animal: the artist's rig and each route's at the frame where the
    first route is farthest from the artist (`<name>.png`), and a loop of
    one clip (`<name>.mp4`); then every sheet in one picture (`all.jpg`)."""
    script = os.path.join(repo, "tools/bench/render_compare.py")
    sheets = []
    for name, rows in summary.items():
        sd = os.path.join(bench, "animals", name, "score")
        rigs = [("artist", os.path.join(sd, "artist.clips.glb"))]
        rigs += [({"new": "today", "old": "old"}.get(r, r), os.path.join(sd, r + ".clips.glb")) for r in routes if "error" not in rows.get(r, {"error": 1})]
        lead = next((r for r in routes if "error" not in rows.get(r, {"error": 1})), None)
        if lead is None:
            continue
        worst = rows[lead]["worst"]
        with open(os.path.join(sd, lead + ".json")) as f:
            clips = [c["clip"] for c in json.load(f)["skin"]["clips"]]
        low = [c.lower() for c in clips]
        pick = next((clips[i] for i, c in enumerate(low) if "walk" in c and "slow" not in c), None) \
            or next((clips[i] for i, c in enumerate(low) if "run" in c or "fly" in c), None) or clips[0]
        png, mp4 = os.path.join(out, name + ".png"), os.path.join(out, name + ".mp4")
        cmd = [blender, "-b", "--factory-startup", "-P", script, "--", "--out", png, "--clip", worst["clip"], "--time", str(worst["time"]),
               "--loop", mp4, "--loop-clip", pick, "--loop-frames", "72"]
        for label, path in rigs:
            cmd += ["--rig", f"{label}={path}"]
        code, text, err = run(cmd)
        if code != 0 or not os.path.exists(png):
            print(f"  {name}: no picture: {(err or text).strip()[-200:]}")
            continue
        rows["picture"] = {"sheet": os.path.basename(png), "frame": f"{worst['clip']} at {worst['time']} s", "loop": os.path.basename(mp4), "loop_clip": pick}
        sheets.append(png)
        print(f"  {name}: {os.path.basename(png)} ({worst['clip']} at {worst['time']} s), {os.path.basename(mp4)} ({pick})")
    if sheets and shutil.which("convert"):
        run(["convert"] + sheets + ["-resize", "1500x", "-append", "-quality", "82", os.path.join(out, "all.jpg")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--routes", default="new,old")
    ap.add_argument("--out", default="")
    ap.add_argument("--creature-bench", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--no-score", action="store_true", help="only rebuild the table from the scores already there")
    ap.add_argument("--pictures", action="store_true", help="also render each animal's sheet and loop (Blender, ffmpeg)")
    ap.add_argument("--blender", default="")
    a = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = a.out or os.path.join(a.bench, "out")
    os.makedirs(out, exist_ok=True)
    routes = [r for r in a.routes.split(",") if r]
    kinds, order = {}, []
    tsv = os.path.join(a.bench, "animals.tsv")
    if os.path.exists(tsv):
        for line in open(tsv):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[0]:
                kinds[p[0]] = p[1]
                order.append(p[0])
    names = a.names or order or sorted(os.listdir(os.path.join(a.bench, "animals")))
    summary_path = os.path.join(out, "summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f).get("animals", {})
    if not a.no_score:
        cb = find("creature-bench", a.creature_bench, "CREATURE_BENCH_BIN", [os.path.join(repo, "target/release/creature-bench")])
        wf = find("weights", a.weights, "WEIGHTS_BIN", [os.path.join(os.path.dirname(repo), "weightforge/rust/target/release/weights")])
        empty = os.path.join(out, "clips_only.ron")
        with open(empty, "w") as f:
            f.write('// No poses: the bend checker then judges a rig on its clips alone.\nPoseSet(\n    name: "clips only",\n    poses: [],\n)\n')
        for n in names:
            d = os.path.join(a.bench, "animals", n)
            if not os.path.exists(os.path.join(d, "truth.glb")):
                print(f"  {n}: no truth.glb")
                continue
            summary[n] = score_animal(cb, wf, d, n, routes, empty)
    summary = {n: summary[n] for n in (order or sorted(summary)) if n in summary}
    if a.pictures:
        blender = find("blender", a.blender, "BLENDER", ["/Applications/Blender.app/Contents/MacOS/Blender"])
        pictures(a.bench, out, summary, routes, blender, repo)
    text, meds = table(summary, routes, kinds)
    with open(os.path.join(out, "table.md"), "w") as f:
        f.write(text)
    with open(summary_path, "w") as f:
        json.dump({"routes": routes, "kinds": kinds, "medians": meds, "animals": summary}, f, indent=1)
    print(text[text.index("## Medians"):])
    return 0


if __name__ == "__main__":
    sys.exit(main())
