#!/usr/bin/env python3
"""Human five-finger hands via the base-body humanoid route (skintokens-7).

Decided 2026-11-18 (`_tools/docs/work-list.md`): TokenRig gives three hand
segments; five hero fingers come from `base_body` via the humanoid route
(`bodygraft`), judged by `motion-1` (`tools/bench/grip_gate.sh --five-finger >= 0.90`)
and `meshweights` (`weights check` hand findings no worse than baseline).

Usage:
    python3 tools/humanoid_hands.py --input IN.glb --out OUT.glb \\
        [--baseline REF.glb] [--base-body BASE.glb] [--motion-clip-rig HERO.glb] \\
        [--report R.json]
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
import tempfile

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS_DIR)
SIBLINGS = os.path.dirname(REPO)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import toolpaths  # noqa: E402

FINGER_CHAINS = ("thumb", "f_index", "f_middle", "f_ring", "f_pinky")
HAND_WORDS = ("hand", "finger", "thumb", "wrist")


def glb_node_names(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"not a binary glTF file: {path}")
    jlen = struct.unpack_from("<I", data, 12)[0]
    gl = json.loads(data[20 : 20 + jlen].decode("utf-8"))
    return [n.get("name", "") for n in gl.get("nodes", [])]


def count_finger_chains(names):
    name_set = set(names)
    counts = {}
    for side in ("L", "R"):
        counts[side] = sum(
            1 for chain in FINGER_CHAINS if f"DEF-{chain}.01.{side}" in name_set
        )
    return counts


def parse_grip_gate_output(text):
    closed = {}
    planted = {}
    for line in text.splitlines():
        m = re.search(
            r"five_finger\s+closed/(\S+)\s*:\s*score=([0-9.]+)", line
        )
        if m:
            closed[m.group(1)] = float(m.group(2))
            continue
        m = re.search(
            r"five_finger\s+(planted_\S+)\s*:\s*score=([0-9.]+)", line
        )
        if m:
            planted[m.group(1)] = float(m.group(2))
    return {"closed": closed, "planted": planted}


def extract_hand_findings(report):
    fails = 0
    total = 0
    area = 0.0
    items = []
    for sec in report.get("headline", []):
        for f in sec.get("findings", []):
            txt = f"{f.get('part', '')} {f.get('detail', '')}".lower()
            if any(w in txt for w in HAND_WORDS):
                total += 1
                if f.get("severity") == "fail":
                    fails += 1
                area += float(f.get("area_cm2", 0.0))
                items.append(
                    {
                        "section": sec.get("name"),
                        "severity": f.get("severity"),
                        "code": f.get("code"),
                        "part": f.get("part"),
                        "worst": f.get("worst"),
                        "area_cm2": f.get("area_cm2", 0.0),
                    }
                )
    return {
        "fails": fails,
        "total": total,
        "area_cm2": round(area, 2),
        "items": items,
    }


def hand_findings_no_worse(baseline, candidate):
    return (
        candidate["fails"] <= baseline["fails"]
        and candidate["total"] <= baseline["total"]
        and candidate["area_cm2"] <= baseline["area_cm2"] + 1e-3
    )


def _sibling_path(*parts):
    return os.path.join(SIBLINGS, *parts)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Attach/verify humanoid-route five-finger hands via motion-1 grip_gate and weights check."
    )
    ap.add_argument("--input", required=True, help="Input humanoid rig GLB")
    ap.add_argument("--out", required=True, help="Output 5-finger rig GLB")
    ap.add_argument("--baseline", default="", help="Baseline skintokens/welded GLB for weights hand comparison")
    ap.add_argument("--base-body", default="", help="Base body GLB carrying 5-finger hands")
    ap.add_argument("--motion-clip-rig", default="", help="Hero GLB with clips for weights check --motion")
    ap.add_argument("--report", default="", help="Optional JSON report path")
    ap.add_argument("--blender", default="", help="Blender binary path")
    ap.add_argument("--weights", default="", help="meshweights `weights` binary path")
    ap.add_argument("--motion-repo", default="", help="motion repo path")
    ap.add_argument("--bodygraft-repo", default="", help="bodygraft repo path")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"missing --input GLB: {args.input}", file=sys.stderr)
        return 2

    motion_repo = args.motion_repo or _sibling_path("motion")
    grip_gate_sh = os.path.join(motion_repo, "tools", "bench", "grip_gate.sh")
    if not os.path.isfile(grip_gate_sh):
        print(f"missing motion grip_gate.sh at {grip_gate_sh}", file=sys.stderr)
        return 2

    weights_bin = toolpaths.find(
        "weights",
        args.weights,
        "WEIGHTS_BIN",
        [_sibling_path("meshweights", "rust", "target", "release", "weights")],
    )

    names = glb_node_names(args.input)
    chains = count_finger_chains(names)
    has_five = chains["L"] == 5 and chains["R"] == 5

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mode = "humanoid_wrapped" if has_five else "bodygraft_hands_graft"

    with tempfile.TemporaryDirectory() as tmp:
        if has_five:
            shutil.copy2(args.input, args.out)
        else:
            bodygraft_repo = args.bodygraft_repo or _sibling_path("bodygraft")
            graft_py = os.path.join(bodygraft_repo, "tools", "graft_part.py")
            if not os.path.isfile(graft_py):
                print(f"missing bodygraft graft_part.py at {graft_py}", file=sys.stderr)
                return 2
            base_body = args.base_body or _sibling_path("bodywrap", "local", "base_body", "base_body.glb")
            if not os.path.isfile(base_body):
                print(f"missing --base-body GLB: {base_body}", file=sys.stderr)
                return 2
            blender = toolpaths.find(
                "blender",
                args.blender,
                "BLENDER",
                ["/Applications/Blender.app/Contents/MacOS/Blender"],
            )
            graft_dir = os.path.join(tmp, "graft")
            r_graft = toolpaths.run(
                [
                    blender,
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "1",
                    "--python",
                    graft_py,
                    "--",
                    "--donor",
                    base_body,
                    "--body",
                    os.path.abspath(args.input),
                    "--part",
                    "hands",
                    "--hands-from",
                    base_body,
                    "--out",
                    graft_dir,
                    "--name",
                    "human",
                    "--size",
                    "700",
                ]
            )
            grafted = os.path.join(graft_dir, "human_grafted.glb")
            graft_json_path = os.path.join(graft_dir, "graft.json")
            refused = None
            if os.path.isfile(graft_json_path):
                with open(graft_json_path, encoding="utf-8") as gf:
                    refused = json.load(gf).get("refused")
            if refused or r_graft["code"] not in (0, 2) or not os.path.isfile(grafted):
                print(
                    f"graft_part.py failed (exit {r_graft['code']}, refused={refused}):\n{r_graft['out']}\n{r_graft['err']}",
                    file=sys.stderr,
                )
                return 1
            shutil.copy2(grafted, args.out)

        out_chains = count_finger_chains(glb_node_names(args.out))
        if out_chains["L"] != 5 or out_chains["R"] != 5:
            print(f"output missing 5-finger chains: {out_chains}", file=sys.stderr)
            return 1

        r_grip = toolpaths.run(
            [
                "bash",
                "-c",
                'cd "$1" && bash tools/bench/grip_gate.sh --five-finger "$2"',
                "grip_gate",
                motion_repo,
                os.path.abspath(args.out),
            ]
        )
        grip_metrics = parse_grip_gate_output(r_grip["out"] + "\n" + r_grip["err"])
        closed = grip_metrics["closed"]
        planted = grip_metrics["planted"]
        grip_ok = (
            r_grip["code"] == 0
            and len(closed) >= 4
            and all(v >= 0.90 for v in closed.values())
            and len(planted) >= 3
            and all(v < 0.30 for v in planted.values())
        )

        baseline_path = args.baseline or args.input
        extra_motion = ["--motion", os.path.abspath(args.motion_clip_rig)] if args.motion_clip_rig else ["--no-clips"]

        def run_weights_check(glb_path):
            r = toolpaths.run([weights_bin, "check", os.path.abspath(glb_path)] + extra_motion + ["--json"])
            if not r["out"].strip():
                raise RuntimeError(f"weights check produced empty JSON on {glb_path}: {r['err']}")
            return extract_hand_findings(json.loads(r["out"]))

        base_hands = run_weights_check(baseline_path)
        out_hands = run_weights_check(args.out)
        weights_ok = hand_findings_no_worse(base_hands, out_hands)

    ok = grip_ok and weights_ok
    report = {
        "ok": ok,
        "mode": mode,
        "input": args.input,
        "out": args.out,
        "finger_chains": out_chains,
        "grip_gate": {
            "exit": r_grip["code"],
            "ok": grip_ok,
            **grip_metrics,
        },
        "weights_hands": {
            "ok": weights_ok,
            "baseline": base_hands,
            "result": out_hands,
        },
    }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    min_closed = min(closed.values()) if closed else 0.0
    print(
        f"HUMANOID_HANDS ok={ok} mode={mode} min_closed={min_closed:.3f} "
        f"hand_findings={base_hands['total']}->{out_hands['total']} "
        f"(fails={base_hands['fails']}->{out_hands['fails']})"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
