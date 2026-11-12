#!/usr/bin/env python3
"""Tabulate tools/transfer_bench.sh results.

    transfer_bench_summary.py OUT_ROOT TAG [TAG...] [--json out.json]

Per name, seed and tag: raw and fixed score, the failing findings (code,
region, vertices), the bad vertices in the hip and arm poses, and every
D_BLEED finding at any severity.
"""
import json
import os
import sys

POSES = ["hip_forward.L", "hip_forward.R", "arm_up.L", "arm_up.R", "arm_forward.L", "arm_forward.R"]


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def digest(d):
    if d is None:
        return None
    pose_bad = {p["pose"]: p["bad"] for p in d.get("worst_poses", [])}
    return {
        "score": d.get("score"),
        "fails": [(f["code"], f["region"], f.get("verts")) for f in d.get("findings", []) if f.get("severity") == "fail"],
        "bleed": [(f["severity"], f["region"], f.get("verts"), f.get("detail")) for f in d.get("findings", []) if f["code"] == "D_BLEED"],
        "poses": {p: pose_bad.get(p, 0) for p in POSES},
        "bad_all_poses": sum(pose_bad.values()),
    }


def main():
    args = sys.argv[1:]
    out_json = None
    if "--json" in args:
        i = args.index("--json")
        out_json = args[i + 1]
        del args[i:i + 2]
    root, tags = args[0], args[1:]
    names = sorted({n for t in tags if os.path.isdir(os.path.join(root, t)) for n in os.listdir(os.path.join(root, t))})
    res = {}
    for n in names:
        for s in (1, 2, 3):
            for t in tags:
                b = os.path.join(root, t, n, f"s{s}")
                r = {"raw": digest(load(b + ".raw.json")), "fixed": digest(load(b + ".json"))}
                rep = load(b + ".skintokens.json")
                if rep:
                    r["transfer"] = rep.get("rig", {}).get("transfer")
                res.setdefault(n, {}).setdefault(str(s), {})[t] = r
    for n in names:
        print(f"## {n}")
        for s in ("1", "2", "3"):
            for t in tags:
                r = res[n][s][t]
                if not r["fixed"]:
                    print(f"  s{s} {t:9s} (missing)")
                    continue
                raw, fx = r["raw"], r["fixed"]
                poses = " ".join(f"{p.replace('hip_forward', 'hipF').replace('arm_forward', 'armF')}={raw['poses'][p]}/{fx['poses'][p]}" for p in POSES)
                print(f"  s{s} {t:9s} raw {raw['score']:5} ({len(raw['fails'])}f, bad {raw['bad_all_poses']:4}) "
                      f"-> fixed {fx['score']:5} ({len(fx['fails'])}f, bad {fx['bad_all_poses']:4})  poses raw/fixed: {poses}")
                for c, reg, v in fx["fails"]:
                    print(f"        fixed fail {c} {reg} {v}")
                for sev, reg, v, det in raw["bleed"]:
                    print(f"        raw bleed {sev} {reg} {v}: {det}")
                for sev, reg, v, det in fx["bleed"]:
                    print(f"        fixed bleed {sev} {reg} {v}: {det}")
                tr = r.get("transfer")
                if tr and tr.get("method") != "straight":
                    print(f"        transfer {tr}")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
