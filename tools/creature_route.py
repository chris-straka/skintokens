#!/usr/bin/env python3
"""One command from a generated creature to a game-ready rigged GLB
(stdlib only). The creature side of motionforge's `tools/tripo_rig.py`:
same inputs, same outputs, a different rig.

    python3 tools/creature_route.py MODEL.fbx OUT_DIR [--name NAME]
        [--class monster|hero|npc] [--tris N] [--height M] [--convention +x|...|none]
        [--seed N] [--no-ml] [--no-clips] [--no-predicted] [--hint PREDICTED.glb]
        [--fix-args "..."] [--keep-tries]

Steps, each a tool of its own:

1. `motionforge/tools/tripo_to_glb.py` (Blender): one mesh, facing
   forward, on the ground, cut to the class's triangle budget. (A GLB
   that is already clean can be given with `--clean-mesh`.)
2. `creature-rig rig`: the body's parts read from its volume, a skeleton
   laid along them, weights through the body at three band widths, and
   the body's range of motion as a pose set.
3. A four-legged creature gets its walk, trot and lunge
   (`motionforge creature-profile`, `creature-gait`).
4. `weights fix` (weightforge) chooses per region among the three band
   widths, the model's own weights for this skeleton (`skintokens skin`,
   unless `--no-ml`) and its own repairs; `weights check` is the gate.
5. `creature-rig check`: the skeleton check (every limb has a chain,
   bones inside the body, sides match).
   The model's own rig of the same mesh (`skintokens rig`) is judged the
   same way and kept instead when it is the better of the two
   (`report.json`: `picked`), unless `--no-predicted`.
6. `rfcheck`: the export contract and the class's budget.

OUT_DIR gets `<name>.glb`, `<name>.sheet.png` (before and after the fix,
posed), `<name>.skeleton.png`, `<name>.rom.ron`, `report.json` and the
cleaned mesh. Exit 0: the rig passes the bend gate, the skeleton check
and the contract. Exit 1: it doesn't (it is still written, and the report
says what failed). Exit 2: a step could not run.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

TRIS = {"hero": 14000, "npc": 7500, "monster": 18000}
# band widths of the weight candidates, in body thicknesses at the joint
BLENDS = [1.6, 1.0, 2.5]


def find(name, explicit, env, fallbacks, first=()):
    for p in [explicit, os.environ.get(env)] + list(first) + [shutil.which(name)] + fallbacks:
        if p and os.path.exists(p):
            return p
    raise SystemExit(f"{name} not found: pass --{name}, set {env}, or put it on PATH")


def run(cmd, timeout=None, env=None):
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        code, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        code, out, err = 124, e.stdout or "", f"timed out after {timeout} s"
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
    return {"code": code, "secs": round(time.time() - t, 1), "out": out, "err": err}


def tail(text, n=300):
    return (text or "").strip()[-n:]


def check(wf, glb, rom):
    r = run([wf, "check", glb, "--json", "--poses", rom], timeout=1800)
    try:
        d = json.loads(r["out"])
    except ValueError:
        return {"score": 0.0, "pass": False, "fails": ["check did not run: " + tail(r["err"], 120)]}
    fails = [f"{f.get('code')} {f.get('region')}: {f.get('detail')}" for f in d.get("findings", [])
             if f.get("severity") == "fail"]
    return {"score": d.get("score", 0.0), "pass": bool(d.get("pass")), "fails": fails, "verts": d.get("verts"),
            "bones": d.get("bones"), "poses": d.get("poses"), "not_tested": d.get("skipped_poses", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--name", default="")
    ap.add_argument("--class", dest="klass", default="monster", choices=["hero", "npc", "monster"])
    ap.add_argument("--tris", type=int, default=0)
    ap.add_argument("--height", type=float, default=0.0)
    ap.add_argument("--convention", default="+x")
    ap.add_argument("--clean-mesh", default="", help="skip step 1: this GLB is already clean")
    ap.add_argument("--seed", type=int, default=1, help="for the model's weights (the only step with chance in it)")
    ap.add_argument("--no-ml", action="store_true", help="don't offer the model's weights to the fix")
    ap.add_argument("--no-clips", action="store_true")
    ap.add_argument("--no-predicted", action="store_true", help="don't also try the model's own rig (step 5b)")
    ap.add_argument("--hint", default="", help="a rig of this model from `skintokens rig`: knees and elbows go where its joints are")
    ap.add_argument("--no-fix", action="store_true")
    ap.add_argument("--fix-args", default="")
    ap.add_argument("--fix-timeout", type=int, default=1800)
    ap.add_argument("--keep-tries", action="store_true")
    for tool in ("blender", "creature-rig", "skintokens", "motionforge", "weights", "rfcheck"):
        ap.add_argument(f"--{tool}", default="")
    a = ap.parse_args()

    home = os.path.expanduser("~")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sib = os.path.dirname(repo)
    cr = find("creature-rig", getattr(a, "creature_rig"), "CREATURE_RIG_BIN", [os.path.join(repo, "target/release/creature-rig")])
    mf = find("motionforge", a.motionforge, "MOTIONFORGE_BIN", [os.path.join(sib, "motionforge/rust/target/release/motionforge")])
    wf = find("weights", a.weights, "WEIGHTS_BIN", [os.path.join(sib, "weightforge/rust/target/release/weights")])
    rf = find("rfcheck", a.rfcheck, "RFCHECK_BIN", [os.path.join(sib, "rfcheck/target/release/rfcheck"),
                                                     os.path.join(home, ".cargo/bin/rfcheck")])
    name = a.name or os.path.basename(os.path.dirname(os.path.abspath(a.model))) or "model"
    os.makedirs(a.out, exist_ok=True)
    report = {"model": os.path.abspath(a.model), "name": name, "kind": "creature", "class": a.klass,
              "route": "creature-rig", "steps": {}, "tries": []}

    def save():
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)

    # 1. Clean mesh.
    mesh = os.path.join(a.out, "mesh.glb")
    if a.clean_mesh:
        if os.path.abspath(a.clean_mesh) != os.path.abspath(mesh):
            shutil.copyfile(a.clean_mesh, mesh)
        report["steps"]["clean"] = {"ok": True, "given": os.path.abspath(a.clean_mesh)}
        # called by tripo_rig.py after its own clean step: keep what that step reported
        try:
            with open(os.path.join(a.out, "report.json")) as f:
                report["steps"]["clean"] = json.load(f)["steps"]["clean"]
        except (OSError, ValueError, KeyError):
            pass
    else:
        blender = find("blender", a.blender, "BLENDER", [], first=["/Applications/Blender.app/Contents/MacOS/Blender"])
        script = os.environ.get("TRIPO_TO_GLB") or os.path.join(sib, "motionforge/tools/tripo_to_glb.py")
        r = run([blender, "-b", "--factory-startup", "-P", script, "--", a.model, mesh, "--kind", "auto",
                 "--convention", a.convention, "--tris", str(a.tris or TRIS[a.klass]), "--height", str(a.height)],
                timeout=900)
        line = [ln for ln in r["out"].splitlines() if ln.startswith("TRIPO_TO_GLB ")]
        if r["code"] != 0 or not line:
            report["steps"]["clean"] = {"ok": False, "error": tail(r["err"] or r["out"])}
            save()
            raise SystemExit(f"clean failed: {tail(r['err'] or r['out'])}")
        report["steps"]["clean"] = dict(json.loads(line[-1][len("TRIPO_TO_GLB "):]), ok=True, secs=r["secs"])
    save()

    # 2. Our rig, and the same skeleton weighted at the other band widths.
    t = {"seed": a.seed, "rigged": False}
    report["tries"].append(t)
    base = os.path.join(a.out, "try")
    rig, rom, fixed = base + ".rig.glb", base + ".rom.ron", base + ".glb"
    hint = ["--hint", a.hint] if a.hint else []
    r = run([cr, "rig", mesh, rig, "--rom", rom, "--json", base + ".rig.json", "--sheet", base + ".skeleton.png",
             "--blend", str(BLENDS[0])] + hint, timeout=900)
    t["rig"] = {"code": r["code"], "secs": r["secs"]}
    if r["code"] != 0 or not os.path.exists(rig):
        t["rig"]["error"] = tail(r["err"] or r["out"])
        save()
        print(f"{name}: no rig: {t['rig']['error']}")
        return 2
    with open(base + ".rig.json") as f:
        info = json.load(f)
    t.update(rigged=True, body=info["kind"] + (", upright" if info["upright"] else ""),
             turned_half_a_turn=info.get("turned_half_a_turn", False),
             limbs={l["name"]: len(l["bones"]) for l in info["limbs"]})
    candidates = []
    for b in BLENDS[1:]:
        c = f"{base}.band{b}.glb"
        if run([cr, "rig", mesh, c, "--blend", str(b)] + hint, timeout=900)["code"] == 0:
            candidates.append(c)
    print(f"{name}: {t['body']}, {len(info['bones'])} bones ({', '.join(f'{k} {v}' for k, v in t['limbs'].items())})"
          + ("; turned to face forward" if t["turned_half_a_turn"] else ""))

    # 3. Clips for a four-legged body. creature-profile may drop joints it
    # takes for strays; ours are all on the body, so a profile that dropped
    # any is not used.
    posed = rig
    if info["kind"] == "four-legged" and not a.no_clips:
        prof, clean_glb, moving = base + ".creature.json", base + ".clean.glb", base + ".moving.glb"
        r = run([mf, "creature-profile", "--rig", rig, "--output", prof, "--clean-glb", clean_glb,
                 "--report", base + ".profile.json", "--name", name])
        t["profile"] = {"code": r["code"]}
        kept = False
        if r["code"] == 0:
            try:
                with open(base + ".profile.json") as f:
                    p = json.load(f)
                kept = p.get("joints_in") == p.get("joints_out")
                t["profile"]["joints"] = [p.get("joints_in"), p.get("joints_out")]
            except (OSError, ValueError):
                pass
        if kept:
            r = run([mf, "creature-gait", "--creature", prof, "--glb-in", clean_glb, "--glb-out", moving,
                     "--rom", base + ".gait_rom.ron"])
            t["gait"] = {"code": r["code"], "clips": [ln for ln in r["out"].splitlines() if ln.startswith("clip ")]}
            if r["code"] == 0 and os.path.exists(moving):
                posed = moving
            else:
                t["gait"]["error"] = tail(r["err"] or r["out"])
        else:
            t["profile"]["error"] = tail(r["err"] or r["out"]) or "the profile changed the skeleton"
    t["clips"] = posed != rig

    # 4. Fix and gate.
    raw = check(wf, posed, rom)
    t["raw"] = raw
    if a.no_fix or raw["pass"]:
        shutil.copyfile(posed, fixed)
    else:
        cmd = [wf, "fix", posed, "--out", fixed, "--sheet", base + ".sheet.png", "--poses", rom]
        for c in candidates:
            cmd += ["--candidate", c]
        env = dict(os.environ)
        if not a.no_ml:
            cmd.append("--skintokens")
            if a.skintokens:
                env["SKINTOKENS_BIN"] = a.skintokens
        r = run(cmd + shlex.split(a.fix_args), timeout=a.fix_timeout, env=env)
        t["fix"] = {"code": r["code"], "secs": r["secs"],
                    "picked": [ln.strip() for ln in r["out"].splitlines() if " via " in ln][:24]}
        if not os.path.exists(fixed):
            t["fix"]["error"] = tail(r["err"] or r["out"])
            shutil.copyfile(posed, fixed)
    final = check(wf, fixed, rom)
    t.update(score=final["score"], fails=final["fails"], not_tested=final.get("not_tested", []))
    t["pass"] = final["pass"]

    # 5. The skeleton check.
    r = run([cr, "check", fixed, "--json", base + ".skeleton.json"])
    try:
        with open(base + ".skeleton.json") as f:
            s = json.load(f)
        t["skeleton"] = {"score": s["score"], "pass": s["pass"],
                         "fails": [f"{x['code']} {x['part']}: {x['detail']}" for x in s["findings"] if x["severity"] == "fail"]}
    except (OSError, ValueError, KeyError):
        t["skeleton"] = {"score": 0.0, "pass": False, "fails": ["skeleton check did not run: " + tail(r["err"], 120)]}

    # 5b. The model's own rig of the same mesh, judged the same way (the
    # body's range of motion on its bones, the skeleton check): it is kept
    # instead when it is the better one, so a creature is never rigged
    # worse than by the model alone.
    def skeleton_of(glb, out_json, rom_out=None, sheet=None):
        cmd = [cr, "check", glb, "--json", out_json]
        if rom_out:
            cmd += ["--rom", rom_out]
        if sheet:
            cmd += ["--sheet", sheet]
        r = run(cmd)
        try:
            with open(out_json) as f:
                s = json.load(f)
            return {"score": s["score"], "pass": s["pass"],
                    "fails": [f"{x['code']} {x['part']}: {x['detail']}" for x in s["findings"] if x["severity"] == "fail"]}
        except (OSError, ValueError, KeyError):
            return {"score": 0.0, "pass": False, "fails": ["skeleton check did not run: " + tail(r["err"], 120)]}

    def rank(x):
        """The creature score (bend x skeleton / 100), 15 off for a failing
        skeleton finding (a limb with no bones in it costs more than a few
        torn faces), 5 on for passing both checks. Passing alone does not
        win: a rig can pass both with bones along half a limb and score 44
        beside one that scores 75 with a single failing finding."""
        both = x["pass"] and x["skeleton"]["pass"]
        return x["score"] * x["skeleton"]["score"] / 100.0 - (0 if x["skeleton"]["pass"] else 15) + (5 if both else 0)

    picked = {"glb": fixed, "rom": rom, "sheet": base + ".sheet.png", "skeleton_png": base + ".skeleton.png", "which": "body"}
    sk = None
    if not a.no_predicted:
        for p in [a.skintokens, os.environ.get("SKINTOKENS_BIN"), shutil.which("skintokens"), os.path.join(repo, "target/release/skintokens")]:
            if p and os.path.exists(p):
                sk = p
                break
    if sk:
        pb = base + ".pred"
        p = {"rigged": False}
        t["predicted"] = p
        # our rigged file as the input: it is the same mesh, already turned to face forward
        r = run([sk, "rig", rig, pb + ".rig.glb", "--report", pb + ".skintokens.json", "--class", "quadruped",
                 "--seed", str(a.seed)], timeout=3600)
        ok = r["code"] == 0 and os.path.exists(pb + ".rig.glb")
        if ok:
            r = run([mf, "standardize", "--input", pb + ".rig.glb", "--output", pb + ".std.glb", "--class", "quadruped",
                     "--report", pb + ".standardize.json"])
            ok = r["code"] == 0 and os.path.exists(pb + ".std.glb")
        if not ok:
            p["error"] = tail(r["err"] or r["out"])
        else:
            p["rigged"] = True
            pposed = pb + ".std.glb"
            if info["kind"] == "four-legged" and not a.no_clips:
                r = run([mf, "creature-profile", "--rig", pposed, "--output", pb + ".creature.json", "--clean-glb",
                         pb + ".clean.glb", "--report", pb + ".profile.json", "--name", name])
                if r["code"] == 0:
                    r = run([mf, "creature-gait", "--creature", pb + ".creature.json", "--glb-in", pb + ".clean.glb",
                             "--glb-out", pb + ".moving.glb", "--rom", pb + ".gait_rom.ron"])
                    if r["code"] == 0 and os.path.exists(pb + ".moving.glb"):
                        pposed = pb + ".moving.glb"
            p["clips"] = pposed.endswith(".moving.glb")
            skeleton_of(pposed, pb + ".skeleton.json", rom_out=pb + ".rom.ron")
            praw = check(wf, pposed, pb + ".rom.ron")
            if a.no_fix or praw["pass"]:
                shutil.copyfile(pposed, pb + ".glb")
            else:
                run([wf, "fix", pposed, "--out", pb + ".glb", "--sheet", pb + ".sheet.png", "--poses", pb + ".rom.ron"]
                    + shlex.split(a.fix_args), timeout=a.fix_timeout)
                if not os.path.exists(pb + ".glb"):
                    shutil.copyfile(pposed, pb + ".glb")
            pfinal = check(wf, pb + ".glb", pb + ".rom.ron")
            p.update(raw=praw["score"], score=pfinal["score"], fails=pfinal["fails"])
            p["pass"] = pfinal["pass"]
            p["skeleton"] = skeleton_of(pb + ".glb", pb + ".skeleton.json", sheet=pb + ".skeleton.png")
            if rank(p) > rank(t):
                picked = {"glb": pb + ".glb", "rom": pb + ".rom.ron", "sheet": pb + ".sheet.png",
                          "skeleton_png": pb + ".skeleton.png", "which": "predicted"}
    t["picked"] = picked["which"]
    best = t["predicted"] if picked["which"] == "predicted" else t

    # 6. Contract.
    r = run([rf, "--json", "--class", a.klass, picked["glb"]])
    t["contract"] = r["code"] == 0
    try:
        d = json.loads(r["out"])
        d = d[0] if isinstance(d, list) and d else d
        t["rfcheck"] = [f"{x.get('code')}: {x.get('detail')}" for x in (d.get("diags") or [])
                        if not str(x.get("code", "")).startswith("P_")]
    except ValueError:
        t["rfcheck"] = [tail(r["out"], 120)]

    shutil.copyfile(picked["glb"], os.path.join(a.out, name + ".glb"))
    shutil.copyfile(picked["rom"], os.path.join(a.out, name + ".rom.ron"))
    if os.path.exists(picked["skeleton_png"]):
        shutil.copyfile(picked["skeleton_png"], os.path.join(a.out, name + ".skeleton.png"))
    sheet = os.path.join(a.out, name + ".sheet.png")
    if os.path.exists(picked["sheet"]):
        shutil.copyfile(picked["sheet"], sheet)
    else:
        run([wf, "sheet", picked["glb"], "--out", sheet, "--poses", picked["rom"]])
    report["best_seed"] = a.seed
    report["ok"] = bool(best["pass"] and best["skeleton"]["pass"] and t["contract"])
    save()
    if not a.keep_tries:
        for f in os.listdir(a.out):
            if f.startswith("try.") and f.endswith(".glb"):
                os.remove(os.path.join(a.out, f))
    why = best["fails"][:4] + best["skeleton"]["fails"][:3] + ([] if t["contract"] else t["rfcheck"][:3])
    other = ""
    if t.get("predicted", {}).get("rigged"):
        o = t["predicted"] if picked["which"] == "body" else t
        other = f" [kept the {'body-read' if picked['which'] == 'body' else 'predicted'} rig; the other: bend {o['score']}, skeleton {o['skeleton']['score']}]"
    print(f"{name}: bend {best['score']} ({'passes' if best['pass'] else str(len(best['fails'])) + ' failing'}), "
          f"skeleton {best['skeleton']['score']}, contract {'clean' if t['contract'] else 'broken'}"
          f"{', with clips' if best.get('clips') else ''}{other}: "
          f"{'PASSES' if report['ok'] else 'FAILS: ' + '; '.join(why)} -> {os.path.join(a.out, name + '.glb')}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
