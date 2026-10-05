"""skintokens: SkinTokens/TokenRig auto-rigging as a file-in, file-out CLI.

    skintokens rig    IN.glb OUT.glb [--report R.json] [--class humanoid|quadruped|custom] [--seed N]
    skintokens skin   IN.glb OUT.glb [--report R.json] [--seed N]
    skintokens joints IN.glb OUT.json [--subject NAME]
    skintokens doctor

rig    generates a skeleton and skin weights for an unrigged mesh. Humanoid
       joints are named by structure (Mixamo names) so motionforge's
       standardize adapter maps them onto the HLL skeleton.
skin   keeps IN's skeleton and replaces only its weights (JOINTS_0/WEIGHTS_0
       bytes): a weightforge `fix --candidate`.
joints writes skintokens-joints/1 (same schema as unirig-joints/1) from a
       rigged GLB: rigforge's joint hints.

The report follows genforge's adapter contract: {"ok", "outputs" (paths
relative to the report's folder), "tool", "version", "stage", "<stage>": {..},
"reason"?}. Exit 0 ok, 1 ran and failed (report written), 2 usage or
environment error (no report). Inputs are never modified. One inference runs
at a time per machine (file lock): the model needs ~4 GB and upstream's bpy
helper listens on a fixed port.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skintokens.glb import Glb  # noqa: E402
from skintokens.naming import name_humanoid  # noqa: E402

VERSION = "0.1.0"
UPSTREAM_COMMIT = "273b691"
MODEL = "VAST-AI/SkinTokens@79736cad grpo_1400 + skin_vae_2_10_32768"
JOINTS_FORMAT = "skintokens-joints/1"


class Usage(Exception):
    """Exit 2: nothing ran, no report."""


def home() -> str:
    return os.environ.get("SKINTOKENS_HOME", os.path.expanduser("~/.local/share/skintokens"))


def upstream() -> str:
    return os.path.join(home(), "upstream")


def write_report(path, out_files, stage, ok, details, reason=None):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    rep = {
        "ok": ok,
        "outputs": [os.path.relpath(os.path.abspath(f), d) for f in out_files] if ok else [],
        "tool": "skintokens",
        "version": VERSION,
        "stage": stage,
        stage: details,
    }
    if reason:
        rep["reason"] = reason
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
        f.write("\n")


def run_model(inp, raw_out, seed, extra):
    """Run upstream demo.py once, under the machine-wide lock. Returns (rc, log tail)."""
    up = upstream()
    for need in ("demo.py", "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
                 "experiments/skin_vae_2_10_32768/last.ckpt", "models/Qwen3-0.6B/config.json"):
        if not os.path.exists(os.path.join(up, need)):
            raise Usage(f"SkinTokens not installed ({need} missing under {up}); run setup.sh")
    os.makedirs(home(), exist_ok=True)
    log_path = raw_out + ".log"
    with open(os.path.join(home(), ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(log_path, "w") as log:
            rc = subprocess.call(
                [sys.executable, "demo.py", "--input", os.path.abspath(inp), "--output", os.path.abspath(raw_out),
                 "--seed", str(seed), *extra],
                cwd=up, stdout=log, stderr=subprocess.STDOUT,
            )
    tail = open(log_path, errors="replace").read().strip().splitlines()[-6:]
    if rc == 0 and not os.path.exists(raw_out):
        rc = 1
    return rc, "\n".join(tail)


def check_input(path):
    if not os.path.isfile(path):
        raise Usage(f"input not found: {path}")
    try:
        return Glb.read(path)
    except Exception as e:  # noqa: BLE001
        raise Usage(f"{path}: unreadable GLB ({e})") from e


def weights_summary(g: Glb) -> dict:
    _, J, W = g.skinned_vertices()
    nz = (W > 1e-6).sum(1)
    return {"verts": int(len(W)), "max_influences": int(nz.max()), "max_sum_error": float(np.abs(W.sum(1) - 1).max())}


# ---------------------------------------------------------------- commands


def cmd_rig(a):
    check_input(a.input)
    report = a.report or os.path.join(os.path.dirname(os.path.abspath(a.out)), "result.json")
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="skintokens_") as tmp:
        raw = os.path.join(tmp, "raw.glb")
        rc, tail = run_model(a.input, raw, a.seed, ["--use_transfer"] + (["--use_postprocess"] if a.postprocess else []))
        details = {"class": a.cls, "seed": a.seed, "model": MODEL, "upstream": UPSTREAM_COMMIT}
        if rc != 0:
            write_report(report, [], "rig", False, details, f"model run failed (exit {rc}): {tail}")
            return 1
        g = Glb.read(raw)
        names, parents, heads = g.skeleton()
        details.update(joints_raw=len(names), seconds=round(time.time() - t0, 1))
        if a.cls == "humanoid":
            mapping, missing = name_humanoid(names, parents, heads)
            if missing:
                details["missing"] = missing
                write_report(report, [], "rig", False, details,
                             f"skeleton is not a humanoid TokenRig could name (missing {', '.join(missing)})")
                return 1
            g.rename_joints(mapping)
            details["naming"] = "mixamo (structural)"
        details["weights"] = weights_summary(g)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        g.write(a.out)
    write_report(report, [a.out], "rig", True, details)
    print(f"skintokens rig: ok ({details['joints_raw']} joints, {details['seconds']} s)")
    return 0


def _similarity(src, dst):
    """Umeyama: R, s, t minimizing |s R src + t - dst|."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / max((a ** 2).sum() / len(src), 1e-12))
    return R, s, md - s * ms @ R.T


def map_weights(orig: Glb, st: Glb) -> dict:
    """Copy st's weights onto orig's skeleton and vertices (nearest head, nearest position)."""
    from scipy.spatial import cKDTree

    _, _, oh = orig.skeleton()
    _, _, sh = st.skeleton()
    sP, sJ, sW = st.skinned_vertices()
    oP = orig.skinned_vertices()[0]
    # The re-export may sit in another frame (upstream's transfer re-places the
    # mesh): align bounding boxes, match joints, then refine on the joints.
    s = float(np.ptp(oP, 0).max() / max(np.ptp(sP, 0).max(), 1e-12))
    t = (oP.min(0) + oP.max(0)) / 2 - s * (sP.min(0) + sP.max(0)) / 2
    sh, sP = sh * s + t, sP * s + t
    dj, jmap = cKDTree(oh).query(sh)
    if len(set(jmap.tolist())) != len(jmap):
        raise ValueError("re-skinned joints do not match the input skeleton one to one")
    # Joint positions come back quantized (1/256 of the model's unit box), so
    # they refine the joint match only; vertices use the box alignment.
    R, s2, t2 = _similarity(sh, oh[jmap])
    dj, jmap = cKDTree(oh).query(s2 * sh @ R.T + t2)
    if len(set(jmap.tolist())) != len(jmap):
        raise ValueError("re-skinned joints do not match the input skeleton one to one")
    tree = cKDTree(sP)
    worst = 0.0
    for p in orig.skinned_primitives():
        at = p["attributes"]
        dist, idx = tree.query(orig.accessor(at["POSITION"]))
        worst = max(worst, float(dist.max()))
        J, W = jmap[sJ[idx]], sW[idx].copy()
        W[W < 1e-6] = 0
        J[W == 0] = 0
        W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
        orig.set_accessor(at["JOINTS_0"], J)
        orig.set_accessor(at["WEIGHTS_0"], W)
    return {"joint_match_max_m": float(dj.max()), "vertex_match_max_m": worst, "frame_scale": s}


def cmd_skin(a):
    g = check_input(a.input)
    try:
        g.skeleton()
        g.skinned_vertices()
    except ValueError as e:
        raise Usage(f"{a.input}: skin needs a rigged GLB ({e})") from e
    report = a.report or os.path.join(os.path.dirname(os.path.abspath(a.out)), "result.json")
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="skintokens_") as tmp:
        raw = os.path.join(tmp, "raw.glb")
        rc, tail = run_model(a.input, raw, a.seed, ["--use_skeleton", "--use_transfer"])
        details = {"seed": a.seed, "model": MODEL, "upstream": UPSTREAM_COMMIT}
        if rc != 0:
            write_report(report, [], "skin", False, details, f"model run failed (exit {rc}): {tail}")
            return 1
        try:
            details.update(map_weights(g, Glb.read(raw)))
        except ValueError as e:
            write_report(report, [], "skin", False, details, str(e))
            return 1
    details["seconds"] = round(time.time() - t0, 1)
    details["weights"] = weights_summary(g)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    g.write(a.out)
    write_report(report, [a.out], "skin", True, details)
    print(f"skintokens skin: ok ({details['seconds']} s)")
    return 0


def joints_doc(g: Glb, subject=None, source=None) -> dict:
    names, parents, heads = g.skeleton()
    pos = g.all_positions()
    lo, hi = pos.min(0), pos.max(0)
    center = (lo + hi) / 2
    scale = float((hi - lo).max() / 2) or 1.0
    kids = {}
    for i, p in enumerate(parents):
        if p is not None:
            kids.setdefault(p, []).append(i)
    joints = []
    for i, n in enumerate(names):
        k = kids.get(i, [])
        joints.append({
            "name": n,
            "parent": parents[i],
            "world": [float(v) for v in heads[i]],
            "normalized": [float(v) for v in (heads[i] - center) / scale],
            "tail_world": [float(v) for v in heads[k[0]]] if len(k) == 1 else None,
            "confidence": None,
        })
    return {
        "format": JOINTS_FORMAT,
        "subject": subject,
        "seed": None,
        "source": source,
        "frame": "source mesh frame as stored (glTF, Y-up); positions in source units",
        "normalizer": {"center": [float(v) for v in center], "scale": scale,
                       "formula": "(world - center) / scale over the mesh bbox; longest extent maps to [-1, 1]"},
        "joints": joints,
    }


def cmd_joints(a):
    g = check_input(a.input)
    try:
        doc = joints_doc(g, a.subject, os.path.abspath(a.input))
    except ValueError as e:
        raise Usage(f"{a.input}: {e}") from e
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(f"skintokens joints: {len(doc['joints'])} joints -> {a.out}")
    return 0


def cmd_doctor(_a):
    up = upstream()
    ok = True
    for need in ("demo.py", "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
                 "experiments/skin_vae_2_10_32768/last.ckpt", "models/Qwen3-0.6B/config.json", "compat"):
        p = os.path.join(up, need)
        there = os.path.exists(p)
        ok &= there
        print(f"{'ok ' if there else 'MISSING'} {p}")
    try:
        import torch  # noqa: PLC0415

        dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"ok  torch {torch.__version__} on {dev}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"MISSING torch ({e})")
    base = subprocess.run(["git", "-C", up, "rev-parse", "--short", "HEAD~1"], capture_output=True, text=True).stdout.strip()
    print(f"{'ok ' if base.startswith(UPSTREAM_COMMIT) else 'WARN'} upstream port commit on {base or '?'} (pinned {UPSTREAM_COMMIT})")
    return 0 if ok else 2


def main(argv=None):
    ap = argparse.ArgumentParser(prog="skintokens", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-V", "--version", action="version", version=f"skintokens {VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rig", help="skeleton + skin weights for an unrigged mesh")
    p.add_argument("input")
    p.add_argument("out")
    p.add_argument("--report", default=None)
    p.add_argument("--class", dest="cls", default="humanoid", choices=["humanoid", "quadruped", "custom"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--postprocess", action="store_true", help="upstream voxel skin postprocess")
    p.set_defaults(func=cmd_rig)
    p = sub.add_parser("skin", help="new weights for IN's existing skeleton")
    p.add_argument("input")
    p.add_argument("out")
    p.add_argument("--report", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_skin)
    p = sub.add_parser("joints", help="skintokens-joints/1 hints from a rigged GLB")
    p.add_argument("input")
    p.add_argument("out")
    p.add_argument("--subject", default=None)
    p.set_defaults(func=cmd_joints)
    p = sub.add_parser("doctor", help="check the install")
    p.set_defaults(func=cmd_doctor)
    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except Usage as e:
        print(f"skintokens {a.cmd}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
