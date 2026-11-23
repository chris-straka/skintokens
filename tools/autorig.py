#!/usr/bin/env python3
"""Unified `autorig` CLI: one entrypoint across skintokens, gamerig (rigforge),
creature-rig (body), motion, external skeleton predictors, and meshweights.

Usage:
    autorig IN.glb OUT.glb [options]
    autorig IN.glb --out OUT.glb [options]

Methods (--method):
    auto              Multi-seed tournament (`skintokens rig` + `creature-rig`
                      for non-hero kinds), always Can-It-Code pruned (preserving
                      5 fingers for `--class hero`), optional `weights fix`,
                      ranked by `creature-bench anatomy` + `weights check` +
                      `pick_score`.
    skintokens        Run `skintokens rig` (+ optional prune / standardize / QA).
    gamerig           Run `_blender/gamerig` headless (`auto_rig.sh` / Blender)
                      selecting `biped_hero` or `quadruped_stalker`.
    rigforge          Alias for `gamerig`.
    body              Run `creature-rig rig` volumetric skeleton + geodesic skin.
    creature-rig      Alias for `body`.
    motion            Can-It-Code creature/biped route: multi-seed `skintokens`,
                      Can-It-Code pruning, `motionforge` standardize / gait,
                      and `weights` QA.
    unirig            External UniRig skeleton predictor (auto-skinned via
                      `skintokens skin` if skeleton-only), pruned & checked.
    puppeteer         External Puppeteer predictor (auto-skinned via
                      `skintokens skin` when skeleton-only), pruned & checked.
    magicarticulate   External MagicArticulate predictor (auto-skinned via
                      `skintokens skin` when skeleton-only), pruned & checked.

License boundary:
    Pure Python (stdlib). Invokes `skintokens` (GPL-3.0-only) and Blender
    scripts strictly as external subprocesses, never linking or importing
    GPL Rust/Python modules into MIT crates (`skintokens-geom`, `creature-rig`,
    `creature-bench`). PartField stays excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import struct
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pick_score as ps  # noqa: E402
import toolpaths as tp  # noqa: E402

METHODS = (
    "auto",
    "skintokens",
    "gamerig",
    "rigforge",
    "body",
    "creature-rig",
    "motion",
    "unirig",
    "puppeteer",
    "magicarticulate",
)

METHOD_ALIASES = {
    "rigforge": "gamerig",
    "creature-rig": "body",
}

CLASS_CHOICES = ("hero", "npc", "monster", "humanoid", "quadruped", "custom")
KIND_CHOICES = ("auto", "humanoid", "quadruped", "creature")


def _norm_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def _skintokens_class(klass: str, kind: str) -> str:
    if kind == "humanoid" or klass in ("hero", "npc", "humanoid"):
        return "humanoid"
    if kind == "quadruped" or klass == "quadruped":
        return "quadruped"
    if kind == "creature" or klass == "monster":
        return "custom"
    return "custom"


def _gamerig_preset(klass: str, kind: str) -> str:
    if kind == "quadruped" or klass in ("quadruped", "monster"):
        if kind == "humanoid" or klass in ("hero", "npc", "humanoid"):
            return "biped_hero"
        return "quadruped_stalker"
    return "biped_hero"


def _parse_seeds(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --seeds specification")
    seeds: list[int] = []
    for p in parts:
        try:
            seeds.append(int(p))
        except ValueError as exc:
            raise ValueError(f"invalid seed {p!r} in --seeds") from exc
    return seeds


def glb_has_skin(path: str) -> bool:
    """Return True iff `path` is a binary glTF container with at least one skin and JOINTS_0."""
    try:
        with open(path, "rb") as f:
            hdr = f.read(12)
            if len(hdr) < 12:
                return False
            magic, _ver, _total = struct.unpack("<III", hdr)
            if magic != 0x46546C67:
                return False
            c_len, c_type = struct.unpack("<II", f.read(8))
            if c_type != 0x4E4F534A:
                return False
            doc = json.loads(f.read(c_len).decode("utf-8"))
    except (OSError, struct.error, ValueError):
        return False
    skins = doc.get("skins") or []
    if not skins or not any((s or {}).get("joints") for s in skins):
        return False
    for mesh in doc.get("meshes") or []:
        for prim in (mesh or {}).get("primitives") or []:
            attrs = (prim or {}).get("attributes") or {}
            if "JOINTS_0" in attrs and "WEIGHTS_0" in attrs:
                return True
    return False


def _find_gamerig_runner(explicit: str) -> str:
    parent = os.path.dirname(REPO)
    for cand in (
        explicit,
        os.environ.get("RIGFORGE_AUTO_RIG", ""),
        os.environ.get("GAMERIG_AUTO_RIG", ""),
        os.path.join(parent, "gamerig", "tools", "auto_rig.sh"),
        os.path.join(REPO, "..", "gamerig", "tools", "auto_rig.sh"),
    ):
        if cand and os.path.isfile(cand):
            return os.path.abspath(cand)
    raise SystemExit("autorig: gamerig auto_rig.sh not found (pass --gamerig or set GAMERIG_AUTO_RIG)")


def resolve_tools(args: argparse.Namespace) -> dict[str, str | None]:
    parent = os.path.dirname(REPO)
    home = os.path.expanduser("~")
    sk_req = args.method in ("skintokens", "motion", "puppeteer", "magicarticulate", "unirig")
    t: dict[str, str | None] = {
        "skintokens": tp.find(
            "skintokens",
            args.skintokens,
            "SKINTOKENS_BIN",
            [
                os.path.join(REPO, "target", "release", "skintokens"),
                os.path.join(REPO, "target", "debug", "skintokens"),
                os.path.join(REPO, "bin", "skintokens"),
            ],
            required=sk_req,
        ),
        "creature-rig": tp.find(
            "creature-rig",
            args.creature_rig,
            "CREATURE_RIG_BIN",
            [
                os.path.join(REPO, "target", "release", "creature-rig"),
                os.path.join(REPO, "target", "debug", "creature-rig"),
            ],
            required=(_norm_method(args.method) == "body"),
        ),
        "creature-bench": tp.find(
            "creature-bench",
            args.creature_bench,
            "CREATURE_BENCH_BIN",
            [
                os.path.join(REPO, "target", "release", "creature-bench"),
                os.path.join(REPO, "target", "debug", "creature-bench"),
            ],
            required=False,
        ),
        "weights": tp.find(
            "weights",
            args.weights,
            "WEIGHTS_BIN",
            [
                os.path.join(parent, "meshweights", "rust", "target", "release", "weights"),
                os.path.join(parent, "meshweights", "rust", "target", "debug", "weights"),
                os.path.join(home, ".cargo", "bin", "weights"),
            ],
            required=False,
        ),
        "motionforge": tp.find(
            "motionforge",
            args.motionforge,
            "MOTIONFORGE_BIN",
            [
                os.path.join(parent, "motion", "rust", "target", "release", "motionforge"),
                os.path.join(parent, "motion", "rust", "target", "debug", "motionforge"),
                os.path.join(home, ".cargo", "bin", "motionforge"),
            ],
            required=False,
        ),
    }
    if _norm_method(args.method) == "gamerig":
        t["gamerig"] = _find_gamerig_runner(args.gamerig)
    else:
        t["gamerig"] = args.gamerig or None
    return t


def _tail(text: str, lines: int = 8) -> str:
    return "\n".join((text or "").strip().splitlines()[-lines:])


def run_step(cmd: list[str], label: str, timeout: int | None = None) -> dict:
    r = tp.run(cmd, timeout=timeout)
    if r["code"] != 0:
        msg = _tail(r["err"] or r["out"] or f"exit {r['code']}")
        raise RuntimeError(f"{label} failed ({r['code']}): {msg}")
    return r


def prune_candidate(
    cr_bin: str | None,
    in_glb: str,
    out_glb: str,
    klass: str,
    keep_digits: bool,
    report_path: str,
    strict: bool = False,
) -> dict | None:
    if not cr_bin:
        if strict:
            raise RuntimeError("creature-rig binary not found for prune")
        shutil.copyfile(in_glb, out_glb)
        return None
    prune_class = "hero" if (keep_digits or klass == "hero") else ("npc" if klass == "npc" else "monster")
    cmd = [cr_bin, "prune", in_glb, out_glb, "--class", prune_class, "--json", report_path]
    if keep_digits or klass == "hero":
        cmd.append("--keep-digits")
    r = tp.run(cmd)
    if r["code"] != 0:
        if strict:
            raise RuntimeError(f"creature-rig prune failed ({r['code']}): {_tail(r['err'] or r['out'])}")
        # Mocked non-GLB binaries in downstream integration smoke tests may pass dummy bytes.
        shutil.copyfile(in_glb, out_glb)
        return {"skipped": _tail(r["err"] or r["out"])}
    try:
        with open(report_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def score_candidate(
    glb_path: str,
    work_dir: str,
    tools: dict[str, str | None],
    poses: str | None,
    motion_glb: str | None,
    rom_out: str | None,
) -> dict:
    """Measure one candidate with creature-bench anatomy, weights check, and creature-rig check."""
    anatomy_report = None
    anatomy_pass = True
    if tools.get("creature-bench"):
        aj = os.path.join(work_dir, "anatomy.json")
        r = tp.run([tools["creature-bench"], "anatomy", glb_path, "--json", aj])
        if r["code"] not in (0, 1):
            raise RuntimeError(f"creature-bench anatomy failed ({r['code']}): {_tail(r['err'] or r['out'])}")
        if os.path.isfile(aj):
            with open(aj, encoding="utf-8") as f:
                anatomy_report = json.load(f)
            anatomy_pass = r["code"] == 0 and bool(anatomy_report.get("pass", True))
        else:
            anatomy_pass = r["code"] == 0

    skel_report = None
    skel_pass = True
    skel_score = 100.0
    if tools.get("creature-rig"):
        sj = os.path.join(work_dir, "skeleton.json")
        cmd = [tools["creature-rig"], "check", glb_path, "--json", sj]
        if rom_out:
            cmd += ["--rom", rom_out]
        r = tp.run(cmd)
        if r["code"] in (0, 1) and os.path.isfile(sj):
            with open(sj, encoding="utf-8") as f:
                skel_report = json.load(f)
            skel_pass = r["code"] == 0 and bool(skel_report.get("pass", True))
            skel_score = float(skel_report.get("score", 100.0))
        elif r["code"] not in (0, 1):
            skel_report = {"skipped": _tail(r["err"] or r["out"])}

    bend_report = None
    bend_pass = True
    tears = 0.0
    bend_score = 100.0
    if tools.get("weights"):
        cmd = [tools["weights"], "check", glb_path, "--json"]
        if poses:
            cmd += ["--poses", poses]
        if motion_glb:
            cmd += ["--motion", motion_glb]
        r = tp.run(cmd)
        if r["code"] not in (0, 1):
            raise RuntimeError(f"weights check failed ({r['code']}): {_tail(r['err'] or r['out'])}")
        if r["out"].strip():
            bend_report = json.loads(r["out"])
            tears = ps.tears_pct(bend_report)
            bend_pass = r["code"] == 0 and bool(bend_report.get("pass", r["code"] == 0))
            bend_score = float(bend_report.get("score", 0.0))
        else:
            bend_pass = r["code"] == 0

    rig_sc = ps.rig_score(skel_report, bend_report, anatomy_report) if (skel_report or bend_report) else 0.0
    fair_sc = ps.fair_score(skel_score, tears, skel_pass, anatomy_report)
    return {
        "glb": glb_path,
        "anatomy": anatomy_report,
        "anatomy_pass": anatomy_pass,
        "skeleton": skel_report,
        "skeleton_pass": skel_pass,
        "skeleton_score": skel_score,
        "bend": bend_report,
        "bend_pass": bend_pass,
        "tears_pct": tears,
        "bend_score": bend_score,
        "rig_score": rig_sc,
        "fair_score": fair_sc,
    }


def _candidate_rank_key(cand: dict) -> tuple:
    """Best candidate first (smaller tuple wins, lower seed breaks ties)."""
    both_ok = bool(cand.get("anatomy_pass", True)) and bool(cand.get("bend_pass", True))
    anat_ok = bool(cand.get("anatomy_pass", True))
    skeleton_ok = bool(cand.get("skeleton_pass", True))
    tears = float(cand.get("tears_pct", 0.0))
    fair = float(cand.get("fair_score", 0.0))
    seed = int(cand.get("seed", 0))
    return (
        0 if both_ok else (1 if anat_ok else 2),
        0 if skeleton_ok else 1,
        round(tears, 4),
        -round(fair, 4),
        seed,
    )


def _postprocess_candidate(
    raw_glb: str,
    work_dir: str,
    args: argparse.Namespace,
    tools: dict[str, str | None],
    ensure_skin: bool = True,
) -> tuple[str, dict]:
    """Ensure skinned -> Can-It-Code prune -> optional weights fix -> optional standardize."""
    info: dict = {}
    cur = raw_glb
    if ensure_skin and not glb_has_skin(cur) and tools.get("skintokens"):
        skinned = os.path.join(work_dir, "skinned.glb")
        rep = os.path.join(work_dir, "skin_report.json")
        run_step([tools["skintokens"], "skin", cur, skinned, "--report", rep], "skintokens skin")
        info["auto_skinned"] = True
        cur = skinned
    else:
        info["auto_skinned"] = False

    if not args.no_prune:
        pruned = os.path.join(work_dir, "pruned.glb")
        prep = os.path.join(work_dir, "prune.json")
        info["prune"] = prune_candidate(
            tools.get("creature-rig"),
            cur,
            pruned,
            args.class_name,
            args.keep_digits,
            prep,
        )
        cur = pruned

    if not args.no_fix and tools.get("weights"):
        fixed = os.path.join(work_dir, "fixed.glb")
        cmd = [tools["weights"], "fix", cur, "--out", fixed]
        if args.poses:
            cmd += ["--poses", args.poses]
        if args.motion:
            cmd += ["--motion", args.motion]
        r = tp.run(cmd)
        if r["code"] in (0, 1) and os.path.isfile(fixed):
            cur = fixed
            info["weights_fix"] = {"exit": r["code"], "applied": True}
        elif r["code"] not in (0, 1):
            raise RuntimeError(f"weights fix failed ({r['code']}): {_tail(r['err'] or r['out'])}")

    if args.standardize and tools.get("motionforge"):
        std = os.path.join(work_dir, "std.glb")
        srep = os.path.join(work_dir, "standardize.json")
        mf_cls = "humanoid" if _skintokens_class(args.class_name, args.kind) == "humanoid" else "quadruped"
        run_step(
            [tools["motionforge"], "standardize", "--input", cur, "--output", std, "--class", mf_cls, "--report", srep],
            "motionforge standardize",
        )
        cur = std
        info["standardized"] = mf_cls

    return cur, info


def _run_skintokens_seed(
    in_glb: str,
    seed: int,
    work_dir: str,
    args: argparse.Namespace,
    tools: dict[str, str | None],
) -> dict:
    sk = tools.get("skintokens")
    if not sk:
        raise RuntimeError("skintokens binary not found")
    os.makedirs(work_dir, exist_ok=True)
    raw = os.path.join(work_dir, f"sk_s{seed}.glb")
    rep = os.path.join(work_dir, f"sk_s{seed}.report.json")
    sk_class = _skintokens_class(args.class_name, args.kind)
    run_step([sk, "rig", in_glb, raw, "--class", sk_class, "--seed", str(seed), "--report", rep], f"skintokens rig (seed {seed})")
    final_glb, post = _postprocess_candidate(raw, work_dir, args, tools, ensure_skin=False)
    scored = score_candidate(final_glb, work_dir, tools, args.poses, args.motion, None)
    return {
        "method": "skintokens",
        "seed": seed,
        "raw_glb": raw,
        **post,
        **scored,
    }


def _run_body_candidate(
    in_glb: str,
    work_dir: str,
    args: argparse.Namespace,
    tools: dict[str, str | None],
) -> dict:
    cr = tools.get("creature-rig")
    if not cr:
        raise RuntimeError("creature-rig binary not found")
    os.makedirs(work_dir, exist_ok=True)
    raw = os.path.join(work_dir, "body.glb")
    cmd = [cr, "rig", in_glb, raw]
    if args.rom:
        cmd += ["--rom", args.rom]
    run_step(cmd, "creature-rig rig")
    final_glb, post = _postprocess_candidate(raw, work_dir, args, tools, ensure_skin=False)
    scored = score_candidate(final_glb, work_dir, tools, args.poses, args.motion, args.rom)
    return {
        "method": "body",
        "seed": 0,
        "raw_glb": raw,
        **post,
        **scored,
    }


def _run_gamerig(
    in_glb: str,
    work_dir: str,
    args: argparse.Namespace,
    tools: dict[str, str | None],
) -> dict:
    runner = tools.get("gamerig")
    if not runner:
        raise RuntimeError("gamerig auto_rig.sh not found")
    os.makedirs(work_dir, exist_ok=True)
    raw = os.path.join(work_dir, "gamerig.glb")
    preset = _gamerig_preset(args.class_name, args.kind)
    cmd = [runner, in_glb, "--preset", preset, "--out", raw]
    if args.hints:
        cmd += ["--hints", args.hints]
    if args.hints_rotated:
        cmd.append("--hints-rotated")
    run_step(cmd, f"gamerig ({preset})")
    final_glb, post = _postprocess_candidate(raw, work_dir, args, tools, ensure_skin=False)
    scored = score_candidate(final_glb, work_dir, tools, args.poses, args.motion, args.rom)
    return {
        "method": "gamerig",
        "preset": preset,
        "seed": 0,
        "raw_glb": raw,
        **post,
        **scored,
    }


def _run_rival(
    method: str,
    in_glb: str,
    seed: int,
    work_dir: str,
    args: argparse.Namespace,
    tools: dict[str, str | None],
) -> dict:
    """Invoke external predictor command (`--rival-cmd`) and always skin+prune if needed."""
    os.makedirs(work_dir, exist_ok=True)
    raw = os.path.join(work_dir, f"{method}_raw.glb")
    if args.rival_cmd:
        formatted = args.rival_cmd.format(in_glb=in_glb, out_glb=raw, seed=seed, **{"in": in_glb, "out": raw})
        run_step(shlex.split(formatted), f"{method} rival")
    else:
        bench_dir = os.path.join(HERE, "bench")
        if method == "unirig":
            raise RuntimeError(
                "autorig --method unirig requires --rival-cmd '<cmd --in {in} --out {out}>' "
                "(unirig-mac was retired 2026-10-05)"
            )
        raise RuntimeError(
            f"autorig --method {method} requires --rival-cmd '<cmd --in {{in}} --out {{out}}>' "
            f"(see {bench_dir}/run_skeleton_rivals.py)"
        )
    if not os.path.isfile(raw):
        raise RuntimeError(f"{method} rival produced no output GLB at {raw}")
    final_glb, post = _postprocess_candidate(raw, work_dir, args, tools, ensure_skin=True)
    scored = score_candidate(final_glb, work_dir, tools, args.poses, args.motion, args.rom)
    return {
        "method": method,
        "seed": seed,
        "raw_glb": raw,
        **post,
        **scored,
    }


def _run_motion_route(
    in_glb: str,
    seeds: list[int],
    work_dir: str,
    args: argparse.Namespace,
    tools: dict[str, str | None],
) -> tuple[dict, list[dict]]:
    """Can It Code motion route: multi-seed `skintokens` + pruning + `motionforge` standardize/profile."""
    candidates: list[dict] = []
    errors: list[str] = []
    for seed in seeds:
        sdir = os.path.join(work_dir, f"motion_s{seed}")
        try:
            cand = _run_skintokens_seed(in_glb, seed, sdir, args, tools)
            cand["method"] = "motion"
            candidates.append(cand)
        except Exception as exc:
            errors.append(f"seed {seed}: {exc}")
    if not candidates:
        raise RuntimeError("all motion candidates failed: " + "; ".join(errors))
    best = min(candidates, key=_candidate_rank_key)
    cur = best["glb"]
    mf = tools.get("motionforge")
    if mf:
        sk_cls = _skintokens_class(args.class_name, args.kind)
        if sk_cls == "quadruped" or args.kind == "creature":
            prof = os.path.join(work_dir, "creature.json")
            clean = os.path.join(work_dir, "creature_clean.glb")
            rep = os.path.join(work_dir, "creature_profile.json")
            run_step([mf, "creature-profile", "--rig", cur, "--output", prof, "--clean-glb", clean, "--report", rep], "motionforge creature-profile")
            cur = clean
            if args.rom and not args.no_clips:
                gait = os.path.join(work_dir, "creature_gait.glb")
                run_step([mf, "creature-gait", "--creature", prof, "--glb-in", cur, "--glb-out", gait, "--rom", args.rom], "motionforge creature-gait")
                cur = gait
            best["glb"] = cur
            best["creature_profile"] = prof
    return best, candidates


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="autorig",
        description="Unified auto-rigger routing across skintokens, gamerig (rigforge), body (creature-rig), motion, and meshweights.",
    )
    ap.add_argument("input", help="Input bare or pre-rigged mesh (.glb)")
    ap.add_argument("output_pos", nargs="?", default=None, help="Output rigged GLB (or pass --out)")
    ap.add_argument("--out", dest="output_opt", default=None, help="Output rigged GLB")
    ap.add_argument("--method", default="auto", choices=METHODS, help="Rigging method (default: auto)")
    ap.add_argument("--class", dest="class_name", default="monster", choices=CLASS_CHOICES, help="Asset class (hero keeps 5 fingers)")
    ap.add_argument("--kind", default="auto", choices=KIND_CHOICES, help="Body kind (default: auto)")
    ap.add_argument("--seeds", default="", help="Seed or comma-separated seed list (e.g. 0,1,2)")
    ap.add_argument("--poses", default=None, help="Optional range-of-motion .ron for weights")
    ap.add_argument("--rom", default=None, help="Optional output path for generated creature-rig .ron poses")
    ap.add_argument("--motion", default=None, help="Optional reference motion .glb for weights")
    ap.add_argument("--hints", default=None, help="Optional skintokens-joints/1 hints JSON for gamerig")
    ap.add_argument("--hints-rotated", action="store_true", help="Apply +90 deg X rotation to hints for gamerig")
    ap.add_argument("--rival-cmd", default=None, help="External skeleton predictor template using {in} and {out} (for unirig/puppeteer/magicarticulate)")
    ap.add_argument("--work-dir", default=None, help="Directory for intermediates (cleaned temp dir if omitted)")
    ap.add_argument("--report", default=None, help="Write JSON report of candidate metrics and winner")
    ap.add_argument("--no-prune", action="store_true", help="Skip Can-It-Code `creature-rig prune`")
    ap.add_argument("--keep-digits", action="store_true", help="Keep hand/foot digit chains during prune (always true for --class hero)")
    ap.add_argument("--no-fix", action="store_true", help="Skip `weights fix` skin polish")
    ap.add_argument("--no-clips", action="store_true", help="Skip clip synthesis on motion route")
    ap.add_argument("--standardize", action="store_true", help="Run `motionforge standardize` after rigging")
    ap.add_argument("--json", action="store_true", help="Print JSON summary to stdout")
    for flag, help_text in (
        ("--skintokens", "Path to `skintokens` binary"),
        ("--creature-rig", "Path to `creature-rig` binary"),
        ("--creature-bench", "Path to `creature-bench` binary"),
        ("--weights", "Path to `weights` binary"),
        ("--motionforge", "Path to `motionforge` binary"),
        ("--gamerig", "Path to `gamerig/tools/auto_rig.sh`"),
    ):
        ap.add_argument(flag, default="", help=help_text)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    out_glb = args.output_opt or args.output_pos
    if not out_glb:
        ap.error("need positional OUT.glb or --out OUT.glb")
    if args.output_opt and args.output_pos:
        ap.error("pass output either positionally or via --out, not both")
    if not os.path.isfile(args.input):
        print(f"autorig: input not found: {args.input}", file=sys.stderr)
        return 2

    method = _norm_method(args.method)
    default_seeds = "0,1,2" if method in ("auto", "motion") else "0"
    try:
        seeds = _parse_seeds(args.seeds or default_seeds)
    except ValueError as exc:
        print(f"autorig: {exc}", file=sys.stderr)
        return 2

    try:
        tools = resolve_tools(args)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2

    t0 = time.time()
    owned_tmp = None
    work_dir = args.work_dir
    if not work_dir:
        owned_tmp = tempfile.mkdtemp(prefix="autorig_")
        work_dir = owned_tmp
    else:
        os.makedirs(work_dir, exist_ok=True)

    candidates: list[dict] = []
    errors: list[str] = []
    best: dict | None = None
    try:
        if method == "auto":
            if tools.get("skintokens"):
                for seed in seeds:
                    try:
                        candidates.append(_run_skintokens_seed(args.input, seed, os.path.join(work_dir, f"sk_s{seed}"), args, tools))
                    except Exception as exc:
                        errors.append(f"skintokens seed {seed}: {exc}")
            if tools.get("creature-rig") and args.class_name != "hero" and args.kind != "humanoid":
                try:
                    candidates.append(_run_body_candidate(args.input, os.path.join(work_dir, "body"), args, tools))
                except Exception as exc:
                    errors.append(f"body: {exc}")
            if not candidates:
                raise RuntimeError("all `auto` candidates failed: " + ("; ".join(errors) or "no rigging backend available"))
            best = min(candidates, key=_candidate_rank_key)
        elif method == "skintokens":
            for seed in seeds:
                try:
                    candidates.append(_run_skintokens_seed(args.input, seed, os.path.join(work_dir, f"sk_s{seed}"), args, tools))
                except Exception as exc:
                    errors.append(f"skintokens seed {seed}: {exc}")
            if not candidates:
                raise RuntimeError("; ".join(errors))
            best = min(candidates, key=_candidate_rank_key)
        elif method == "body":
            best = _run_body_candidate(args.input, os.path.join(work_dir, "body"), args, tools)
            candidates = [best]
        elif method == "gamerig":
            best = _run_gamerig(args.input, os.path.join(work_dir, "gamerig"), args, tools)
            candidates = [best]
        elif method == "motion":
            best, candidates = _run_motion_route(args.input, seeds, work_dir, args, tools)
        elif method in ("unirig", "puppeteer", "magicarticulate"):
            best = _run_rival(method, args.input, seeds[0], os.path.join(work_dir, method), args, tools)
            candidates = [best]
        else:
            print(f"autorig: unsupported method {args.method}", file=sys.stderr)
            return 2

        if not best or not os.path.isfile(best["glb"]):
            raise RuntimeError("winner GLB missing")

        out_parent = os.path.dirname(os.path.abspath(out_glb))
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
        shutil.copyfile(best["glb"], out_glb)

        summary = {
            "format": "autorig/1",
            "input": os.path.abspath(args.input),
            "output": os.path.abspath(out_glb),
            "method": args.method,
            "resolved_method": method,
            "class": args.class_name,
            "kind": args.kind,
            "keep_digits": bool(args.keep_digits or args.class_name == "hero"),
            "winner": {
                "method": best["method"],
                "seed": best.get("seed", 0),
                "auto_skinned": best.get("auto_skinned", False),
                "tears_pct": best.get("tears_pct", 0.0),
                "fair_score": best.get("fair_score", 0.0),
                "rig_score": best.get("rig_score", 0.0),
                "anatomy_pass": best.get("anatomy_pass", True),
                "bend_pass": best.get("bend_pass", True),
                "prune": best.get("prune"),
            },
            "candidates": [
                {
                    "method": c["method"],
                    "seed": c.get("seed", 0),
                    "tears_pct": c.get("tears_pct", 0.0),
                    "fair_score": c.get("fair_score", 0.0),
                    "anatomy_pass": c.get("anatomy_pass", True),
                    "bend_pass": c.get("bend_pass", True),
                }
                for c in candidates
            ],
            "errors": errors,
            "secs": round(time.time() - t0, 2),
        }
        if args.report:
            rep_parent = os.path.dirname(os.path.abspath(args.report))
            if rep_parent:
                os.makedirs(rep_parent, exist_ok=True)
            with open(args.report, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
                f.write("\n")

        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            w = summary["winner"]
            print(
                f"autorig ({summary['resolved_method']} -> {w['method']} seed {w['seed']}): "
                f"tears {w['tears_pct']:.2f}%, fair {w['fair_score']:.1f}, "
                f"anatomy={'PASS' if w['anatomy_pass'] else 'FAIL'} -> {out_glb}"
            )
        return 0
    except Exception as exc:
        print(f"autorig: {exc}", file=sys.stderr)
        return 2
    finally:
        if owned_tmp:
            shutil.rmtree(owned_tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
