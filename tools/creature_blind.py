#!/usr/bin/env python3
"""Blind same-frame walk/pose A/B benchmark at the game camera (`gk blind`).

Stages unmarked synchronized walk frames at HLH's follow camera (75° FOV)
and delegates pair shuffling, LEFT/RIGHT coin-flip, sheet compositing,
can't-tell control, planted bad case, key anti-leak, and answer validation
to `gk blind make` and `gk blind score`.

Usage:
    # From a creature-bench folder (compares predicted vs body-read on Walk):
    python3 tools/creature_blind.py bench BENCH --out DIR/pack \
        [--animals fox,trex,velociraptor,bat] [--seed 20261116]

    # From explicit animated rig pairs:
    python3 tools/creature_blind.py make --out DIR/pack [--seed 20261116] \
        NAME=rig_a.glb,rig_b.glb[:clip[:paint_ref.glb]] ...

    # Score a blind judge's answers (`{"pair-01": "left", "pair-02": "cant_tell", ...}`):
    python3 tools/creature_blind.py score DIR/pack --answers answers.json \
        [--key DIR/pack.key.json] [--out DIR/pack.verdict.json]
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamecam import resolve_follow_fov_deg  # noqa: E402
import pick_score  # noqa: E402
from pngfont import read_png_rgb, write_png_rgb  # noqa: E402
from rigjudge import judge as judge_rig_metrics  # noqa: E402
import toolpaths  # noqa: E402

BLIND_QUESTION = "Same walk frames at game camera - which side moves cleaner? Answer left, right, or cant_tell."


def find_gk_bin():
    """Locate the `gk` binary (`GK_BIN`, `PATH`, sibling `_tools/gk`, `~/.cargo/bin/gk`)."""
    env_gk = os.environ.get("GK_BIN")
    if env_gk is not None and env_gk.strip():
        if os.path.isfile(env_gk) and os.access(env_gk, os.X_OK):
            return env_gk
        raise RuntimeError(f"GK_BIN={env_gk!r} does not point to an executable gk binary")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    games_dir = os.path.dirname(os.path.dirname(repo))
    tools_root = os.environ.get("GAMES_TOOLS_DIR") or os.path.join(games_dir, "_tools")
    home = os.path.expanduser("~")
    cands = [
        shutil.which("gk") or "",
        os.path.join(tools_root, "gk", "target", "release", "gk"),
        os.path.join(tools_root, "gk", "target", "debug", "gk"),
        os.path.join(home, ".cargo", "bin", "gk"),
    ]
    for p in cands:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise RuntimeError("gk binary not found (install gk, build _tools/gk, or set GK_BIN)")


def _stack_pngs_vertical(pngs, out_path):
    """Stack one or more synchronized same-size PNG frames vertically into one cell PNG."""
    if not pngs:
        raise ValueError("empty frame list")
    if len(pngs) == 1:
        shutil.copyfile(pngs[0], out_path)
        return
    imgs = [read_png_rgb(p) for p in pngs]
    cw, ch = imgs[0][0], imgs[0][1]
    for w, h, _ in imgs:
        if (w, h) != (cw, ch):
            raise ValueError(f"frame size mismatch {(w, h)} vs {(cw, ch)}")
    gap = 4
    total_h = len(imgs) * ch + max(0, len(imgs) - 1) * gap
    buf = bytearray([24, 26, 32] * (cw * total_h))
    row_bytes = cw * 3
    for idx, (_, _, px) in enumerate(imgs):
        y0 = idx * (ch + gap)
        for y in range(ch):
            src = y * row_bytes
            dst = (y0 + y) * row_bytes
            buf[dst:dst + row_bytes] = px[src:src + row_bytes]
    write_png_rgb(out_path, cw, total_h, buf)


# ---------------------------------------------------------- blind scoring

def norm_answer(ans):
    s = "".join(c for c in str(ans).strip().lower() if c not in " '-_")
    if s in ("canttell", "cant", "neither", "same", "equal", "tie"):
        return "cant_tell"
    if s in ("left", "l", "a"):
        return "left"
    if s in ("right", "r", "b"):
        return "right"
    return s


def _parse_pair_num(raw):
    m = re.search(r"(\d+)$", str(raw).strip())
    if not m:
        raise ValueError(f"invalid pair identifier: {raw!r}")
    return int(m.group(1))


def parse_answers(answers):
    """Accepts `{pair: ans}`, `{"answers": [{"pair": N, "better"|"answer"|"choice": ans, "why": ...}]}`,
    or a plain list of entry dicts."""
    norm_map = {}
    why_map = {}
    rows = answers.get("answers") if isinstance(answers, dict) and isinstance(answers.get("answers"), list) else answers
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict) or "pair" not in row:
                continue
            pnum = _parse_pair_num(row["pair"])
            raw = row.get("better", row.get("answer", row.get("choice", row.get("verdict", ""))))
            norm_map[pnum] = norm_answer(raw)
            if row.get("why"):
                why_map[pnum] = str(row["why"])
    elif isinstance(rows, dict):
        for k, v in rows.items():
            pnum = _parse_pair_num(k)
            if isinstance(v, dict):
                norm_map[pnum] = norm_answer(v.get("better", v.get("answer", v.get("choice", ""))))
                if v.get("why"):
                    why_map[pnum] = str(v["why"])
            else:
                norm_map[pnum] = norm_answer(v)
    return norm_map, why_map


def _score_via_gk(key, ans_by_id, out_dir=None, key_path=None):
    gk = find_gk_bin()
    tmp = tempfile.mkdtemp(prefix="gk_blind_score_")
    try:
        run_dir = out_dir if out_dir else os.path.join(tmp, "pack")
        run_key = key_path if key_path else os.path.join(tmp, "pack.key.json")
        if not out_dir or not os.path.exists(run_key):
            os.makedirs(run_dir, exist_ok=True)
            pub_pairs = []
            for p in key.get("pairs", []):
                pid = p.get("id") or f"pair-{int(p['pair']):02d}"
                pub_pairs.append({
                    "id": pid,
                    "left": f"{pid}-left.png",
                    "right": f"{pid}-right.png",
                    "sheet": f"{pid}.png",
                })
            with open(os.path.join(run_dir, "judge.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": 1,
                        "question": BLIND_QUESTION,
                        "answers": "one of left | right | cant_tell per pair",
                        "pairs": pub_pairs,
                    },
                    f,
                )
            with open(run_key, "w", encoding="utf-8") as f:
                json.dump(key, f)
        ans_file = os.path.join(tmp, "answers.json")
        with open(ans_file, "w", encoding="utf-8") as f:
            json.dump(ans_by_id, f)
        cmd = [gk, "blind", "score", run_dir, "--answers", ans_file, "--key", run_key, "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode not in (0, 1):
            raise RuntimeError(f"gk blind score failed ({res.returncode}): {(res.stderr or res.stdout).strip()}")
        try:
            return json.loads(res.stdout)
        except ValueError as exc:
            raise RuntimeError(f"gk blind score returned invalid JSON: {exc}\n{res.stdout}") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def score_blind_run(key, answers, out_dir=None, key_path=None):
    """Scores judge answers against `<out>.key.json` via `gk blind score`. Enforces
    mandatory can't-tell control and planted-bad visual + metric gates."""
    norm_map, why_map = parse_answers(answers)
    ans_by_id = {}
    for p in key.get("pairs", []):
        pnum = int(p["pair"])
        pid = p.get("id") or f"pair-{pnum:02d}"
        if pnum in norm_map:
            ans_by_id[pid] = norm_map[pnum]

    gk_verdict = _score_via_gk(key, ans_by_id, out_dir=out_dir, key_path=key_path)
    gk_rows = {row.get("pair"): row for row in gk_verdict.get("pairs", [])}

    pairs_out = []
    saw_control = False
    saw_planted = False
    control_ok = bool(gk_verdict.get("control_ok", False))
    planted_judge_ok = bool(gk_verdict.get("planted_ok", False))
    planted_metric_ok = False
    claims_judged = 0
    judge_agrees_truth = 0
    truth_claims = 0
    pick_agrees_judge = 0
    old_agrees_judge = 0
    old_retired = []
    pick_retired = []

    for p in key.get("pairs", []):
        pnum = int(p["pair"])
        pid = p.get("id") or f"pair-{pnum:02d}"
        ans = norm_map.get(pnum)
        cat = p["category"]
        gk_row = gk_rows.get(pid, {})
        judge_rig = p["left_rig"] if ans == "left" else (p["right_rig"] if ans == "right" else None)
        retired_old = False
        retired_new = False

        if cat == "control":
            saw_control = True
            rmse_ok = float(p.get("panel_rmse", p.get("rmse", 0.0))) == 0.0
            row_ctrl_ok = control_ok and (ans == "cant_tell") and rmse_ok
            control_ok = row_ctrl_ok
            note = (
                "control passed: identical renders answered cant_tell"
                if row_ctrl_ok
                else f"control FAILED: answered {ans!r} (rmse={p.get('panel_rmse', p.get('rmse'))}) — discard judge run"
            )
        elif cat == "planted":
            saw_planted = True
            planted_metric_ok = bool(p.get("planted_metric_ok", False))
            if p.get("expected") is not None:
                planted_judge_ok = planted_judge_ok and (judge_rig == p.get("expected"))
            if planted_judge_ok and planted_metric_ok:
                note = "planted bad case passed: picked out by blind judge and flagged by numeric score"
            elif not planted_judge_ok:
                note = f"planted bad case FAILED: judge answered {ans!r}, expected {p.get('expected')}"
            else:
                note = "planted bad case FAILED: numeric score did not flag the planted rig"
        else:
            if judge_rig is not None:
                claims_judged += 1
                tw = p.get("truth_winner")
                pw = p.get("pick_winner")
                ow = p.get("old_winner")
                if tw and tw != "tie":
                    truth_claims += 1
                    if judge_rig == tw:
                        judge_agrees_truth += 1
                if pw and judge_rig == pw:
                    pick_agrees_judge += 1
                if ow and judge_rig == ow:
                    old_agrees_judge += 1
                anchor = tw if (tw and tw != "tie") else pw
                if ow and anchor and judge_rig == anchor and ow != anchor:
                    retired_old = True
                    old_retired.append(pnum)
                if pw and tw and tw != "tie" and judge_rig == tw and pw != tw:
                    retired_new = True
                    pick_retired.append(pnum)
                if retired_old and not retired_new:
                    note = "judge, truth and new score agree; old score disagreed and is retired"
                elif retired_new:
                    note = "judge and truth agree against the new score: retired on this claim"
                elif pw and judge_rig == pw:
                    note = "judge agrees with calibrated pick_score"
                else:
                    note = f"judge picked {judge_rig}, pick_score said {pw}"
            elif ans == "cant_tell":
                note = "answered cant_tell at game distance"
            else:
                note = "missing or unrecognised answer"

        row_out = {
            "pair": pnum,
            "id": pid,
            "category": cat,
            "subject": p.get("subject", ""),
            "answer": ans,
            "winner": gk_row.get("winner"),
            "judge_rig": judge_rig,
            "truth_winner": p.get("truth_winner"),
            "pick_winner": p.get("pick_winner"),
            "old_winner": p.get("old_winner"),
            "old_score_retired": retired_old,
            "score_retired": retired_new,
            "note": note,
        }
        if pnum in why_map:
            row_out["why"] = why_map[pnum]
        pairs_out.append(row_out)

    passed = (
        saw_control
        and saw_planted
        and control_ok
        and planted_judge_ok
        and planted_metric_ok
        and bool(gk_verdict.get("pass", True))
    )
    return {
        "format": "creature-blind-verdict/1",
        "version": gk_verdict.get("version", 1),
        "pass": passed,
        "control_ok": control_ok,
        "planted_ok": planted_judge_ok,
        "planted_judge_ok": planted_judge_ok,
        "planted_metric_ok": planted_metric_ok,
        "claims": gk_verdict.get("claims", {}),
        "claims_judged": claims_judged,
        "truth_claims_judged": truth_claims,
        "judge_agrees_truth": judge_agrees_truth,
        "pick_agrees_judge": pick_agrees_judge,
        "old_agrees_judge": old_agrees_judge,
        "old_score_retired_pairs": old_retired,
        "score_retired_pairs": pick_retired,
        "pairs": pairs_out,
    }


# --------------------------------------------------------- rendering driver

def render_pair_spec(blender, script, rig_a, rig_b, clip, at, paint_ref, work, tag, metrics_path=None, fov_deg=None):
    """Measures shared camera bounds over both rigs, renders each rig in its
    own clean headless Blender process with that shared `--cam-spec`, and
    returns `(a_pngs, b_pngs, spec_info)`."""
    resolved_fov = resolve_follow_fov_deg(metrics_path=metrics_path, fov_deg=fov_deg)
    fov_args = ["--fov", f"{resolved_fov:.4f}"]
    if metrics_path:
        fov_args += ["--metrics", metrics_path]
    specs = []
    for k, rpath in enumerate((rig_a, rig_b)):
        cmd = [
            blender, "-b", "--factory-startup", "-P", script, "--",
            "--rig", rpath, "--clip", clip, "--at", at, "--measure-only", "yes",
        ] + fov_args
        if paint_ref:
            cmd += ["--paint-from", paint_ref]
        res = toolpaths.run(cmd, timeout=1800)
        lines = [ln for ln in res["out"].splitlines() if ln.startswith("RENDER_GAMECAM_SPEC ")]
        if res["code"] != 0 or not lines:
            raise RuntimeError(f"gamecam measure failed on {rpath}: {(res['err'] or res['out'])[-240:]}")
        specs.append(json.loads(lines[-1][len("RENDER_GAMECAM_SPEC "):]))

    ca, cb = specs[0]["cam_spec"], specs[1]["cam_spec"]
    u0, u1 = min(ca[0], cb[0]), max(ca[1], cb[1])
    v0, v1 = min(ca[2], cb[2]), max(ca[3], cb[3])
    ground_z = min(ca[5], cb[5])
    fov = math.radians(resolved_fov)
    span_u = max(u1 - u0, 1e-3) / 0.72
    span_v = max(v1 - v0, 1e-3) / 0.68
    dist = max(
        ca[4],
        cb[4],
        span_u / (2.0 * math.tan(fov * 0.5) * (560.0 / 360.0)),
        span_v / (2.0 * math.tan(fov * 0.5)),
    )
    cam_arg = ",".join(f"{x:.5f}" for x in (u0, u1, v0, v1, dist, ground_z))
    times = specs[0]["sampled_times"]
    times_arg = ",".join(f"{t:.4f}" for t in times)

    rendered = []
    for side_idx, rpath in enumerate((rig_a, rig_b)):
        prefix = os.path.join(work, f"{tag}_s{side_idx}")
        cmd = [
            blender, "-b", "--factory-startup", "-P", script, "--",
            "--rig", rpath, "--clip", clip, "--times", times_arg,
            "--cam-spec", cam_arg, "--out", prefix,
        ] + fov_args
        if paint_ref:
            cmd += ["--paint-from", paint_ref]
        res = toolpaths.run(cmd, timeout=1800)
        pngs = [f"{prefix}_{i + 1}.png" for i in range(len(times))]
        if res["code"] != 0 or not all(os.path.exists(p) for p in pngs):
            raise RuntimeError(f"gamecam render failed on {rpath}: {(res['err'] or res['out'])[-240:]}")
        rendered.append(pngs)
    return rendered[0], rendered[1], {"clip": specs[0]["clip"], "times": times, "cam_spec": cam_arg, "fov_deg": resolved_fov}

def _is_owned_blind_pack(out_dir, key_path):
    judge_file = os.path.join(out_dir, "judge.json")
    pairs_file = os.path.join(out_dir, "pairs.json")
    if os.path.islink(key_path) or not os.path.isfile(key_path):
        return False
    try:
        with open(key_path, encoding="utf-8") as f:
            kdoc = json.load(f)
    except (OSError, ValueError):
        return False
    key_ok = isinstance(kdoc, dict) and (
        (kdoc.get("version") == 1 and isinstance(kdoc.get("pairs"), list))
        or kdoc.get("format") == "creature-blind-key/1"
    )
    if not key_ok:
        return False
    if os.path.isfile(judge_file) and not os.path.islink(judge_file):
        try:
            with open(judge_file, encoding="utf-8") as f:
                jdoc = json.load(f)
            return isinstance(jdoc, dict) and jdoc.get("version") == 1 and isinstance(jdoc.get("pairs"), list)
        except (OSError, ValueError):
            return False
    if os.path.isfile(pairs_file) and not os.path.islink(pairs_file):
        try:
            with open(pairs_file, encoding="utf-8") as f:
                pdoc = json.load(f)
            return isinstance(pdoc, dict) and pdoc.get("format") == "creature-blind-pairs/1"
        except (OSError, ValueError):
            return False
    return False


def write_blind_pack(raw_items, out_dir, seed):
    """Stages synchronized pairs and delegates shuffling, side coin-flips,
    sheet compositing, control, planted bad case, and external key generation
    to `gk blind make`."""
    if not out_dir or not str(out_dir).strip():
        raise ValueError("refusing empty output directory")
    out_dir = os.path.abspath(str(out_dir).rstrip(os.sep) or os.sep)
    if out_dir == os.sep:
        raise ValueError(f"refusing filesystem root output directory {out_dir}")
    key_path = out_dir + ".key.json"
    if os.path.lexists(out_dir):
        if os.path.islink(out_dir) or not os.path.isdir(out_dir):
            raise ValueError(f"refusing to overwrite {out_dir}: not a directory")
        if os.listdir(out_dir) and not _is_owned_blind_pack(out_dir, key_path):
            raise ValueError(
                f"refusing to delete {out_dir}: not a blind pack (need judge.json and {key_path}, or an empty folder)"
            )

    gk = find_gk_bin()
    claims = [it for it in raw_items if it.get("category") == "claim"]
    controls = [it for it in raw_items if it.get("category") == "control"]
    planteds = [it for it in raw_items if it.get("category") == "planted"]

    stage = tempfile.mkdtemp(prefix="gk_blind_pack_")
    try:
        a_dir = os.path.join(stage, "a")
        b_dir = os.path.join(stage, "b")
        os.makedirs(a_dir)
        os.makedirs(b_dir)

        staged_claims = {}
        feed = claims if claims else (controls[:1] or planteds[:1])
        if not feed:
            raise ValueError("write_blind_pack needs at least one item")
        for idx, item in enumerate(feed, 1):
            subj = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item.get("subject", f"item_{idx}")))
            stem = f"{idx:03d}_{subj}"
            _stack_pngs_vertical(item["a_pngs"], os.path.join(a_dir, f"{stem}.png"))
            _stack_pngs_vertical(item["b_pngs"], os.path.join(b_dir, f"{stem}.png"))
            staged_claims[stem] = item

        stage_out = os.path.join(stage, "pack_out")
        cmd = [
            gk,
            "blind",
            "make",
            "--a",
            a_dir,
            "--b",
            b_dir,
            "--out",
            stage_out,
            "--seed",
            str(int(seed)),
            "--question",
            BLIND_QUESTION,
        ]
        if planteds:
            p_img = os.path.join(stage, "planted.png")
            _stack_pngs_vertical(planteds[0]["b_pngs"], p_img)
            cmd += ["--planted", p_img]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"gk blind make failed ({res.returncode}): {(res.stderr or res.stdout).strip()}")

        gk_key_file = stage_out + ".key.json"
        with open(gk_key_file, encoding="utf-8") as f:
            kdoc = json.load(f)

        ctrl_meta = controls[0] if controls else {}
        plant_meta = planteds[0] if planteds else {}
        enriched_pairs = []
        for idx, gp in enumerate(kdoc.get("pairs", []), 1):
            kind = gp.get("kind", "claim")
            left_tag = gp.get("left", "a")
            claim_stem = gp.get("claim", "")
            meta = staged_claims.get(claim_stem) if kind == "claim" else (ctrl_meta if kind == "control" else plant_meta)
            meta = meta or {}
            a_name = meta.get("a_name", "A")
            b_name = meta.get("b_name", "B")
            if kind == "planted":
                # In gk blind make, unplanted A is left_tag == "a"; planted side is "b"/"planted".
                left_rig = a_name if left_tag == "a" else b_name
                right_rig = b_name if left_tag == "a" else a_name
            elif kind == "control":
                left_rig = a_name
                right_rig = b_name
            else:
                left_rig = a_name if left_tag == "a" else b_name
                right_rig = b_name if left_tag == "a" else a_name

            entry = dict(gp)
            entry.update({
                "pair": idx,
                "category": kind,
                "subject": meta.get("subject", claim_stem),
                "clip": meta.get("clip", ""),
                "times": meta.get("times", []),
                "left_rig": left_rig,
                "right_rig": right_rig,
                "left_side": "A" if left_tag == "a" else "B",
                "right_side": "B" if left_tag == "a" else "A",
                "expected": meta.get("expected"),
                "panel_rmse": round(float(gp.get("rmse", 0.0)), 4),
            })
            if kind == "control" and entry["panel_rmse"] != 0.0:
                raise RuntimeError(f"control pair {idx} has non-zero pixel RMSE {entry['panel_rmse']}")
            for extra in (
                "truth_winner",
                "pick_winner",
                "old_winner",
                "skin_mean_pct",
                "rig_scores",
                "old_scores",
                "planted_metric_ok",
            ):
                if extra in meta:
                    entry[extra] = meta[extra]
            enriched_pairs.append(entry)

        kdoc["format"] = "creature-blind-key/1"
        kdoc["judged_dir"] = out_dir
        kdoc["pairs"] = enriched_pairs

        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        shutil.copytree(stage_out, out_dir)
        with open(key_path, "w", encoding="utf-8") as f:
            json.dump(kdoc, f, indent=1)
        inside = os.listdir(out_dir)
        if any("key" in fn.lower() for fn in inside):
            raise RuntimeError(f"answer key leaked into judged directory {out_dir}")
        return key_path
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def cmd_bench(a):
    home = os.path.expanduser("~")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sib = os.path.dirname(repo)
    blender = toolpaths.find("blender", a.blender, "BLENDER", ["/Applications/Blender.app/Contents/MacOS/Blender", os.path.join(home, ".local/bin/blender")])
    cb = toolpaths.find("creature-bench", a.creature_bench, "CREATURE_BENCH_BIN", [os.path.join(repo, "target/release/creature-bench")])
    cr = toolpaths.find("creature-rig", a.creature_rig, "CREATURE_RIG_BIN", [os.path.join(repo, "target/release/creature-rig")])
    wf = toolpaths.find("weights", a.weights, "WEIGHTS_BIN", [os.path.join(sib, "meshweights/rust/target/release/weights")])
    script = os.path.join(repo, "tools/bench/render_gamecam.py")

    animals = [x.strip() for x in a.animals.split(",") if x.strip()]
    work = tempfile.mkdtemp(prefix="creature_blind_")
    try:
        raw_items = []
        control_source = None
        for n in animals:
            d = os.path.join(a.bench, "animals", n)
            truth = os.path.join(d, "truth.glb")
            roles = os.path.join(d, "roles.json")
            nd = os.path.join(d, "new2")
            body_glb = os.path.join(nd, "try.glb")
            if not (os.path.exists(truth) and os.path.exists(body_glb)):
                raise SystemExit(f"{n}: missing truth.glb or new2/try.glb in {d}")

            b_m = judge_rig_metrics(cr, wf, cb, body_glb, work, f"{n}_body")
            preds = []
            for seed in (1, 2, 3):
                pg = os.path.join(nd, f"try.pred{seed}.glb")
                if not os.path.exists(pg):
                    continue
                pm = judge_rig_metrics(cr, wf, cb, pg, work, f"{n}_pred{seed}")
                preds.append({
                    "seed": seed,
                    "rigged": True,
                    "glb": pg,
                    "skeleton": pm["skeleton_raw"],
                    "tears_pct": pm["tears_pct"],
                    "anatomy": pm["anatomy"],
                    "rig_score": pm["rig_score"],
                    "old_score": pm["old_score"],
                })
            choice = pick_score.pick_candidate(
                {"skeleton": b_m["skeleton_raw"], "tears_pct": b_m["tears_pct"], "anatomy": b_m["anatomy"], "rig_score": b_m["rig_score"]},
                preds,
            )
            best_p = choice["best_predicted"]
            pred_glb = best_p["glb"]

            # Fit artist clips onto both candidates so both play identical motion in artist frame.
            pred_clips = os.path.join(work, f"{n}_pred.clips.glb")
            body_clips = os.path.join(work, f"{n}_body.clips.glb")
            pred_sc_j = os.path.join(work, f"{n}_pred.score.json")
            body_sc_j = os.path.join(work, f"{n}_body.score.json")
            extra_roles = ["--roles", roles] if os.path.exists(roles) else []
            for src_g, out_c, out_j in ((pred_glb, pred_clips, pred_sc_j), (body_glb, body_clips, body_sc_j)):
                res = toolpaths.run([cb, "score", truth, src_g, "--clips", out_c, "--json", out_j] + extra_roles, timeout=1800)
                if res["code"] != 0:
                    raise RuntimeError(f"score failed for {n}: {res['err'][-200:]}")
            ps = json.load(open(pred_sc_j))
            bs = json.load(open(body_sc_j))
            clip_names = [c["clip"] for c in ps["skin"]["clips"]]
            low = [c.lower() for c in clip_names]
            walk_clip = (
                next((clip_names[i] for i, c in enumerate(low) if "walk" in c and "slow" not in c), None)
                or next((clip_names[i] for i, c in enumerate(low) if "run" in c or "fly" in c), None)
                or clip_names[0]
            )
            a_pngs, b_pngs, rinfo = render_pair_spec(
                blender, script, pred_clips, body_clips, walk_clip, a.at, truth, work, f"claim_{n}", a.metrics, a.fov
            )
            ps_m, bs_m = ps["skin"]["mean_pct"], bs["skin"]["mean_pct"]
            truth_win = "tie" if abs(ps_m - bs_m) <= 0.02 else ("predicted" if ps_m < bs_m else "body")
            old_best = max(preds, key=lambda x: (x["old_score"], -x["seed"]))
            old_win = "predicted" if old_best["old_score"] > b_m["old_score"] else "body"

            raw_items.append({
                "category": "claim",
                "subject": n,
                "clip": rinfo["clip"],
                "times": rinfo["times"],
                "a_name": "predicted",
                "b_name": "body",
                "a_pngs": a_pngs,
                "b_pngs": b_pngs,
                "expected": truth_win,
                "truth_winner": truth_win,
                "pick_winner": choice["which"],
                "old_winner": old_win,
                "skin_mean_pct": {"predicted": ps_m, "body": bs_m},
                "rig_scores": {"predicted": best_p["rig_score"], "body": b_m["rig_score"]},
                "old_scores": {"predicted": old_best["old_score"], "body": b_m["old_score"]},
            })
            if control_source is None:
                control_source = (n, pred_clips, pred_glb, best_p["rig_score"], walk_clip, truth, a_pngs, rinfo)

        # Mandatory can't-tell control: two independent Blender renders of the same rig & camera.
        cn, c_clips, c_raw_glb, c_rscore, c_walk, c_truth, _, _ = control_source
        c_pngs_a, c_pngs_b, c_info = render_pair_spec(
            blender, script, c_clips, c_clips, c_walk, a.at, c_truth, work, f"ctrl_{cn}", a.metrics, a.fov
        )
        raw_items.append({
            "category": "control",
            "subject": cn,
            "clip": c_info["clip"],
            "times": c_info["times"],
            "a_name": "same_a",
            "b_name": "same_b",
            "a_pngs": c_pngs_a,
            "b_pngs": c_pngs_b,
            "expected": "cant_tell",
        })

        # Mandatory planted bad case: mirrored left/right limb weights swapped on c_clips.
        planted_clips = os.path.join(work, f"{cn}_planted.clips.glb")
        res = toolpaths.run([cb, "plant", c_clips, planted_clips], timeout=1800)
        if res["code"] != 0:
            raise RuntimeError(f"plant failed on {c_clips}: {res['err'][-200:]}")
        pl_m = judge_rig_metrics(cr, wf, cb, planted_clips, work, f"{cn}_planted")
        planted_metric_ok = (
            pl_m["rig_score"] < 80.0
            and (c_rscore - pl_m["rig_score"]) >= 15.0
            and (not pl_m["anatomy_pass"] or pl_m["tears_pct"] >= 5.0)
        )
        u_pngs, p_pngs, p_info = render_pair_spec(
            blender, script, c_clips, planted_clips, c_walk, a.at, c_truth, work, f"plant_{cn}", a.metrics, a.fov
        )
        raw_items.append({
            "category": "planted",
            "subject": cn,
            "clip": p_info["clip"],
            "times": p_info["times"],
            "a_name": "clean",
            "b_name": "planted_swapped_limbs",
            "a_pngs": u_pngs,
            "b_pngs": p_pngs,
            "expected": "clean",
            "planted_metric_ok": planted_metric_ok,
            "rig_scores": {"clean": c_rscore, "planted_swapped_limbs": pl_m["rig_score"]},
        })

        key_path = write_blind_pack(raw_items, a.out, a.seed)
        print(f"wrote {len(raw_items)} blind pairs to {os.path.abspath(a.out)} (key beside it at {key_path})")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cmd_make(a):
    home = os.path.expanduser("~")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sib = os.path.dirname(repo)
    blender = toolpaths.find("blender", a.blender, "BLENDER", ["/Applications/Blender.app/Contents/MacOS/Blender", os.path.join(home, ".local/bin/blender")])
    cb = toolpaths.find("creature-bench", a.creature_bench, "CREATURE_BENCH_BIN", [os.path.join(repo, "target/release/creature-bench")])
    cr = toolpaths.find("creature-rig", a.creature_rig, "CREATURE_RIG_BIN", [os.path.join(repo, "target/release/creature-rig")])
    wf = toolpaths.find("weights", a.weights, "WEIGHTS_BIN", [os.path.join(sib, "meshweights/rust/target/release/weights")])
    script = os.path.join(repo, "tools/bench/render_gamecam.py")

    work = tempfile.mkdtemp(prefix="creature_blind_make_")
    try:
        raw_items = []
        first_ref = None
        for spec in a.runs:
            name, sep, rest = spec.partition("=")
            parts = rest.split(":")
            rigs = parts[0].split(",")
            if not sep or len(rigs) != 2:
                raise SystemExit(f"bad spec {spec!r}: want NAME=rig_a.glb,rig_b.glb[:clip[:paint_ref.glb]]")
            clip = parts[1] if len(parts) > 1 else ""
            pref = parts[2] if len(parts) > 2 else rigs[0]
            ma = judge_rig_metrics(cr, wf, cb, rigs[0], work, f"{name}_a")
            mb = judge_rig_metrics(cr, wf, cb, rigs[1], work, f"{name}_b")
            a_pngs, b_pngs, rinfo = render_pair_spec(
                blender, script, rigs[0], rigs[1], clip, a.at, pref, work, f"claim_{name}", a.metrics, a.fov
            )
            p_win = "A" if ma["rig_score"] >= mb["rig_score"] else "B"
            o_win = "A" if ma["old_score"] >= mb["old_score"] else "B"
            raw_items.append({
                "category": "claim",
                "subject": name,
                "clip": rinfo["clip"],
                "times": rinfo["times"],
                "a_name": "A",
                "b_name": "B",
                "a_pngs": a_pngs,
                "b_pngs": b_pngs,
                "pick_winner": p_win,
                "old_winner": o_win,
                "rig_scores": {"A": ma["rig_score"], "B": mb["rig_score"]},
                "old_scores": {"A": ma["old_score"], "B": mb["old_score"]},
            })
            if first_ref is None:
                first_ref = (name, rigs[0], ma["rig_score"], clip, pref, a_pngs, rinfo)

        cn, c_glb, c_score, c_clip, c_pref, _, _ = first_ref
        c_pngs_a, c_pngs_b, c_info = render_pair_spec(blender, script, c_glb, c_glb, c_clip, a.at, c_pref, work, f"ctrl_{cn}", a.metrics, a.fov)
        raw_items.append({
            "category": "control",
            "subject": cn,
            "clip": c_info["clip"],
            "times": c_info["times"],
            "a_name": "same_a",
            "b_name": "same_b",
            "a_pngs": c_pngs_a,
            "b_pngs": c_pngs_b,
            "expected": "cant_tell",
        })

        planted_glb = os.path.join(work, f"{cn}_planted.glb")
        res = toolpaths.run([cb, "plant", c_glb, planted_glb], timeout=1800)
        if res["code"] != 0:
            raise RuntimeError(f"plant failed on {c_glb}: {res['err'][-200:]}")
        pl_m = judge_rig_metrics(cr, wf, cb, planted_glb, work, f"{cn}_planted")
        planted_metric_ok = (
            pl_m["rig_score"] < 80.0
            and (c_score - pl_m["rig_score"]) >= 15.0
            and (not pl_m["anatomy_pass"] or pl_m["tears_pct"] >= 5.0)
        )
        u_pngs, p_pngs, p_info = render_pair_spec(blender, script, c_glb, planted_glb, c_clip, a.at, c_pref, work, f"plant_{cn}", a.metrics, a.fov)
        raw_items.append({
            "category": "planted",
            "subject": cn,
            "clip": p_info["clip"],
            "times": p_info["times"],
            "a_name": "clean",
            "b_name": "planted_swapped_limbs",
            "a_pngs": u_pngs,
            "b_pngs": p_pngs,
            "expected": "clean",
            "planted_metric_ok": planted_metric_ok,
            "rig_scores": {"clean": c_score, "planted_swapped_limbs": pl_m["rig_score"]},
        })

        key_path = write_blind_pack(raw_items, a.out, a.seed)
        print(f"wrote {len(raw_items)} blind pairs to {os.path.abspath(a.out)} (key beside it at {key_path})")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cmd_score(a):
    out_dir = os.path.abspath(a.dir.rstrip("/"))
    key_path = a.key or (out_dir + ".key.json")
    with open(key_path, encoding="utf-8") as f:
        key = json.load(f)
    with open(a.answers, encoding="utf-8") as f:
        ans = json.load(f)
    verdict = score_blind_run(key, ans, out_dir=out_dir, key_path=key_path)
    vpath = a.out or os.path.join(out_dir, "verdict.json")
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=1)
    for p in verdict["pairs"]:
        print(f"  pair {p['pair']:02d} [{p['category']:7s}] ans={str(p['answer']):9s} -> {p['note']}")
    print(
        f"blind run {'PASS' if verdict['pass'] else 'FAIL'}: "
        f"control_ok={verdict['control_ok']}, planted_judge_ok={verdict['planted_judge_ok']}, "
        f"planted_metric_ok={verdict['planted_metric_ok']}; "
        f"claims={verdict['claims_judged']}, pick_agrees_judge={verdict['pick_agrees_judge']}/{verdict['claims_judged']}, "
        f"old_retired={verdict['old_score_retired_pairs']}, new_retired={verdict['score_retired_pairs']}"
    )
    return 0 if verdict["pass"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_bench = sub.add_parser("bench", help="build blind walk pairs from a creature-bench folder")
    p_bench.add_argument("bench")
    p_bench.add_argument("--out", required=True)
    p_bench.add_argument("--animals", default="fox,trex,velociraptor,bat")
    p_bench.add_argument("--seed", type=int, default=20261116)
    p_bench.add_argument("--at", default="0.20,0.50,0.80")
    for t in ("blender", "creature-bench", "creature-rig", "weights", "metrics", "fov"):
        p_bench.add_argument(f"--{t}", default="")

    p_make = sub.add_parser("make", help="build blind walk pairs from explicit animated GLBs")
    p_make.add_argument("--out", required=True)
    p_make.add_argument("--seed", type=int, default=20261116)
    p_make.add_argument("--at", default="0.20,0.50,0.80")
    for t in ("blender", "creature-bench", "creature-rig", "weights", "metrics", "fov"):
        p_make.add_argument(f"--{t}", default="")
    p_make.add_argument("runs", nargs="+", help="NAME=rig_a.glb,rig_b.glb[:clip[:paint_ref.glb]]")

    p_score = sub.add_parser("score", help="score blind judge answers against the external key")
    p_score.add_argument("dir")
    p_score.add_argument("--answers", required=True)
    p_score.add_argument("--key", default="")
    p_score.add_argument("--out", default="")

    a = ap.parse_args()
    if a.cmd == "bench":
        return cmd_bench(a)
    if a.cmd == "make":
        return cmd_make(a)
    return cmd_score(a)


if __name__ == "__main__":
    sys.exit(main())
