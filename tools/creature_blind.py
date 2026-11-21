#!/usr/bin/env python3
"""Blind same-frame walk/pose A/B benchmark at the game camera (stdlib only).

Builds unmarked LEFT/RIGHT walk-frame pair sheets (`pair_01.png` ...) at
HLH's follow camera with:
- seeded `LEFT`/`RIGHT` coin-flip and shuffled presentation order (`--seed`),
- mandatory **can't-tell control** (identical render on both sides, verified
  `pixel_rmse == 0.0` below the header bar),
- mandatory **planted bad case** (`creature-bench plant`: mirrored left/right
  limb skin weights swapped, verified flagged by `pick_score` and `anatomy` /
  `tears`),
- answer key written **beside** the judged folder (`<out>.key.json`), never
  inside `<out>/`.

Usage:
    # From a creature-bench folder (compares predicted vs body-read on Walk):
    python3 tools/creature_blind.py bench BENCH --out DIR/pack \
        [--animals fox,trex,velociraptor,bat] [--seed 20261116]

    # From explicit animated rig pairs:
    python3 tools/creature_blind.py make --out DIR/pack [--seed 20261116] \
        NAME=rig_a.glb,rig_b.glb[:clip[:paint_ref.glb]] ...

    # Score a blind judge's answers (`{"1": "left", "2": "cant_tell", ...}`):
    python3 tools/creature_blind.py score DIR/pack --answers answers.json \
        [--key DIR/pack.key.json] [--out DIR/pack.verdict.json]
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pick_score  # noqa: E402
from pngfont import FONT_5X7, draw_text, read_png_rgb, text_width, write_png_rgb  # noqa: E402
from rigjudge import judge as judge_rig_metrics  # noqa: E402
import toolpaths  # noqa: E402


# ------------------------------------------------ seeded coin-flip & shuffle

def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def a_on_left(seed, pair_idx):
    s = (int(seed) ^ ((int(pair_idx) + 1) * 0xD1B54A32D192ED03)) & 0xFFFFFFFFFFFFFFFF
    _, r = splitmix64(s)
    return (r & 1) == 0


def presentation_order(n, seed):
    idx = list(range(n))
    s = (int(seed) ^ 0xA0761D6478BD642F) & 0xFFFFFFFFFFFFFFFF
    for i in range(n - 1, 0, -1):
        s, r = splitmix64(s)
        j = r % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


# ----------------------------------------- pair sheet compositor

HEAD_H = 62
GAP = 4


def compose_pair_sheet(left_pngs, right_pngs, pair_num, out_path):
    """Composites synchronized LEFT and RIGHT frame lists into `pair_NN.png`
    and returns `(sheet_w, sheet_h, panel_rmse)` where `panel_rmse` compares
    all LEFT pixels against all RIGHT pixels directly (0.0 on the control)."""
    if len(left_pngs) != len(right_pngs) or not left_pngs:
        raise ValueError("mismatched left/right frame lists")
    l_imgs = [read_png_rgb(p) for p in left_pngs]
    r_imgs = [read_png_rgb(p) for p in right_pngs]
    cw, ch = l_imgs[0][0], l_imgs[0][1]
    for w, h, _ in l_imgs + r_imgs:
        if (w, h) != (cw, ch):
            raise ValueError(f"frame size mismatch {(w, h)} vs {(cw, ch)}")

    rows = len(l_imgs)
    W = cw * 2 + GAP
    H = HEAD_H + rows * ch + max(0, rows - 1) * GAP
    buf = bytearray([24, 26, 32] * (W * H))

    draw_text(buf, W, H, 12, 8, f"PAIR {pair_num:02d}", 3, (235, 238, 244))
    prompt = "SAME WALK FRAMES AT GAME CAMERA - WHICH SIDE MOVES CLEANER? ANSWER LEFT, RIGHT, OR CANT_TELL"
    draw_text(buf, W, H, 180, 14, prompt, 1, (164, 170, 182))
    lw = text_width("LEFT", 2)
    rw = text_width("RIGHT", 2)
    draw_text(buf, W, H, (cw - lw) // 2, 40, "LEFT", 2, (235, 238, 244))
    draw_text(buf, W, H, cw + GAP + (cw - rw) // 2, 40, "RIGHT", 2, (235, 238, 244))

    sq_sum = 0.0
    total_vals = 0
    for r_idx, ((_, _, lb), (_, _, rb)) in enumerate(zip(l_imgs, r_imgs)):
        y_off = HEAD_H + r_idx * (ch + GAP)
        for y in range(ch):
            s0 = y * cw * 3
            s1 = s0 + cw * 3
            d_left = ((y_off + y) * W) * 3
            d_right = ((y_off + y) * W + cw + GAP) * 3
            buf[d_left:d_left + cw * 3] = lb[s0:s1]
            buf[d_right:d_right + cw * 3] = rb[s0:s1]
        for a, b in zip(lb, rb):
            diff = a - b
            sq_sum += diff * diff
        total_vals += len(lb)

    write_png_rgb(out_path, W, H, buf)
    rmse = math.sqrt(sq_sum / max(total_vals, 1))
    return W, H, round(rmse, 4)


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


def parse_answers(answers):
    """Accepts `{pair: ans}`, `{"answers": [{"pair": N, "better"|"answer": ans, "why": ...}]}`,
    or a plain list of entry dicts."""
    norm_map = {}
    why_map = {}
    rows = answers.get("answers") if isinstance(answers, dict) and isinstance(answers.get("answers"), list) else answers
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict) or "pair" not in row:
                continue
            pnum = int(row["pair"])
            raw = row.get("better", row.get("answer", row.get("verdict", "")))
            norm_map[pnum] = norm_answer(raw)
            if row.get("why"):
                why_map[pnum] = str(row["why"])
    elif isinstance(rows, dict):
        for k, v in rows.items():
            pnum = int(str(k).replace("pair_", "").replace("pair", "").strip())
            if isinstance(v, dict):
                norm_map[pnum] = norm_answer(v.get("better", v.get("answer", "")))
                if v.get("why"):
                    why_map[pnum] = str(v["why"])
            else:
                norm_map[pnum] = norm_answer(v)
    return norm_map, why_map


def score_blind_run(key, answers):
    """Scores judge answers against `<out>.key.json`. Enforces mandatory
    can't-tell control and planted-bad visual + metric gates."""
    norm_map, why_map = parse_answers(answers)
    pairs_out = []
    saw_control = False
    saw_planted = False
    control_ok = False
    planted_judge_ok = False
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
        ans = norm_map.get(pnum)
        cat = p["category"]
        judge_rig = p["left_rig"] if ans == "left" else (p["right_rig"] if ans == "right" else None)
        retired_old = False
        retired_new = False

        if cat == "control":
            saw_control = True
            rmse_ok = float(p.get("panel_rmse", 0.0)) == 0.0
            control_ok = (ans == "cant_tell") and rmse_ok
            note = (
                "control passed: identical renders answered cant_tell"
                if control_ok
                else f"control FAILED: answered {ans!r} (rmse={p.get('panel_rmse')}) — discard judge run"
            )
        elif cat == "planted":
            saw_planted = True
            planted_judge_ok = judge_rig == p.get("expected")
            planted_metric_ok = bool(p.get("planted_metric_ok", False))
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
            "category": cat,
            "subject": p.get("subject", ""),
            "answer": ans,
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
    )
    return {
        "format": "creature-blind-verdict/1",
        "pass": passed,
        "control_ok": control_ok,
        "planted_judge_ok": planted_judge_ok,
        "planted_metric_ok": planted_metric_ok,
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

def render_pair_spec(blender, script, rig_a, rig_b, clip, at, paint_ref, work, tag):
    """Measures shared camera bounds over both rigs, renders each rig in its
    own clean headless Blender process with that shared `--cam-spec`, and
    returns `(a_pngs, b_pngs, spec_info)`."""
    specs = []
    for k, rpath in enumerate((rig_a, rig_b)):
        cmd = [
            blender, "-b", "--factory-startup", "-P", script, "--",
            "--rig", rpath, "--clip", clip, "--at", at, "--measure-only", "yes",
        ]
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
    fov = math.radians(45.0)
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
        ]
        if paint_ref:
            cmd += ["--paint-from", paint_ref]
        res = toolpaths.run(cmd, timeout=1800)
        pngs = [f"{prefix}_{i + 1}.png" for i in range(len(times))]
        if res["code"] != 0 or not all(os.path.exists(p) for p in pngs):
            raise RuntimeError(f"gamecam render failed on {rpath}: {(res['err'] or res['out'])[-240:]}")
        rendered.append(pngs)
    return rendered[0], rendered[1], {"clip": specs[0]["clip"], "times": times, "cam_spec": cam_arg}


def _is_owned_blind_pack(out_dir, key_path):
    pairs_file = os.path.join(out_dir, "pairs.json")
    if (
        not os.path.isfile(pairs_file)
        or os.path.islink(pairs_file)
        or not os.path.isfile(key_path)
        or os.path.islink(key_path)
    ):
        return False
    try:
        with open(pairs_file, encoding="utf-8") as f:
            pdoc = json.load(f)
        with open(key_path, encoding="utf-8") as f:
            kdoc = json.load(f)
    except (OSError, ValueError):
        return False
    return (
        isinstance(pdoc, dict)
        and pdoc.get("format") == "creature-blind-pairs/1"
        and isinstance(kdoc, dict)
        and kdoc.get("format") == "creature-blind-key/1"
    )


def write_blind_pack(raw_items, out_dir, seed):
    """Shuffles `raw_items` with `seed`, coins LEFT/RIGHT per pair, writes
    `pair_NN.png` and `pairs.json` into `out_dir`, and writes
    `<out_dir>.key.json` beside `out_dir`."""
    if not out_dir or not str(out_dir).strip():
        raise ValueError("refusing empty output directory")
    out_dir = os.path.abspath(str(out_dir).rstrip(os.sep) or os.sep)
    key_path = out_dir + ".key.json"
    if os.path.lexists(out_dir):
        if os.path.islink(out_dir) or not os.path.isdir(out_dir):
            raise ValueError(f"refusing to overwrite {out_dir}: not a directory")
        if os.listdir(out_dir):
            if not _is_owned_blind_pack(out_dir, key_path):
                raise ValueError(
                    f"refusing to delete {out_dir}: not a blind pack (need pairs.json and {key_path}, or an empty folder)"
                )
            shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    order = presentation_order(len(raw_items), seed)
    pairs_public = []
    pairs_key = []
    for pos_idx, src_idx in enumerate(order):
        pair_num = pos_idx + 1
        item = raw_items[src_idx]
        left_is_a = a_on_left(seed, pair_num)
        if left_is_a:
            l_pngs, r_pngs = item["a_pngs"], item["b_pngs"]
            l_rig, r_rig = item["a_name"], item["b_name"]
        else:
            l_pngs, r_pngs = item["b_pngs"], item["a_pngs"]
            l_rig, r_rig = item["b_name"], item["a_name"]

        img_name = f"pair_{pair_num:02d}.png"
        img_path = os.path.join(out_dir, img_name)
        w, h, rmse = compose_pair_sheet(l_pngs, r_pngs, pair_num, img_path)
        if item["category"] == "control" and rmse != 0.0:
            raise RuntimeError(f"control pair {pair_num} has non-zero pixel RMSE {rmse}")

        pairs_public.append({
            "pair": pair_num,
            "image": img_name,
            "frames": len(l_pngs),
            "size": [w, h],
        })
        entry = {
            "pair": pair_num,
            "category": item["category"],
            "subject": item["subject"],
            "clip": item.get("clip", ""),
            "times": item.get("times", []),
            "left_rig": l_rig,
            "right_rig": r_rig,
            "expected": item.get("expected"),
            "panel_rmse": rmse,
        }
        for extra in (
            "truth_winner",
            "pick_winner",
            "old_winner",
            "skin_mean_pct",
            "rig_scores",
            "old_scores",
            "planted_metric_ok",
        ):
            if extra in item:
                entry[extra] = item[extra]
        pairs_key.append(entry)

    with open(os.path.join(out_dir, "pairs.json"), "w") as f:
        json.dump(
            {
                "format": "creature-blind-pairs/1",
                "seed": int(seed),
                "question": "Which side's creature rig walks and bends cleaner across the same frames? Answer left, right, or cant_tell per pair.",
                "pairs": pairs_public,
            },
            f,
            indent=1,
        )
    # Guardrail: key sits BESIDE out_dir, never inside it.
    with open(key_path, "w") as f:
        json.dump(
            {
                "format": "creature-blind-key/1",
                "seed": int(seed),
                "judged_dir": out_dir,
                "pairs": pairs_key,
            },
            f,
            indent=1,
        )
    inside = os.listdir(out_dir)
    if any("key" in fn.lower() for fn in inside):
        raise RuntimeError(f"answer key leaked into judged directory {out_dir}")
    return key_path


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
                blender, script, pred_clips, body_clips, walk_clip, a.at, truth, work, f"claim_{n}"
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
            blender, script, c_clips, c_clips, c_walk, a.at, c_truth, work, f"ctrl_{cn}"
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
            blender, script, c_clips, planted_clips, c_walk, a.at, c_truth, work, f"plant_{cn}"
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
                blender, script, rigs[0], rigs[1], clip, a.at, pref, work, f"claim_{name}"
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
        c_pngs_a, c_pngs_b, c_info = render_pair_spec(blender, script, c_glb, c_glb, c_clip, a.at, c_pref, work, f"ctrl_{cn}")
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
        u_pngs, p_pngs, p_info = render_pair_spec(blender, script, c_glb, planted_glb, c_clip, a.at, c_pref, work, f"plant_{cn}")
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
    key = json.load(open(key_path))
    ans = json.load(open(a.answers))
    verdict = score_blind_run(key, ans)
    vpath = a.out or os.path.join(out_dir, "verdict.json")
    with open(vpath, "w") as f:
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
    for t in ("blender", "creature-bench", "creature-rig", "weights"):
        p_bench.add_argument(f"--{t}", default="")

    p_make = sub.add_parser("make", help="build blind walk pairs from explicit animated GLBs")
    p_make.add_argument("--out", required=True)
    p_make.add_argument("--seed", type=int, default=20261116)
    p_make.add_argument("--at", default="0.20,0.50,0.80")
    for t in ("blender", "creature-bench", "creature-rig", "weights"):
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
