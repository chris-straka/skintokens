#!/usr/bin/env python3
"""Calibration and leave-one-out verification for `tools/pick_score.py`
against the 12 hand-made benchmark animals (stdlib only).

    python3 tools/bench/score_artists.py BENCH [--json OUT.json] [--picker-check]

1. Judges each animal's hand-made `truth.glb`, crude rigid-weight control
   (`creature-bench rigid`), and planted swapped-limb bad control
   (`creature-bench plant`) through the exact checks a candidate rig sees:
   `creature-rig check --rom`, `weights check --poses`, and
   `creature-bench anatomy`.
   Gates (exit 1 on any failure):
   - every hand-made artist rig scores >= 90 (`pick_score.TRUSTED_AT`),
   - every hand-made artist rig strictly outscores its crude control,
   - every hand-made artist rig outscores its planted bad control by >= 20
     and the planted bad control fails (< 80).
2. With `--picker-check`, evaluates the actual route candidates on disk in
   `animals/<name>/new2/` (`try.glb` = body-read vs `try.pred{1..3}.glb` =
   predicted seeds) against ground-truth `creature-bench score` (`skin` and
   `joints`), reporting both overall and leave-one-out cross-validated
   agreement (fitting `tear_override_pct` on 11 animals and testing on the
   held-out 12th).
"""
import argparse
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pick_score  # noqa: E402


def find(name, explicit, env, fallbacks):
    for p in [explicit, os.environ.get(env)] + fallbacks:
        if p and os.path.exists(p):
            return p
    raise SystemExit(f"{name} not found: set {env}")


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return r.returncode, r.stdout, r.stderr


def picker_baselines(picker, loo=None, seed=0):
    """Compute picker agreement alongside always-predicted, always-body, and
    coin-flip baselines (`skintokens-2`)."""
    names = sorted(picker)
    total = len(names)
    rng = random.Random(seed)
    coin_hits = sum(
        1 for n in names if rng.choice(("predicted", "body")) == picker[n].get("truth_winner")
    )
    loo_hits = sum(1 for v in (loo or {}).values() if v.get("agrees"))
    return {
        "total": total,
        "picker": sum(1 for v in picker.values() if v.get("agrees")),
        "leave_one_out": loo_hits,
        "old_score": sum(1 for v in picker.values() if v.get("old_agrees")),
        "always_predicted": sum(
            1 for v in picker.values() if v.get("truth_winner") == "predicted"
        ),
        "always_body": sum(1 for v in picker.values() if v.get("truth_winner") == "body"),
        "coin_flip_expected": total * 0.5,
        "coin_flip_seeded": coin_hits,
        "coin_flip_seed": seed,
    }


def format_picker_baselines(b):
    n = b["total"]
    return (
        f"picker agreement with ground truth: {b['picker']} of {n} "
        f"(leave-one-out: {b['leave_one_out']} of {n}; "
        f"old score: {b['old_score']} of {n}; "
        f"always-predicted: {b['always_predicted']} of {n}; "
        f"always-body: {b['always_body']} of {n}; "
        f"coin-flip: {b['coin_flip_expected']:.1f} of {n} expected, "
        f"{b['coin_flip_seeded']} of {n} seed={b['coin_flip_seed']})"
    )


def judge(cr, wf, cb, glb, work, tag, cut_at_limit=False):
    rom = os.path.join(work, tag + ".rom.ron")
    sj = os.path.join(work, tag + ".skel.json")
    aj = os.path.join(work, tag + ".anatomy.json")
    sk = pick_score.checked_json("creature-rig check", run([cr, "check", glb, "--json", sj, "--rom", rom]), sj)
    bend = pick_score.checked_json("weights check", run([wf, "check", glb, "--json", "--poses", rom]))
    an = pick_score.checked_json("creature-bench anatomy", run([cb, "anatomy", glb, "--json", aj]), aj)
    tears = pick_score.tears_pct(bend)
    score = pick_score.rig_score(sk, bend, an)
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
        "old": pick_score.old_score(bend.get("score", 0.0), sk.get("score", 0.0), bool(sk.get("pass"))),
        "rig_score": score,
        "anatomy": an,
        "skeleton_raw": sk,
        "cut_at_token_limit": cut_at_limit,
    }


def compact(j):
    return {k: v for k, v in j.items() if k not in ("anatomy", "skeleton_raw")}


def truth_score(cb, truth, glb, roles_path, out_json):
    cmd = [cb, "score", truth, glb, "--json", out_json]
    if os.path.exists(roles_path):
        cmd += ["--roles", roles_path]
    code, _, err = run(cmd)
    if code != 0 or not os.path.exists(out_json):
        raise RuntimeError(f"score failed on {glb}: {err[-160:]}")
    d = json.load(open(out_json))
    return {
        "skin_mean_pct": d["skin"]["mean_pct"],
        "skin_p95_pct": d["skin"]["p95_pct"],
        "joints_pct": d["skeleton"]["joint_error_pct"],
        "missing_chains": len(d["skeleton"]["missing_chains"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench")
    ap.add_argument("--json", default="")
    ap.add_argument("--picker-check", action="store_true")
    ap.add_argument("--work", default="")
    ap.add_argument("--creature-rig", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument("--creature-bench", default="")
    a = ap.parse_args()

    home = os.path.expanduser("~")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cr = find("creature-rig", a.creature_rig, "CREATURE_RIG_BIN", [
        os.path.join(repo, "target/release/creature-rig"),
    ])
    wf = find("weights", a.weights, "WEIGHTS_BIN", [
        os.path.join(os.path.dirname(repo), "meshweights/rust/target/release/weights"),
    ])
    cb = find("creature-bench", a.creature_bench, "CREATURE_BENCH_BIN", [
        os.path.join(repo, "target/release/creature-bench"),
    ])

    work = a.work or os.path.join(a.bench, "calibration")
    os.makedirs(work, exist_ok=True)
    names = [ln.split("\t")[0] for ln in open(os.path.join(a.bench, "animals.tsv")) if ln.strip()]

    rows, gate_failures = {}, []
    for n in names:
        d = os.path.join(a.bench, "animals", n)
        truth = os.path.join(d, "truth.glb")
        artist = judge(cr, wf, cb, truth, work, n + ".artist")
        row = {"artist": compact(artist)}

        crude = os.path.join(work, n + ".crude.glb")
        if run([cb, "rigid", truth, crude])[0] == 0:
            row["crude"] = compact(judge(cr, wf, cb, crude, work, n + ".crude"))

        planted = os.path.join(work, n + ".planted.glb")
        if run([cb, "plant", truth, planted])[0] == 0:
            row["planted"] = compact(judge(cr, wf, cb, planted, work, n + ".planted"))

        rows[n] = row
        reasons = []
        if artist["rig_score"] < pick_score.TRUSTED_AT:
            reasons.append(f"artist {artist['rig_score']} < {pick_score.TRUSTED_AT}")
        if "crude" in row and artist["rig_score"] <= row["crude"]["rig_score"]:
            reasons.append(f"artist {artist['rig_score']} <= crude {row['crude']['rig_score']}")
        if "planted" in row and (
            row["planted"]["rig_score"] >= 80.0
            or artist["rig_score"] - row["planted"]["rig_score"] < 20.0
        ):
            reasons.append(f"planted {row['planted']['rig_score']} not flagged")
        if reasons:
            gate_failures.append(f"{n}: {'; '.join(reasons)}")

        print(
            f"{n:13s} artist {artist['rig_score']:5.1f} (skel {artist['skeleton']}, "
            f"tears {artist['tears_pct']}%, old {artist['old']:5.1f})"
            + (f" | crude {row['crude']['rig_score']:5.1f}" if "crude" in row else "")
            + (f" | planted {row['planted']['rig_score']:5.1f}" if "planted" in row else "")
            + (f"  FAIL: {'; '.join(reasons)}" if reasons else "")
        )

    picker, loo = {}, {}
    if a.picker_check:
        cands = {}
        for n in names:
            d = os.path.join(a.bench, "animals", n)
            truth = os.path.join(d, "truth.glb")
            roles = os.path.join(d, "roles.json")
            nd = os.path.join(d, "new2")
            body_glb = os.path.join(nd, "try.glb")
            if not os.path.exists(body_glb):
                continue
            b_j = judge(cr, wf, cb, body_glb, work, f"{n}.body")
            b_t = truth_score(cb, truth, body_glb, roles, os.path.join(work, f"{n}.body.truth.json"))
            body_cand = {
                "skeleton": b_j["skeleton_raw"],
                "tears_pct": b_j["tears_pct"],
                "anatomy": b_j["anatomy"],
                "rig_score": b_j["rig_score"],
                "old_rank": b_j["old"],
                "truth": b_t,
            }
            preds = []
            for seed in (1, 2, 3):
                p_glb = os.path.join(nd, f"try.pred{seed}.glb")
                if not os.path.exists(p_glb):
                    continue
                skj = os.path.join(nd, f"try.pred{seed}.skintokens.json")
                cut = False
                if os.path.exists(skj):
                    try:
                        cut = bool((json.load(open(skj)).get("rig") or {}).get("skeleton_cut_at_token_limit"))
                    except (OSError, ValueError):
                        pass
                p_j = judge(cr, wf, cb, p_glb, work, f"{n}.pred{seed}", cut_at_limit=cut)
                p_t = truth_score(cb, truth, p_glb, roles, os.path.join(work, f"{n}.pred{seed}.truth.json"))
                preds.append({
                    "seed": seed,
                    "rigged": True,
                    "cut_at_token_limit": cut,
                    "skeleton": p_j["skeleton_raw"],
                    "tears_pct": p_j["tears_pct"],
                    "anatomy": p_j["anatomy"],
                    "rig_score": p_j["rig_score"],
                    "old_rank": p_j["old"],
                    "truth": p_t,
                })
            if preds:
                cands[n] = (body_cand, preds)

        def agrees(choice, b_truth, p_truth):
            # Within 0.02% skin error is a tie -> break by joint error.
            ds = p_truth["skin_mean_pct"] - b_truth["skin_mean_pct"]
            if abs(ds) <= 0.02:
                want = "predicted" if p_truth["joints_pct"] <= b_truth["joints_pct"] else "body"
            else:
                want = "predicted" if ds < 0 else "body"
            return choice == want, want

        for n, (b, ps) in cands.items():
            res = pick_score.pick_candidate(b, ps)
            best_p = res["best_predicted"]
            ok_new, want = agrees(res["which"], b["truth"], best_p["truth"])
            old_p = max(ps, key=lambda x: (x["old_rank"], -x["seed"]))
            old_which = "predicted" if old_p["old_rank"] > b["old_rank"] else "body"
            ok_old, _ = agrees(old_which, b["truth"], old_p["truth"])
            picker[n] = {
                "truth_winner": want,
                "picked": res["which"],
                "picked_seed": best_p["seed"],
                "agrees": ok_new,
                "old_picked": old_which,
                "old_agrees": ok_old,
                "reason": res["reason"],
                "trusted": res["trusted"],
                "body_truth": b["truth"],
                "pred_truth": best_p["truth"],
                "body_rig_score": b["rig_score"],
                "pred_rig_score": best_p["rig_score"],
            }
            print(
                f"{n:13s} truth={want:9s} (pred s{best_p['seed']} skin {best_p['truth']['skin_mean_pct']:.2f}% j {best_p['truth']['joints_pct']:.2f}% "
                f"vs body skin {b['truth']['skin_mean_pct']:.2f}% j {b['truth']['joints_pct']:.2f}%) | "
                f"pick={res['which']:9s} ({'OK' if ok_new else 'MISS'}) | old={old_which:9s}"
            )

        # Leave-one-out cross-validation over tear_override_pct grid (prefer
        # threshold nearest the hand-made max ROM tear rate on training ties).
        grid = [8.0, 5.0, 10.0, 3.0, 2.0]
        loo_hits = 0
        for held in cands:
            best_thr = max(
                grid,
                key=lambda thr: sum(
                    1
                    for m, (bm, psm) in cands.items()
                    if m != held
                    and agrees(
                        pick_score.pick_candidate(bm, psm, tear_override_pct=thr)["which"],
                        bm["truth"],
                        pick_score.pick_candidate(bm, psm, tear_override_pct=thr)["best_predicted"]["truth"],
                    )[0]
                ),
            )
            bh, psh = cands[held]
            rh = pick_score.pick_candidate(bh, psh, tear_override_pct=best_thr)
            hit, _ = agrees(rh["which"], bh["truth"], rh["best_predicted"]["truth"])
            loo[held] = {"fit_tear_override_pct": best_thr, "picked": rh["which"], "agrees": hit}
            if hit:
                loo_hits += 1
        baselines = {}
        if picker:
            baselines = picker_baselines(picker, loo=loo)
            print(format_picker_baselines(baselines))

    if a.json:
        with open(a.json, "w") as f:
            json.dump(
                {
                    "artists": rows,
                    "gate_failures": gate_failures,
                    "picker": picker,
                    "leave_one_out": loo,
                    "baselines": baselines if a.picker_check else {},
                },
                f,
                indent=1,
            )
    if gate_failures:
        print("CALIBRATION FAILED:\n  " + "\n  ".join(gate_failures))
        return 1
    print(f"all {len(names)} hand-made rigs >= {pick_score.TRUSTED_AT:.0f} and beat crude & planted controls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
