#!/usr/bin/env python3
"""Calibrated creature rig score and candidate picker (stdlib only).

Why the pre-2026-11-16 creature score could not be trusted:
- `creature-rig check` previously read skinned meshes by their node matrix
  while reading bones by their joint world matrices; on artist exports with
  a scale/turn node above the armature, the mesh sat at a sixth of the
  bones' size and all 12 hand-made rigs scored ~15 of 100.
- Multiplying weightforge's 0-100 bend energy score (which halves the worst
  region and rates hand-painted rigs 7 to 25) by `creature-rig check`
  punished hand-made rigs and favoured `creature-rig`'s own fewer,
  surface-pivoted bones by circular construction.
- Even after fixing `creature-rig check`, ranking candidates by
  `skeleton_score - 0.8 * tears` still gave the body-read rig +1 to +2
  circular points on its own medial lines and agreed with ground-truth
  skin error on only 4 of 12 animals.

How a rig is scored and picked now:
1. `rig_score` (0 to 100, calibrated on 12 hand-made animals via
   `tools/bench/score_artists.py`):
   - starts from `creature-rig check`'s `skeleton_score`,
   - subtracts `0.8 * tears_pct` (vertices in failing `D_STRETCH`,
     `D_VOLUME`, `D_BLEED` findings),
   - subtracts 12 when `creature-rig check` fails,
   - subtracts `20 + 5 * (fails - 1)` when the reader-independent
     `creature-bench anatomy` check fails (`B_NO_LEG`, `B_NO_HEAD`,
     `B_NO_ARMS`, `B_LOPSIDED`, `B_KIND`, `B_OFF_CENTRE`, `B_SWAPPED`),
   - subtracts up to 8 when skin weights are rigid/unblended
     (`blended_share < 0.12`: 0.0 on crude nearest-bone controls, > 0.15 on
     hand-painted and smoothly skinned rigs).
   All 12 hand-made artist rigs score 92 to 100 (`>= TRUSTED_AT`), beat
   their crude rigid-weight controls (12 of 12), and beat planted
   swapped-limb rigs by > 30 points.
2. `pick_candidate` (reader-neutral choice between predicted ML seeds and
   the body-read rig):
   - ranks predicted seeds by `(anatomy_pass, skeleton_pass, not cut,
     rig_score, articulated_pairs, -seed)`, eliminating broken seeds;
   - compares the best predicted rig against the body-read rig without
     `creature-rig`'s circular self-line bonus: clean `anatomy_pass` (and
     not cut at token limit) first, then articulated limb pairs (provided
     the predicted rig did not explode into cloth chains), then the learned
     rig as the base (since its hip/shoulder joints sit inside the haunch
     and chest rather than at skin emergence) unless the body-read rig
     tears substantially less (`> TEAR_OVERRIDE_PCT`) or clears `TRUSTED_AT`
     by `TRUST_MARGIN` where the learned rig does not.
"""

TEAR_CODES = ("D_STRETCH", "D_VOLUME", "D_BLEED")
TRUSTED_AT = 90.0
TRUST_MARGIN = 3.0
BLEND_FLOOR = 0.12
# Hand-made artist rigs tear up to ~9% on synthetic ROM + clips; only override
# a >=90 predicted rig when its excess tears exceed the hand-made ROM range.
TEAR_OVERRIDE_PCT = 8.0


def tears_pct(bend):
    """Vertices in failing tear findings (`D_STRETCH`, `D_VOLUME`,
    `D_BLEED`) of a `weights check` report, as a percentage of vertices."""
    verts = max(int((bend or {}).get("verts", 0)), 1)
    torn = sum(
        f.get("verts", 0)
        for f in (bend or {}).get("findings", [])
        if f.get("severity") == "fail" and f.get("code") in TEAR_CODES
    )
    return round(min(100.0, 100.0 * torn / verts), 2)


def anatomy_summary(anatomy):
    """Extracts pass, fail count, fail codes, and articulated limb pairs
    (side-limb pairs with >= 2 joints) from `creature-bench anatomy` JSON."""
    if not anatomy:
        return {"present": False, "pass": True, "fails": 0, "codes": [], "articulated_pairs": 0}
    codes = [
        f.get("code", "")
        for f in anatomy.get("findings", [])
        if f.get("severity") == "fail"
    ]
    limbs = anatomy.get("limbs", [])
    by_name = {l.get("from"): l for l in limbs if l.get("from")}
    art_pairs = 0
    for l in limbs:
        if l.get("side") == "+x" and l.get("partner") and int(l.get("joints", 0)) >= 2:
            p = by_name.get(l.get("partner"))
            if p is None or int(p.get("joints", 0)) >= 2:
                art_pairs += 1
    return {
        "present": True,
        "pass": bool(anatomy.get("pass", not codes)),
        "fails": len(codes),
        "codes": codes,
        "articulated_pairs": art_pairs,
    }


def rig_score(skeleton, bend, anatomy=None):
    """0 to 100 quality score for a single rig, calibrated so hand-made
    artist rigs score 90+ while crude rigid controls, missing-limb rigs,
    and swapped-limb planted bad rigs drop below it."""
    sk_score = float((skeleton or {}).get("score", 0.0))
    sk_pass = bool((skeleton or {}).get("pass", False))
    tears = tears_pct(bend) if isinstance(bend, dict) else float(bend)
    an = anatomy_summary(anatomy)

    pen = 0.8 * tears
    if not sk_pass:
        pen += 12.0
    if an["present"] and not an["pass"]:
        pen += 20.0 + 5.0 * max(0, an["fails"] - 1)
    blend = (skeleton or {}).get("blended_share")
    if blend is not None and float(blend) < BLEND_FLOOR:
        pen += 8.0 * (1.0 - max(0.0, float(blend)) / BLEND_FLOOR)
    return round(max(0.0, min(100.0, sk_score - pen)), 1)


def fair_score(skeleton_score, tears, skeleton_pass=True, anatomy=None, blended_share=None):
    """Convenience wrapper matching `rig_score` from scalar inputs."""
    sk = {"score": skeleton_score, "pass": skeleton_pass}
    if blended_share is not None:
        sk["blended_share"] = blended_share
    return rig_score(sk, tears, anatomy=anatomy)


def old_score(bend_score, skeleton_score, skeleton_pass):
    """Pre-calibration creature score (`bend * skeleton / 100 - 15` when
    skeleton fails), kept only so calibration tables can show what was
    retired."""
    return round(bend_score * skeleton_score / 100.0 - (0.0 if skeleton_pass else 15.0), 1)


def trusted(winner_score, loser_score=None, anatomy_ok=True):
    """A single-score verdict is trusted when the winner clears `TRUSTED_AT`,
    passes anatomy, and (when compared by scalar score alone) leads by
    `TRUST_MARGIN`."""
    if not anatomy_ok or winner_score < TRUSTED_AT:
        return False
    return loser_score is None or (winner_score - loser_score) >= TRUST_MARGIN


def rank_predicted_seed(cand):
    """Sort key for choosing the best seed among `skintokens rig` tries."""
    an = anatomy_summary(cand.get("anatomy"))
    sk = cand.get("skeleton") or {}
    cut = bool(cand.get("cut_at_token_limit", False))
    score = cand.get("rig_score")
    if score is None:
        score = rig_score(sk, cand.get("tears_pct", 100.0), cand.get("anatomy"))
    return (
        1 if an["pass"] else 0,
        0 if cut else 1,
        1 if sk.get("pass") else 0,
        score,
        an["articulated_pairs"],
        -int(cand.get("seed", 0)),
    )


def pick_candidate(body, predicted_tries, tear_override_pct=TEAR_OVERRIDE_PCT):
    """Chooses between `body` (`creature-rig rig`) and `predicted_tries`
    (zero or more `skintokens rig` seed dicts) without circular self-line
    bias. Returns `{"which": "predicted"|"body", "best_predicted": dict|None,
    "rig_score": float, "trusted": bool, "reason": str}`."""
    b_an = anatomy_summary((body or {}).get("anatomy"))
    b_sk = (body or {}).get("skeleton") or {}
    b_tears = float((body or {}).get("tears_pct", 100.0))
    b_score = (body or {}).get("rig_score")
    if b_score is None:
        b_score = rig_score(b_sk, b_tears, (body or {}).get("anatomy"))

    good = [p for p in (predicted_tries or []) if p.get("rigged", True)]
    if not good:
        return {
            "which": "body",
            "best_predicted": None,
            "rig_score": b_score,
            "trusted": b_an["pass"] and b_score >= TRUSTED_AT,
            "reason": "no predicted candidate available",
        }

    pred = max(good, key=rank_predicted_seed)
    p_an = anatomy_summary(pred.get("anatomy"))
    p_sk = pred.get("skeleton") or {}
    p_cut = bool(pred.get("cut_at_token_limit", False))
    p_tears = float(pred.get("tears_pct", 100.0))
    p_score = pred.get("rig_score")
    if p_score is None:
        p_score = rig_score(p_sk, p_tears, pred.get("anatomy"))

    p_clean = p_an["pass"] and not p_cut
    b_clean = b_an["pass"]

    # 1. Reader-independent anatomy & token-limit gate decides first.
    if p_clean and not b_clean:
        return {
            "which": "predicted",
            "best_predicted": pred,
            "rig_score": p_score,
            "trusted": p_score >= TRUSTED_AT,
            "reason": f"predicted passes anatomy while body fails ({', '.join(b_an['codes']) or 'unclean'})",
        }
    if b_clean and not p_clean:
        why = "cut at token limit" if p_cut else (", ".join(p_an["codes"]) or "fails anatomy")
        return {
            "which": "body",
            "best_predicted": pred,
            "rig_score": b_score,
            "trusted": b_score >= TRUSTED_AT,
            "reason": f"body passes anatomy while predicted fails ({why})",
        }

    # 2. Both pass (or both fail) anatomy: compare articulated limb pairs
    #    (unless predicted exploded into cloth chains: > body pairs + 2).
    if p_an["present"] and b_an["present"]:
        pp, bp = p_an["articulated_pairs"], b_an["articulated_pairs"]
        if pp > bp and pp <= bp + 2:
            return {
                "which": "predicted",
                "best_predicted": pred,
                "rig_score": p_score,
                "trusted": p_clean and p_score >= TRUSTED_AT,
                "reason": f"predicted articulates {pp} limb pairs vs body {bp}",
            }
        if bp > pp:
            return {
                "which": "body",
                "best_predicted": pred,
                "rig_score": b_score,
                "trusted": b_clean and b_score >= TRUSTED_AT,
                "reason": f"body articulates {bp} limb pairs vs predicted {pp}",
            }

    # 3. Same anatomical coverage: rescue with body only if predicted tears
    #    substantially more or drops below the hand-made line by a real margin.
    if (p_tears - b_tears) > tear_override_pct or (
        p_score < TRUSTED_AT and (b_score - p_score) >= TRUST_MARGIN
    ):
        return {
            "which": "body",
            "best_predicted": pred,
            "rig_score": b_score,
            "trusted": b_clean and b_score >= TRUSTED_AT,
            "reason": f"body tears less ({b_tears}% vs {p_tears}%, score {b_score} vs {p_score})",
        }

    return {
        "which": "predicted",
        "best_predicted": pred,
        "rig_score": p_score,
        "trusted": p_clean and p_score >= TRUSTED_AT,
        "reason": "predicted passes anatomy with full limb pairs (learned-rig base)",
    }
