#!/usr/bin/env bash
# Aggregate-only gate for the frozen `skintokens-1` held-out animal set.
#
#   BENCH_HELDOUT=/home/f/rigwork/bench_heldout tools/bench/heldout.sh [--run-route] [--json OUT.json]
#
# Verifies every held-out animal's bare.glb and truth.glb against the committed
# `tools/bench/heldout_manifest.json`, optionally runs `creature_route.py --keep-tries`
# on missing animals, scores the picker and the route once, and prints ONLY
# aggregate picker agreement and route medians (never per-animal breakdowns).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
MANIFEST="$HERE/heldout_manifest.json"
BENCH="${BENCH_HELDOUT:-${FLEET_DATA_SKINTOKENS_HELDOUT:-}}"
RUN_ROUTE=0
OUT_JSON=""
TAG="new2"

while [ $# -gt 0 ]; do
  case "$1" in
    --bench) BENCH="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --route-tag) TAG="$2"; shift 2 ;;
    --run-route) RUN_ROUTE=1; shift ;;
    --json) OUT_JSON="$2"; shift 2 ;;
    *) echo "usage: heldout.sh [--bench DIR] [--run-route] [--json OUT.json]" >&2; exit 2 ;;
  esac
done

if [ -z "$BENCH" ] || [ ! -d "$BENCH" ]; then
  echo "heldout.sh: set BENCH_HELDOUT (or pass --bench DIR) to the frozen held-out directory" >&2
  exit 2
fi

# 1. Verify frozen manifest & SHA-256 checksums before touching anything.
python3 "$HERE/verify_heldout.py" --manifest "$MANIFEST" --bench "$BENCH"

# 2. Optionally run the creature route on any held-out animal missing <tag>/report.json.
if [ "$RUN_ROUTE" -eq 1 ]; then
  IDS="$(python3 -c 'import json,sys; print("\n".join(a["id"] for a in json.load(open(sys.argv[1]))["animals"]))' "$MANIFEST")"
  while IFS= read -r id; do
    [ -n "$id" ] || continue
    d="$BENCH/animals/$id"
    if [ ! -f "$d/$TAG/report.json" ] || ! grep -qs '"picked"' "$d/$TAG/report.json"; then
      echo "running creature_route on held-out animal (id masked) ..." >&2
      rc=0
      python3 "$REPO/tools/creature_route.py" "$d/bare.glb" "$d/$TAG" \
        --name "$id" --convention none --keep-tries > "$d/$TAG.log" 2>&1 || rc=$?
      if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
        echo "heldout.sh: creature_route crashed with exit $rc (see $d/$TAG.log)" >&2
        exit "$rc"
      fi
      if [ ! -f "$d/$TAG/report.json" ] || ! grep -qs '"picked"' "$d/$TAG/report.json"; then
        echo "heldout.sh: creature_route wrote no picked report (see $d/$TAG.log)" >&2
        exit 2
      fi
    fi
  done <<< "$IDS"
fi

# 3. Score picker + baselines + route medians, printing ONLY aggregate numbers.
WORK_TMP="$(mktemp -d)"
trap 'rm -rf "$WORK_TMP"' EXIT

rc=0
python3 "$HERE/score_artists.py" "$BENCH" --picker-check --json "$WORK_TMP/picker.json" > "$WORK_TMP/score_artists.log" 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
  cat "$WORK_TMP/score_artists.log" >&2 || true
  echo "heldout.sh: score_artists.py failed with exit $rc" >&2
  exit "$rc"
fi

python3 - "$BENCH" "$MANIFEST" "$WORK_TMP/picker.json" "$TAG" "$OUT_JSON" <<'PY'
import json
import os
import shutil
import statistics
import subprocess
import sys

bench, manifest_path, picker_json, tag, out_json = sys.argv[1:6]
manifest = json.load(open(manifest_path))
ids = [a["id"] for a in manifest["animals"]]

cb = os.environ.get("CREATURE_BENCH_BIN") or shutil.which("creature-bench")
wf = os.environ.get("WEIGHTS_BIN") or shutil.which("weights")
if not cb or not os.path.exists(cb):
    print("heldout.sh: missing CREATURE_BENCH_BIN", file=sys.stderr)
    sys.exit(2)

picker_doc = json.load(open(picker_json)) if os.path.isfile(picker_json) else {}
baselines = picker_doc.get("baselines") or picker_doc.get("picker_baselines") or {}
gate_failures = picker_doc.get("gate_failures") or []
cal_summary = {
    "artists_total": len(picker_doc.get("artists") or {}),
    "calibration_gate_failures": len(gate_failures),
}

def score_one(truth, cand, out_p):
    r = subprocess.run([cb, "score", truth, cand, "--json", out_p], capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not os.path.isfile(out_p):
        err = (r.stderr or r.stdout or "").strip()[-300:]
        print(f"heldout.sh: creature-bench score failed (exit {r.returncode}): {err}", file=sys.stderr)
        sys.exit(2 if r.returncode in (0, None) else r.returncode)
    d = json.load(open(out_p))
    return {
        "joints_pct": d["skeleton"]["joint_error_pct"],
        "missing_chains": len(d["skeleton"]["missing_chains"]),
        "skin_mean_pct": d["skin"]["mean_pct"],
        "skin_p95_pct": d["skin"]["p95_pct"],
        "weights_only_pct": (d.get("skin_weights_only") or d.get("weights_only") or {}).get("mean_pct"),
    }

def median_or_none(vals):
    xs = [x for x in vals if x is not None]
    return round(statistics.median(xs), 3) if xs else None

routes = {"route": [], "predicted": [], "body": [], "artist": [], "crude": []}
score_tmp = os.path.join(os.path.dirname(picker_json), "scores")
os.makedirs(score_tmp, exist_ok=True)

for aid in ids:
    adir = os.path.join(bench, "animals", aid)
    truth = os.path.join(adir, "truth.glb")
    rdir = os.path.join(adir, tag)

    # Artist truth (0 control)
    s_art = score_one(truth, truth, os.path.join(score_tmp, f"{aid}.artist.json"))
    if s_art:
        routes["artist"].append(s_art)

    # Crude rigid control
    crude_glb = os.path.join(score_tmp, f"{aid}.crude.glb")
    r_rig = subprocess.run([cb, "rigid", truth, crude_glb], capture_output=True, text=True)
    if r_rig.returncode != 0:
        err = (r_rig.stderr or r_rig.stdout or "").strip()[-300:]
        print(f"heldout.sh: creature-bench rigid failed (exit {r_rig.returncode}): {err}", file=sys.stderr)
        sys.exit(2 if r_rig.returncode in (0, None) else r_rig.returncode)
    s_crude = score_one(truth, crude_glb, os.path.join(score_tmp, f"{aid}.crude.json"))
    routes["crude"].append(s_crude)

    # Final picked route rig
    picked_glb = os.path.join(rdir, f"{aid}.glb")
    if os.path.isfile(picked_glb):
        s_route = score_one(truth, picked_glb, os.path.join(score_tmp, f"{aid}.route.json"))
        if s_route:
            routes["route"].append(s_route)

    # Body-read candidate
    body_glb = os.path.join(rdir, "try.glb")
    if os.path.isfile(body_glb):
        s_body = score_one(truth, body_glb, os.path.join(score_tmp, f"{aid}.body.json"))
        if s_body:
            routes["body"].append(s_body)

    # Best predicted seed (first available try.pred*.glb or chosen by report)
    rep_path = os.path.join(rdir, "report.json")
    pred_glb = None
    if os.path.isfile(rep_path):
        rep = json.load(open(rep_path))
        tries = rep.get("tries") or []
        if tries and isinstance(tries[0].get("predicted"), dict):
            seed = tries[0]["predicted"].get("seed")
            if seed is not None:
                cand = os.path.join(rdir, f"try.pred{seed}.glb")
                if os.path.isfile(cand):
                    pred_glb = cand
    if not pred_glb:
        for k in (1, 2, 3):
            cand = os.path.join(rdir, f"try.pred{k}.glb")
            if os.path.isfile(cand):
                pred_glb = cand
                break
    if pred_glb:
        s_pred = score_one(truth, pred_glb, os.path.join(score_tmp, f"{aid}.pred.json"))
        if s_pred:
            routes["predicted"].append(s_pred)

agg = {
    "manifest_count": len(ids),
    "route_tag": tag,
    "picker_baselines": baselines,
    "calibration_summary": cal_summary,
    "route_medians": {},
}

for rname, items in routes.items():
    agg["route_medians"][rname] = {
        "n": len(items),
        "joints_median_pct": median_or_none([x["joints_pct"] for x in items]),
        "missing_chain_animals": sum(1 for x in items if x["missing_chains"] > 0),
        "skin_mean_median_pct": median_or_none([x["skin_mean_pct"] for x in items]),
        "skin_p95_median_pct": median_or_none([x["skin_p95_pct"] for x in items]),
        "weights_only_median_pct": median_or_none([x["weights_only_pct"] for x in items]),
    }

print("=== skintokens-1 frozen held-out aggregate results (n=%d) ===" % len(ids))
if baselines:
    n = baselines.get("total", len(ids))
    print(
        f"Picker agreement: {baselines.get('picker')} / {n} "
        f"(LOO: {baselines.get('leave_one_out')}/{n}, "
        f"old: {baselines.get('old_score')}/{n}, "
        f"always-predicted: {baselines.get('always_predicted')}/{n}, "
        f"always-body: {baselines.get('always_body')}/{n}, "
        f"coin-flip: {baselines.get('coin_flip_expected', 0):.1f}/{n} expected)"
    )
print("")
print("| Rig | N | Joints med | Missing chain | Skin mean med | Skin p95 med | Weights-only med |")
print("|---|---:|---:|---:|---:|---:|---:|")
for rname in ("artist", "crude", "route", "predicted", "body"):
    m = agg["route_medians"][rname]
    def f(v): return "-" if v is None else f"{v:.2f}%"
    print(
        f"| {rname} | {m['n']} | {f(m['joints_median_pct'])} | "
        f"{m['missing_chain_animals']} of {m['n']} | {f(m['skin_mean_median_pct'])} | "
        f"{f(m['skin_p95_median_pct'])} | {f(m['weights_only_median_pct'])} |"
    )

if out_json:
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(agg, fp, indent=2)
        fp.write("\n")

fleet_art = os.environ.get("FLEET_ARTIFACTS")
if fleet_art and os.path.isdir(fleet_art):
    with open(os.path.join(fleet_art, "heldout_aggregate.json"), "w", encoding="utf-8") as fp:
        json.dump(agg, fp, indent=2)
        fp.write("\n")
PY
