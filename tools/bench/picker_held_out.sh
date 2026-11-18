#!/bin/bash
# The picker on hand-made animals it was never fitted on: prepare each animal's truth and bare
# mesh, run the creature route on the bare mesh (both rigs kept), and score the picker against
# the artist's rig, beside the 12 it was fitted on.
#   tools/bench/picker_held_out.sh BENCH LIST
# LIST: `name <tab> source file` lines (the machine's own, off the repo). BENCH is a folder of
# its own: animals/<name>/{truth,bare}.glb, animals/<name>/new2/ (the route, every try kept),
# animals.tsv, picker.json (tools/bench/score_artists.py --picker-check) and baseline.json
# (how often the artist's winner is the learned rig: what a picker that always keeps it scores).
# Tools from the environment (motion's tools/creature_deps.sh sets them up): BLENDER,
# CREATURE_RIG_BIN, CREATURE_BENCH_BIN, SKINTOKENS_BIN, MOTIONFORGE_BIN, WEIGHTS_BIN, RFCHECK_BIN.
set -uo pipefail
[ $# -eq 2 ] || { echo "usage: picker_held_out.sh BENCH LIST" >&2; exit 2; }
B=$1; list=$2
here=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$B/animals"
: > "$B/animals.tsv"
while IFS=$'\t' read -r n src; do
  [ -n "$n" ] || continue
  d=$B/animals/$n
  if [ ! -f "$d/bare.glb" ]; then
    "$BLENDER" -b --factory-startup -P "$here/tools/bench/prep_truth.py" -- "$src" "$d" --name "$n" > "$d.prep.log" 2>&1 \
      || { echo "$n: prep failed: $(tail -2 "$d.prep.log")"; continue; }
  fi
  printf '%s\t%s\t%s\n' "$n" "held out" "$(basename "$src")" >> "$B/animals.tsv"
  if ! grep -qs best_seed "$d/new2/report.json"; then
    t0=$(date +%s)
    python3 "$here/tools/creature_route.py" "$d/bare.glb" "$d/new2" --name "$n" --convention none --keep-tries > "$d/new2.log" 2>&1
    echo "$n: route exit $? in $(( $(date +%s) - t0 )) s: $(tail -1 "$d/new2.log" | cut -c1-220)"
  fi
done < "$list"
python3 "$here/tools/bench/score_artists.py" "$B" --picker-check --json "$B/picker.json"
echo "score_artists exit $?"
python3 - "$B/picker.json" > "$B/baseline.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])).get("picker", {})
n = len(p)
agree = sum(1 for v in p.values() if v["agrees"])
learned = sum(1 for v in p.values() if v["truth_winner"] == "predicted")
print(json.dumps({"animals": n, "picker_agrees": agree, "always_learned_rig_agrees": learned,
                  "per_animal": {k: {"truth": v["truth_winner"], "picked": v["picked"], "agrees": v["agrees"],
                                     "reason": v["reason"]} for k, v in p.items()}}, indent=1))
PY
cat "$B/baseline.json"
if [ -n "${FLEET_ARTIFACTS:-}" ]; then
  cp "$B/picker.json" "$B/baseline.json" "$B/animals.tsv" "$FLEET_ARTIFACTS/" 2>/dev/null
fi
