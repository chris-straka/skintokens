#!/bin/bash
# The creature route over a set of generated models, one at a time (the
# rigger wants the whole GPU), then the no-artist skeleton check on each.
#
#   BENCH=/path/to/bench tools/bench/route_set.sh TAG SET.tsv
#
# SET.tsv: name <tab> kind <tab> model file (FBX or GLB), one a line.
# Writes $BENCH/owner/TAG/<name>/ (the route's output and report),
# <name>.anatomy.json, and $BENCH/owner/TAG.txt: one line a creature.
# A creature already done is not run again (REDO=1 runs it anyway).
# Tools as in run_routes.sh; CREATURE_RIG_BIN picks the body reader.
set -uo pipefail
BENCH=${BENCH:?set BENCH}; TAG=$1; SET=$2
TOOLS=${TOOLS:-/home/f/rigwork/creatures}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export CREATURE_RIG_BIN=${CREATURE_RIG_BIN:-$BENCH/bin/creature-rig} SKINTOKENS_BIN=${SKINTOKENS_BIN:-$TOOLS/bin/sk_wrap.sh} \
  MOTIONFORGE_BIN=$TOOLS/bin/motionforge WEIGHTS_BIN=$TOOLS/bin/weights RFCHECK_BIN=$TOOLS/bin/rfcheck \
  BLENDER=$(which blender) TRIPO_TO_GLB=$TOOLS/src/motionforge/tools/tripo_to_glb.py
CB=${CREATURE_BENCH_BIN:-$BENCH/bin/creature-bench}
O=$BENCH/owner/$TAG; mkdir -p "$O"
while IFS=$'\t' read -r n kind model; do
  [ -z "$n" ] && continue
  d=$O/$n
  if [ -n "${REDO:-}" ] || ! grep -qs best_seed "$d/report.json"; then
    t0=$(date +%s)
    python3 "$REPO/tools/creature_route.py" "$model" "$d" --name "$n" --keep-tries > "$d.log" 2>&1
    echo "$n: route exit $? in $(( $(date +%s) - t0 )) s"
  fi
  [ -f "$d/$n.glb" ] || { echo "$n: no rig"; continue; }
  "$CB" anatomy "$d/$n.glb" --json "$d/$n.anatomy.json" > /dev/null
  python3 - "$d" "$n" "$kind" <<'PY' | tee -a "$O.txt"
import json, sys
d, n, kind = sys.argv[1:4]
r = json.load(open(f"{d}/report.json")); t = r["tries"][0]
best = t["predicted"] if t.get("picked") == "predicted" else t
a = json.load(open(f"{d}/{n}.anatomy.json"))
fails = [f["code"] for f in a["findings"] if f["severity"] == "fail"]
print("\t".join(str(x) for x in [n, kind, t.get("body"), t.get("picked"), ", ".join(f"{k} {v}" for k, v in (t.get("limbs") or {}).items()),
      best.get("score"), len(best.get("fails", [])), best["skeleton"]["score"], "passes" if not fails else " ".join(fails)]))
PY
done < "$SET"
