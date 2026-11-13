#!/bin/bash
# The body reading alone, before and after a change to creature-rig: each
# mesh is rigged from its body only (no model, no weight repair) and the
# rig is put through the no-artist check.
#
#   CREATURE_RIG_BIN=... CREATURE_BENCH_BIN=... tools/bench/reread.sh OUT_DIR 'MESH PATTERN with %n' names...
#
# Writes OUT_DIR/<name>.glb, .json (creature-rig's reading), .anatomy.json
# and prints one line a mesh.
set -uo pipefail
OUT=$1; PAT=$2; shift 2
CR=${CREATURE_RIG_BIN:-creature-rig}; CB=${CREATURE_BENCH_BIN:-creature-bench}
mkdir -p "$OUT"
for n in "$@"; do
  mesh=${PAT//%n/$n}
  [ -f "$mesh" ] || { echo "$n: no mesh"; continue; }
  "$CR" rig "$mesh" "$OUT/$n.glb" --json "$OUT/$n.json" --sheet "$OUT/$n.png" > /dev/null 2> "$OUT/$n.err" || { echo "$n: no rig: $(tail -1 "$OUT/$n.err")"; continue; }
  "$CB" anatomy "$OUT/$n.glb" --json "$OUT/$n.anatomy.json" > /dev/null
  python3 - "$OUT/$n.json" "$OUT/$n.anatomy.json" "$n" <<'PY'
import json, sys
d, a, n = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
fails = [f["code"] for f in a["findings"] if f["severity"] == "fail"]
print(f"{n:20s} {d['kind']}{', upright' if d['upright'] else ''}; turned {d.get('turned_deg', 0):.0f}; " + ", ".join(f"{l['name']} {len(l['bones'])}" for l in d["limbs"]) + " | " + ("passes" if not fails else "FAILS " + " ".join(fails)))
PY
done
