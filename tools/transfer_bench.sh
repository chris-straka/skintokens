#!/usr/bin/env bash
# Rig route for a set of characters, one skin transfer at a time, scored by
# weightforge: rig -> motionforge standardize -> weights check (raw) ->
# weights fix -> weights check (final). Same steps and settings as
# motionforge's tools/tripo_rig.py for a human (class humanoid, no clips).
#
#   tools/transfer_bench.sh IN_ROOT OUT_ROOT TAG [SKINTOKENS_ARGS...]
#
# IN_ROOT/<name>/mesh.glb are the cleaned meshes; results go to
# OUT_ROOT/TAG/<name>/s<seed>.* Environment: SK (skintokens binary), WF
# (weights), MF (motionforge), SEEDS (default "1 2 3"), NAMES (default:
# every folder in IN_ROOT), PRED_CACHE (a folder: the model's prediction
# per name and seed is kept there, so two transfers see the same samples
# and the same per-sample skin; the model runs once).
set -u
in_root=$1; out_root=$2; tag=$3; shift 3
SK=${SK:-$HOME/rigwork/bin/skintokens}
WF=${WF:-$HOME/rigwork/bin/weights}
MF=${MF:-$HOME/rigwork/bin/motionforge}
SEEDS=${SEEDS:-"1 2 3"}
NAMES=${NAMES:-$(cd "$in_root" && ls -d */ | tr -d /)}
for n in $NAMES; do
  o=$out_root/$tag/$n
  mkdir -p "$o"
  for s in $SEEDS; do
    b=$o/s$s
    if [ -n "${PRED_CACHE:-}" ]; then
      mkdir -p "$PRED_CACHE"
      export SKINTOKENS_PRED_CACHE=$PRED_CACHE/$n.s$s.pred
    fi
    "$SK" rig "$in_root/$n/mesh.glb" "$b.rig.glb" --class humanoid --seed "$s" --report "$b.skintokens.json" "$@" \
      > "$b.rig.log" 2>&1 || { echo "$n s$s: rig failed"; tail -3 "$b.rig.log"; continue; }
    "$MF" standardize --input "$b.rig.glb" --output "$b.std.glb" --class humanoid --report "$b.standardize.json" \
      > "$b.std.log" 2>&1 || { echo "$n s$s: standardize failed"; continue; }
    "$WF" check "$b.std.glb" --json --vertex-json "$b.raw.verts.json" > "$b.raw.json" 2> "$b.raw.log"
    "$WF" fix "$b.std.glb" --out "$b.glb" > "$b.fix.log" 2>&1
    [ -f "$b.glb" ] || cp "$b.std.glb" "$b.glb"
    "$WF" check "$b.glb" --json --vertex-json "$b.verts.json" > "$b.json" 2> "$b.check.log"
    python3 - "$b" <<'EOF'
import json, sys
b = sys.argv[1]
def s(p):
    try:
        d = json.load(open(p))
    except Exception as e:
        return f"? ({e})"
    f = [x for x in d.get("findings", []) if x.get("severity") == "fail"]
    return f"{d.get('score')} ({len(f)} failing)"
print(f"{b}: raw {s(b + '.raw.json')} -> fixed {s(b + '.json')}", flush=True)
EOF
  done
done
