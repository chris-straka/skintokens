#!/usr/bin/env bash
# Rig route for a set of characters, one skin transfer at a time, scored by
# meshweights: rig -> motionforge standardize -> weights check (raw) ->
# weights fix -> weights check (final). Same steps and settings as
# motionforge's tools/tripo_rig.py for a human (class humanoid, no clips).
#
#   tools/transfer_bench.sh IN_ROOT OUT_ROOT TAG [SKINTOKENS_ARGS...]
#
# IN_ROOT/<name>/mesh.glb are the cleaned meshes; results go to
# OUT_ROOT/TAG/<name>/s<seed>.* Environment: SKINTOKENS_BIN or SK (skintokens
# binary), WEIGHTS_BIN or WF (weights), MOTIONFORGE_BIN or MF (motionforge),
# SEEDS (default "1 2 3"), NAMES (default: every folder in IN_ROOT),
# PRED_CACHE (a folder: the model's prediction per name and seed is kept
# there, so two transfers see the same samples and the same per-sample skin;
# the model runs once).
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
SK=${SK:-${SKINTOKENS_BIN:-$REPO/target/release/skintokens}}
WF=${WF:-${WEIGHTS_BIN:-$REPO/../meshweights/rust/target/release/weights}}
MF=${MF:-${MOTIONFORGE_BIN:-$REPO/../motion/rust/target/release/motionforge}}
[ -f "$SK" ] || { echo "skintokens not found: $SK (set SKINTOKENS_BIN or SK)" >&2; exit 2; }
[ -f "$WF" ] || { echo "weights not found: $WF (set WEIGHTS_BIN or WF)" >&2; exit 2; }
[ -f "$MF" ] || { echo "motionforge not found: $MF (set MOTIONFORGE_BIN or MF)" >&2; exit 2; }
in_root=$1; out_root=$2; tag=$3; shift 3
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
        with open(p) as fh:
            d = json.load(fh)
    except Exception as e:
        return f"? ({e})"
    f = [x for x in d.get("findings", []) if x.get("severity") == "fail"]
    return f"{d.get('score')} ({len(f)} failing)"
print(f"{b}: raw {s(b + '.raw.json')} -> fixed {s(b + '.json')}", flush=True)
EOF
  done
done
