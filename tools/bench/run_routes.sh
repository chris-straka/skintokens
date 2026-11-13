#!/bin/bash
# Both rig routes on each bench animal's bare mesh, one animal at a time
# (the rigger wants the whole GPU).
#
#   BENCH=/path/to/bench tools/bench/run_routes.sh [names...]
#
# BENCH holds animals/<name>/bare.glb (from prep_truth.py). Each animal
# gets new/ (tools/creature_route.py of this checkout) and old/ (the
# predicted-skeleton route: motionforge's tripo_rig.py as it was before
# the creature route, $OLD_ROUTE). Tools come from $TOOLS/bin
# (creature-rig, skintokens or its wrapper, motionforge, weights,
# rfcheck). A route that already wrote its report is not run again
# (REDO=1 runs it anyway). ONLY=new or ONLY=old runs one route.
# NEW_DIR names the new route's folder (default new): a later version of
# the route goes beside the earlier one, so both can be scored.
set -uo pipefail
BENCH=${BENCH:?set BENCH}
TOOLS=${TOOLS:-/home/f/rigwork/creatures}
OLD_ROUTE=${OLD_ROUTE:-$TOOLS/src/motionforge/tools/tripo_rig.py}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
export CREATURE_RIG_BIN=${CREATURE_RIG_BIN:-$TOOLS/bin/creature-rig} SKINTOKENS_BIN=${SKINTOKENS_BIN:-$TOOLS/bin/sk_wrap.sh} \
  MOTIONFORGE_BIN=$TOOLS/bin/motionforge WEIGHTS_BIN=$TOOLS/bin/weights RFCHECK_BIN=$TOOLS/bin/rfcheck \
  BLENDER=$(which blender) TRIPO_TO_GLB=$TOOLS/src/motionforge/tools/tripo_to_glb.py
names=("$@")
[ ${#names[@]} -eq 0 ] && names=($(ls "$BENCH/animals"))
for n in "${names[@]}"; do
  d=$BENCH/animals/$n
  [ -f "$d/bare.glb" ] || { echo "$n: no bare.glb"; continue; }
  N=${NEW_DIR:-new}
  if [ "${ONLY:-new}" = new ] && { [ -n "${REDO:-}" ] || ! grep -qs best_seed "$d/$N/report.json"; }; then
    t0=$(date +%s)
    python3 "$REPO/tools/creature_route.py" "$d/bare.glb" "$d/$N" --name "$n" --convention none --keep-tries > "$d/$N.log" 2>&1
    echo "$N $n: exit $? in $(( $(date +%s) - t0 )) s: $(tail -1 "$d/$N.log" | cut -c1-260)"
  fi
  if [ "${ONLY:-old}" = old ] && { [ -n "${REDO:-}" ] || ! grep -qs best_seed "$d/old/report.json"; }; then
    t0=$(date +%s)
    python3 "$OLD_ROUTE" "$d/bare.glb" "$d/old" --name "$n" --kind creature --seeds 1,2,3 --convention none --keep-tries > "$d/old.log" 2>&1
    echo "old $n: exit $? in $(( $(date +%s) - t0 )) s: $(tail -1 "$d/old.log" | cut -c1-260)"
  fi
done
