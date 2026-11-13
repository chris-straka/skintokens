#!/bin/bash
# The no-artist skeleton check over a set of rigs.
#
#   tools/bench/anatomy_set.sh OUT_DIR TAG 'PATTERN with %n for the name' names...
#
# Writes OUT_DIR/TAG/<name>.json and prints one line a rig.
set -uo pipefail
OUT=$1; TAG=$2; PAT=$3; shift 3
CB=${CREATURE_BENCH_BIN:-creature-bench}
mkdir -p "$OUT/$TAG"
for n in "$@"; do
  rig=${PAT//%n/$n}
  [ -f "$rig" ] || { echo "$TAG $n: no rig at $rig"; continue; }
  "$CB" anatomy "$rig" --json "$OUT/$TAG/$n.json" | sed "s#^.*: stands#$TAG $n: stands#" | cut -c1-420
done
