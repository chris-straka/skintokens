#!/bin/bash
# Does the predicted-skeleton route read the body or remember the model?
# Each named animal is stretched (taller, shorter) and rigged again by the
# old route; `creature_bench.py` on $BENCH/stretched then gives its joint
# error against the stretched artist's joints.
#
#   BENCH=/path/to/bench tools/bench/leak_probe.sh fox horse cow trex bat
set -uo pipefail
BENCH=${BENCH:?set BENCH}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$BENCH/stretched/animals"
: > "$BENCH/stretched/animals.tsv"
for n in "$@"; do
  python3 "$HERE/stretch.py" "$BENCH/animals/$n" "$BENCH/stretched/animals/$n" --scale "${STRETCH:-1.0,1.35,0.8}"
  cp "$BENCH/animals/$n/truth.json" "$BENCH/stretched/animals/$n/" 2>/dev/null
  grep -P "^$n\t" "$BENCH/animals.tsv" >> "$BENCH/stretched/animals.tsv"
done
BENCH=$BENCH/stretched ONLY=${ONLY:-old} bash "$HERE/run_routes.sh" "$@"
