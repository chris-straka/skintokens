#!/bin/bash
# Builds creature-bench (release) from this checkout and puts it in $BENCH/bin.
set -euo pipefail
BENCH=${BENCH:?set BENCH}
export PATH=$HOME/.cargo/bin:$PATH
cd "$(dirname "$0")/../.."
CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$BENCH/target} cargo build --release -p creature-bench 2>&1 | grep -E "^(warning|error)|Finished" -A6 | head -40
mkdir -p "$BENCH/bin"
cp "${CARGO_TARGET_DIR:-$BENCH/target}/release/creature-bench" "$BENCH/bin/creature-bench.new"
mv "$BENCH/bin/creature-bench.new" "$BENCH/bin/creature-bench"
echo "built $BENCH/bin/creature-bench"
