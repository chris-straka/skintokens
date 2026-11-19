#!/usr/bin/env bash
# skintokens' gate for work the goal runner hands to Carbon (skintokens-4). Every item gate in
# this repo runs it first, then its own checks.
#   bash tools/gate/carbon.sh               the scope, then the Rust and Python suites
#   bash tools/gate/carbon.sh --scope-only  the scope only (tools/gate/planted.sh uses this)
# The worker's changes are its own commits ("carbon <slug>: round N", after GATE_BASE, default
# the commit that last changed tools/gate/) and anything uncommitted. They must not touch what
# judges the work:
#   tools/pick_score.py, tools/bench/ (the benches, the held-out manifest), crates/bench/,
#   crates/creature/src/check.rs, tools/gate/, any test file (tools/test_*.py, crates/*/tests/);
#   and no #[test] may disappear and no #[ignore] appear anywhere in crates/.
# Then: cargo test --release --workspace --no-fail-fast (Linux: --no-default-features, the
# default `metal` feature is the Mac's), and python3 -m unittest discover -s tools -p "test_*.py".
# Prints PASS/FAIL per check, ends with GATE PASS or GATE FAIL; exit 0 only on a pass.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
FAILS=0
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; FAILS=$((FAILS + 1)); }
show() { sed 's/^/      /' | head -n "${2:-20}"; }
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
BASE=${GATE_BASE:-$(git log -1 --format=%H -- tools/gate)}
[ -n "$BASE" ] || { fail "no base commit (GATE_BASE, or tools/gate/ committed)"; BASE=HEAD; }

echo "== scope (skintokens' guarded paths)"
mine=$(for c in $(git log --format=%H --grep '^carbon [^ ]*: round' "$BASE..HEAD"); do git diff-tree --no-commit-id --name-only -r "$c"; done)
touched=$( (echo "$mine"; git status --porcelain | cut -c4-) | grep -v '^$' | sort -u)
guarded=$(echo "$touched" | grep -E '^(tools/pick_score\.py|tools/bench/|crates/bench/|crates/creature/src/check\.rs|tools/gate/|tools/test_[^/]*\.py|crates/[^/]+/tests/)')
if [ -z "$guarded" ]; then pass "no guarded path touched ($(echo "$touched" | grep -c .) files changed)"; else fail "guarded paths touched:"; echo "$guarded" | show; fi
nt=$(git grep -c '#\[test\]' -- crates | awk -F: '{s+=$NF} END {print s+0}')
nb=$(git grep -c '#\[test\]' "$BASE" -- crates | awk -F: '{s+=$NF} END {print s+0}')
wt=$(grep -rc '#\[test\]' crates --include='*.rs' | awk -F: '{s+=$NF} END {print s+0}')
ni=$(grep -rc '#\[ignore' crates --include='*.rs' | awk -F: '{s+=$NF} END {print s+0}')
nib=$(git grep -c '#\[ignore' "$BASE" -- crates | awk -F: '{s+=$NF} END {print s+0}')
if [ "$wt" -ge "$nb" ] && [ "$nt" -ge "$nb" ] && [ "$ni" -le "$nib" ]; then pass "Rust tests $wt (base $nb), ignored $ni (base $nib)"; else fail "Rust tests $wt (base $nb), ignored $ni (base $nib)"; fi
if [ "${1:-}" = --scope-only ]; then
  if [ "$FAILS" -eq 0 ]; then echo "GATE PASS"; exit 0; else echo "GATE FAIL ($FAILS)"; exit 1; fi
fi

echo "== Python unit tests"
python3 -m unittest discover -s tools -p "test_*.py" > "$T/py.out" 2>&1
pc=$?
pn=$(grep -E '^Ran [0-9]+ test' "$T/py.out" | awk '{print $2}')
if [ $pc -eq 0 ]; then pass "python3 -m unittest: ${pn:-?} tests OK"; else fail "python3 -m unittest: exit $pc"; grep -E '^(FAIL|ERROR):|Error' "$T/py.out" | show; fi

echo "== Rust workspace"
command -v cargo >/dev/null 2>&1 || { fail "no cargo on this machine"; echo "GATE FAIL ($FAILS)"; exit 1; }
feat=()
[ "$(uname -s)" = Darwin ] || feat=(--no-default-features)
cargo test --release --workspace --no-fail-fast ${feat[@]+"${feat[@]}"} > "$T/rs.out" 2>&1
rc=$?
rn=$(grep -E '^test result: ok' "$T/rs.out" | awk '{s+=$4} END {print s+0}')
rf=$(grep -cE '^test result: FAILED' "$T/rs.out")
if [ $rc -eq 0 ] && [ "$rf" -eq 0 ] && [ "$rn" -gt 0 ]; then pass "cargo test --release --workspace ${feat[*]:-}: $rn passed"
else fail "cargo test --release --workspace ${feat[*]:-}: exit $rc, $rn passed, $rf targets failed"; grep -E '^(error|test .* FAILED|    [a-z_:]+$)' "$T/rs.out" | show; tail -5 "$T/rs.out" | show 5; fi

if [ "$FAILS" -eq 0 ]; then echo "GATE PASS"; exit 0; else echo "GATE FAIL ($FAILS)"; exit 1; fi
