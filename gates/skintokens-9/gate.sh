#!/usr/bin/env bash
# Gate for skintokens-9 (game-tools docs/plans/skintokens.md): paths that resolve to today's
# folders.
#  - tools/gate/carbon.sh passes (the worker leaves the judges alone; Python and Rust suites);
#  - `grep -rn "weightforge\|rigwork\|/home/f" tools` prints nothing;
#  - creature_bench.py and creature_follow.py fall back to the sibling
#    meshweights/rust/target/release/weights (the folder's name since 2026-11-17);
#  - transfer_bench.sh finds its tools from the environment or the sibling repos' own builds;
#  - every changed Python file compiles and every changed shell script parses.
# Builds the Rust workspace (through fleet, about 8 cores). Prints PASS/FAIL per check, ends
# with GATE PASS or GATE FAIL.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
SLUG=skintokens-9
FAILS=0
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; FAILS=$((FAILS + 1)); }
show() { sed 's/^/      /' | head -n "${2:-20}"; }
BASE=$(git log -1 --format=%H -- gates/$SLUG)
[ -n "$BASE" ] || { fail "cannot find the commit that added the gate"; BASE=HEAD; }

echo "== paths"
hits=$(grep -rn "weightforge\|rigwork\|/home/f" tools | grep -v "/__pycache__/")
if [ -z "$hits" ]; then pass "no weightforge, rigwork or /home/f in tools/"; else fail "$(echo "$hits" | wc -l | tr -d ' ') old paths left:"; echo "$hits" | show; fi
for f in tools/creature_bench.py tools/creature_follow.py; do
  grep -q "meshweights/rust/target/release/weights" "$f" && pass "$f falls back to the sibling meshweights build" || fail "$f has no meshweights/rust/target/release/weights fallback"
done
if grep -qE '^(SK|WF|MF)=\$\{(SK|WF|MF):-\$HOME' tools/transfer_bench.sh; then fail "transfer_bench.sh still defaults to a home-folder build"; else pass "transfer_bench.sh takes no home-folder build by default"; fi
mine=$(for c in $(git log --format=%H --grep "^carbon $SLUG: " "$BASE..HEAD"); do git diff-tree --no-commit-id --name-only -r "$c"; done)
changed=$( (echo "$mine"; git status --porcelain | cut -c4-) | grep -v '^$' | sort -u)
bad=0
for f in $(echo "$changed" | grep -E '\.py$'); do [ -f "$f" ] && { python3 -m py_compile "$f" 2>/dev/null || { fail "$f does not compile"; bad=1; }; }; done
for f in $(echo "$changed" | grep -E '\.sh$'); do [ -f "$f" ] && { bash -n "$f" 2>/dev/null || { fail "$f does not parse"; bad=1; }; }; done
[ $bad -eq 0 ] && pass "changed files compile and parse ($(echo "$changed" | grep -c .) changed)"
other=$(echo "$changed" | grep -v -E '^(tools/|README\.md$|docs/)')
[ -z "$other" ] && pass "only tools/, README.md, docs/ changed" || { fail "other files changed:"; echo "$other" | show; }

echo "== skintokens' gate"
GATE_BASE=$BASE bash tools/gate/carbon.sh
[ $? -eq 0 ] || FAILS=$((FAILS + 1))

if [ "$FAILS" -eq 0 ]; then echo "GATE PASS"; exit 0; else echo "GATE FAIL ($FAILS)"; exit 1; fi
