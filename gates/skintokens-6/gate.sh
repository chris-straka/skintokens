#!/usr/bin/env bash
# Gate for skintokens-6 (code + synthetic tests): per-limb skeleton check in
# crates/creature/src/check.rs, subtle planted rigs (--mode mirror|falloff) in
# crates/bench/src/main.rs, and per-limb falloff/skin-mass anatomy checks in
# crates/bench/src/anatomy.rs.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
SLUG=skintokens-6
FAILS=0
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; FAILS=$((FAILS + 1)); }
show() { sed 's/^/      /' | head -n "${2:-20}"; }
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
BASE=$(git log -1 --format=%H -- gates/$SLUG)
[ -n "$BASE" ] || { fail "cannot find the commit that added the gate"; BASE=HEAD; }

echo "== scope (only check.rs, bench/src/main.rs, bench/src/anatomy.rs)"
mine=$(for c in $(git log --format=%H --grep "^carbon $SLUG: round" "$BASE..HEAD"); do git diff-tree --no-commit-id --name-only -r "$c"; done)
touched=$( (echo "$mine"; git status --porcelain | cut -c4-) | grep -v '^$' | sort -u)
bad=$(echo "$touched" | grep -vE '^(crates/bench/src/main\.rs|crates/bench/src/anatomy\.rs|crates/creature/src/check\.rs)$' | grep -v '^$')
if [ -z "$bad" ]; then
  pass "scope clean ($(echo "$touched" | grep -c .) files changed)"
else
  fail "out-of-scope files touched:"
  echo "$bad" | show
fi

nt=$(git grep -c '#\[test\]' -- crates | awk -F: '{s+=$NF} END {print s+0}')
nb=$(git grep -c '#\[test\]' "$BASE" -- crates | awk -F: '{s+=$NF} END {print s+0}')
wt=$(grep -rc '#\[test\]' crates --include='*.rs' | awk -F: '{s+=$NF} END {print s+0}')
ni=$(grep -rc '#\[ignore' crates --include='*.rs' | awk -F: '{s+=$NF} END {print s+0}')
nib=$(git grep -c '#\[ignore' "$BASE" -- crates | awk -F: '{s+=$NF} END {print s+0}')
if [ "$wt" -ge "$nb" ] && [ "$nt" -ge "$nb" ] && [ "$ni" -le "$nib" ]; then
  pass "Rust tests $wt (base $nb), ignored $ni (base $nib)"
else
  fail "Rust tests $wt (base $nb), ignored $ni (base $nib)"
fi

echo "== no partial_cmp left in touched runtime files"
left=$(grep -rn "partial_cmp" crates/bench/src/main.rs crates/bench/src/anatomy.rs crates/creature/src/check.rs)
if [ -z "$left" ]; then
  pass "zero partial_cmp in bench/main.rs, bench/anatomy.rs, creature/check.rs"
else
  fail "$(echo "$left" | wc -l | tr -d ' ') partial_cmp calls left:"
  echo "$left" | show
fi

echo "== Python unit tests"
python3 -m unittest discover -s tools -p "test_*.py" > "$T/py.out" 2>&1
pc=$?
pn=$(grep -E '^Ran [0-9]+ test' "$T/py.out" | awk '{print $2}')
if [ $pc -eq 0 ]; then
  pass "python3 -m unittest: ${pn:-?} tests OK"
else
  fail "python3 -m unittest: exit $pc"
  grep -E '^(FAIL|ERROR):|Error' "$T/py.out" | show
fi

echo "== Rust workspace"
command -v cargo >/dev/null 2>&1 || { fail "no cargo on this machine"; echo "GATE FAIL ($FAILS)"; exit 1; }
feat=()
[ "$(uname -s)" = Darwin ] || feat=(--no-default-features)
cargo test --release --workspace --no-fail-fast ${feat[@]+"${feat[@]}"} > "$T/rs.out" 2>&1
rc=$?
rn=$(grep -E '^test result: ok' "$T/rs.out" | awk '{s+=$4} END {print s+0}')
rf=$(grep -cE '^test result: FAILED' "$T/rs.out")
if [ $rc -eq 0 ] && [ "$rf" -eq 0 ] && [ "$rn" -gt 0 ]; then
  pass "cargo test --release --workspace: $rn passed"
else
  fail "cargo test --release --workspace: exit $rc, $rn passed, $rf targets failed"
  grep -E '^(error|test .* FAILED|    [a-z_:]+$)' "$T/rs.out" | show
  tail -5 "$T/rs.out" | show 5
fi

echo "== skintokens-6 integration tests ran and passed"
for t in \
  clean_quadruped_and_biped_pass_anatomy_with_skin_mass \
  plant_diagonal_and_mirror_modes_are_distinct_and_fail_anatomy \
  plant_falloff_keeps_bones_and_fails_anatomy_on_quadruped_and_biped_arm \
  clean_box_dog_reports_per_limb_scores_and_passes \
  one_limb_falloff_drops_per_limb_score_and_fails_symmetry \
  healthy_three_vs_two_bone_leg_pair_still_passes_symmetry
do
  if grep -q "^test $t \.\.\. ok" "$T/rs.out"; then
    pass "test $t"
  else
    fail "test $t did not pass"
  fi
done

if [ "$FAILS" -eq 0 ]; then echo "GATE PASS"; exit 0; else echo "GATE FAIL ($FAILS)"; exit 1; fi
