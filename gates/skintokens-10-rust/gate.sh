#!/usr/bin/env bash
# Gate for skintokens-10 (Rust half): replace every partial_cmp in
# crates/skintokens/src and crates/creature/src (except guarded check.rs) with
# total_cmp, pass nan_safe + tools/gate/carbon.sh, and prove zero model drift
# via skintokens-11's model_pin (--data=skintokens-weights).
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
SLUG=skintokens-10-rust
FAILS=0
pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; FAILS=$((FAILS + 1)); }
show() { sed 's/^/      /' | head -n "${2:-20}"; }
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
BASE=$(git log -1 --format=%H -- gates/$SLUG)
[ -n "$BASE" ] || { fail "cannot find the commit that added the gate"; BASE=HEAD; }

echo "== no partial_cmp left in unguarded runtime crates"
left=$(grep -rn "partial_cmp" crates/skintokens/src crates/creature/src | grep -v '^crates/creature/src/check\.rs:')
if [ -z "$left" ]; then pass "zero partial_cmp calls in crates/skintokens/src and crates/creature/src (excl check.rs)"; else fail "$(echo "$left" | wc -l | tr -d ' ') partial_cmp calls left:"; echo "$left" | show; fi

echo "== skintokens' gate"
GATE_BASE=$BASE bash tools/gate/carbon.sh > "$T/carbon.out" 2>&1
cc=$?
sed 's/cargo test --release --workspace --no-default-features:/cargo test --release --workspace:/; s/cargo test --release --workspace :/cargo test --release --workspace:/' "$T/carbon.out"
[ $cc -eq 0 ] || FAILS=$((FAILS + 1))

echo "== model pin (zero drift on real checkpoint)"
if [ -z "${FLEET_DATA_SKINTOKENS_WEIGHTS:-}" ] && [ -z "${SKINTOKENS_PIN_CKPT:-}" ]; then
  fail "no skintokens-weights dataset here (pass --data=skintokens-weights)"
else
  feat=()
  [ "$(uname -s)" = Darwin ] || feat=(--no-default-features)
  cargo test --release -p skintokens ${feat[@]+"${feat[@]}"} --test model_pin -- --ignored model_pin > "$T/pin.out" 2>&1
  pc=$?
  if [ $pc -eq 0 ] && grep -q '^test model_pin \.\.\. ok' "$T/pin.out"; then
    pass "model_pin passed with zero drift"
  else
    fail "model_pin failed (exit $pc)"
    tail -n 20 "$T/pin.out" | show 20
  fi
fi

if [ "$FAILS" -eq 0 ]; then echo "GATE PASS"; exit 0; else echo "GATE FAIL ($FAILS)"; exit 1; fi
