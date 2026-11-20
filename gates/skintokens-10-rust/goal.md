# skintokens-10 (Rust half): NaN-safe `total_cmp` sorts, zero model drift

Repo: skintokens (`~/Games/_blender/skintokens`). One goal, this repo only. Rust only.

## Why

The Python half of `skintokens-10` already landed (`69422da`). The Rust crates
`crates/skintokens/src/` (`mesh.rs`, `naming.rs`, `geom.rs`, `generate.rs`) and
`crates/creature/src/` (`skeleton.rs`, `roles.rs`, `solid.rs`, `symmetry.rs`,
`weights.rs`, `parts.rs`) still have ~36 `partial_cmp(...).unwrap()` calls on
mesh/geometry/score floats, which panic on a NaN from a malformed GLB. Now that
`skintokens-11`'s `model_pin` is landed on `main`, we can move every one of those
sorts to `f64::total_cmp` / `f32::total_cmp` and prove zero drift on the real
neural model checkpoint.

## What to build

1. Replace every `partial_cmp` call in `crates/skintokens/src/` and
   `crates/creature/src/` (EXCEPT the guarded `crates/creature/src/check.rs`,
   which must NOT be edited!) with `total_cmp` (preserving exact ascending vs
   descending order and tie-breakers like `.then(...)`).
   - Note on tuple sorts such as `out.sort_by(|a, b| a.partial_cmp(b).unwrap())`
     where `a: &(f64, usize)`: write `a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))`
     so ordering is total and deterministic without `partial_cmp`.
   - Note on `geom.rs` `KdTree::new`: `(hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b]))`
     and `pts[a][axis].total_cmp(&pts[b][axis])`.
2. Ensure the new tests in `crates/skintokens/tests/nan_safe.rs` pass (do not
   edit any test file).
3. Keep every existing test and `skintokens-11`'s `model_pin` passing with zero
   drift.

## Evidence

`bash gates/skintokens-10-rust/gate.sh` (through fleet with
`--data=skintokens-weights`, 8 cores) prints `GATE PASS`. It checks:
- `grep -rn 'partial_cmp' crates/skintokens/src crates/creature/src | grep -v '^crates/creature/src/check\.rs:'`
  prints nothing;
- `tools/gate/carbon.sh` passes (guarded paths including `crates/bench/`,
  `crates/creature/src/check.rs`, `tools/pick_score.py`, `tools/bench/`, and
  all tests untouched; Python tests and full Rust workspace release tests
  including `nan_safe` green);
- `cargo test --release -p skintokens --test model_pin -- --ignored model_pin`
  (with `--no-default-features` on Linux) passes against `$FLEET_DATA_SKINTOKENS_WEIGHTS`,
  proving zero drift on the real model checkpoint.

## Out of scope; don't

- Do NOT edit `crates/creature/src/check.rs`, `crates/bench/`, `tools/`,
  `gates/`, or any test file (`crates/*/tests/`).
- Do NOT change any model math, threshold, or tie-breaking direction.
- Do NOT build or run tests on this Mac; use the check script through fleet.
