# skintokens-9: paths that resolve to today's folders

Repo: skintokens (`~/Games/_blender/skintokens`). One goal, this repo only.

## Why

On 2026-11-17 the tool folders were renamed (weightforge is now `meshweights`, motionforge's
folder is `motion`). Some fallbacks still name the old folders, so they never resolve, and
some name one machine's home folder (`~/rigwork/...`), so an old build there can be picked
silently. An old binary silently picked is worse than a missing one. The bench scripts under
`tools/bench/` are already fixed (by the lane, commit 370a28a); what is left is yours.

## What to do

`grep -rn "weightforge\|rigwork\|/home/f" tools` lists what is left today:

1. `tools/creature_bench.py:250` and `tools/creature_follow.py:97`: the `weights` fallback
   becomes the sibling `meshweights/rust/target/release/weights` (as `tools/creature_route.py`
   already does: read its `find` calls). Their doc lines (`:19`) say meshweights, not
   weightforge.
2. `tools/creature_route.py:24`: the doc line says meshweights.
3. `tools/transfer_bench.sh`: line 3's comment says meshweights; `SK`, `WF`, `MF` (lines
   17-19) no longer default to `$HOME/rigwork/bin/...`: take them from the environment
   (`SKINTOKENS_BIN`, `WEIGHTS_BIN`, `MOTIONFORGE_BIN`, keeping `SK`/`WF`/`MF` as overrides),
   else the sibling repos' own release builds (`$REPO/target/release/skintokens`,
   `../meshweights/rust/target/release/weights`, `../motion/rust/target/release/motionforge`),
   and stop with a clear message naming the variable when the file is not there.
4. Nothing else changes.

## Evidence

`bash gates/skintokens-9/gate.sh` (through fleet; it builds the Rust workspace) prints
`GATE PASS`. It checks the grep above prints nothing, the two fallbacks, transfer_bench.sh's
defaults, that changed files compile and parse, that only `tools/`, README.md and `docs/`
changed, and then runs `tools/gate/carbon.sh` (read it: the guarded paths, the Python tests,
`cargo test --release --workspace`).

## Out of scope; don't

- Don't edit `tools/pick_score.py`, `tools/bench/`, `crates/bench/`,
  `crates/creature/src/check.rs`, `tools/gate/`, `gates/` or any test: the gate fails on it.
- Don't change what any tool does beyond where it looks for its programs.
- Don't start Blender or build anything on this Mac.
