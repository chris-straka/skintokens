# skintokens-6 — Per-limb skeleton check and subtle planted rigs

Implement the per-limb skeleton scoring and subtle planted-rig checks described in `crates/bench/tests/subtle_plant.rs` and `crates/creature/tests/per_limb_check.rs`, touching **only** these three files:
- `crates/bench/src/main.rs`
- `crates/bench/src/anatomy.rs`
- `crates/creature/src/check.rs`

Do not modify any tests, `gates/`, `tools/`, or any other files.

## Requirements

### 1. Subtle planted rig modes in `crates/bench/src/main.rs` (`plant`)
Extend `creature-bench plant IN.glb OUT.glb [--mode diagonal|mirror|falloff] [--json OUT.json]`:
- Default `--mode` is `diagonal` (preserving existing diagonal hip-to-opposite-foot swap behavior). Reject any unknown `--mode` value with a clear error (`bail!`).
- Always include `"mode": mode` in the JSON report written by `--json` (and update `HELP` text to document `--mode diagonal|mirror|falloff`).
- `--mode mirror`: plain left/right mirror swap across `x0` — pair each `+x` weighted limb joint (`pos[i][0] - x0 > tol && rig.mass[i] > 1e-6`) with its **nearest** `-x` weighted limb joint by mirrored distance `norm(pos[i] - [2*x0 - pos[j][0], pos[j][1], pos[j][2]])` (greedy ascending distance), swapping their skin weights in `JOINTS_0`. Report `"mode": "mirror"`, `"swapped_joint_pairs"`, `"swapped_vertices"`, `"pairs"`.
- `--mode falloff`: one-limb-falloff subtle plant — find the side-limb branches hanging off the centre line (subtrees of joints with `|pos[j][0] - x0| > tol` whose parent is a centre-line joint, with weighted vertices). Pick one `+x` side limb deterministically (preferring a limb with >= 2 weighted joints, or largest weighted joint count, tie-breaking by lowest root joint index). Keep **all bones in the skeleton untouched**, and reassign the skin weights of its limb joints onto its centre-line parent joint (or joint 0 if none):
  - If the limb has >= 2 weighted joints, clear **all distal joints** (all weighted joints in the limb subtree after its root joint) onto the centre-line parent, and reduce/reassign most of the limb root's vertices (or all distal joints so the limb clearly loses its distal deformation and majority skin mass onto the trunk).
  - If the limb has only 1 weighted joint, reassign its vertices onto the centre-line parent.
  - Report `"mode": "falloff"`, `"limb_root"`, `"target_joint"`, `"cleared_joints"`, `"reassigned_vertices"`.
- Replace any `partial_cmp(...).unwrap()` in `crates/bench/src/main.rs` with `total_cmp(...)`.

### 2. Per-limb skin-mass and distal-collapse checks in `crates/bench/src/anatomy.rs`
- Track each limb's total skin mass (`skin_of(&below(r)).1`) and emit `"skin_mass": r3(l.mass)` in each entry of `report["limbs"]`.
- Fix the zero-length paired-limb loophole: when two limbs are paired (`la`, `lb`), if one limb collapsed to `length <= 1e-9` (1 live joint) while its partner has `length > 0.05 * len`, `ratio` must be `f64::INFINITY` (not `1.0`) so it fails `B_LOPSIDED`.
- Add a paired-limb skin-mass falloff check (when `!lopsided_body`): for each paired `(la, lb)`, if `la.mass.max(lb.mass) >= 5.0` and `la.mass.min(lb.mass) / la.mass.max(lb.mass) < 0.35`, emit a failing finding with code `"B_FALLOFF"` naming the limbs and their skin masses.
- Replace all `partial_cmp(...).unwrap()` in `crates/bench/src/anatomy.rs` with `total_cmp(...)`.

### 3. Per-limb skeleton scoring in `crates/creature/src/check.rs`
- In `report["limbs"]`, include per-limb fields alongside existing `limb`, `bones`, `covered`, `body_share`:
  - `"role"`: lowercase role name (`"leg"`, `"arm"`, `"wing"`, `"tail"`, `"head"`, `"extra"`),
  - `"score"`: `(s * 100.0).round() / 100.0`,
  - `"pass"`: `bool` — `true` iff this limb produced no `"fail"` finding (`S_NO_CHAIN` / `S_SHORT_CHAIN` with severity `"fail"`).
- Add top-level `"min_limb_score"` to `report`: the minimum `s` (rounded to 2 decimals) across all non-minor, non-`Role::Extra` limbs (or `1.0` if there are none).
- In the left/right symmetry check (lines ~585-614): keep healthy artist rigs with 2-vs-3 well-covered bones passing (`na.abs_diff(nb_) <= 1` AND both limbs' per-limb scores agree within `0.35` and neither side has `0` deforming bones when the other has `> 0`), while failing `S_LOPSIDED` (and not incrementing `pairs.0`) whenever `na.abs_diff(nb_) > 1`, OR one side has `0` bones while the other has `> 0`, OR the paired limbs' per-limb scores diverge by `> 0.35` (catching a 1-vs-2 broken falloff limb!).
- Replace all `partial_cmp(...).unwrap()` in `crates/creature/src/check.rs` with `total_cmp(...)`.

## Verification
Run `./check` (which executes `gates/skintokens-6/gate.sh` through fleet). Every existing Rust and Python test plus all 6 new integration tests in `subtle_plant.rs` and `per_limb_check.rs` must pass.
