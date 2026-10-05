# SkinTokens vs UniRig vs the game rig (2026-10-05)

Question: should SkinTokens replace UniRig (`unirig-mac`) as the
pipeline's ML rigger, and does it unblock genforge's skin-weight gate
(weightforge fails every Andras rig: 22 -> 37 after fix, rigforge rerig
17.7)? Verdict: **yes, replace; no, it does not unblock the gate alone.**

## Setup

- SkinTokens: official code `273b691d` + `patches/apple-silicon.patch`,
  official bf16 checkpoints (HF `79736cad`), PyTorch 2.14.1 on MPS
  (M4, 16 GB). 35-62 s per character, ~4 GB.
- UniRig: `unirig-mac` at `59f9f18` (venv rebuilt today), its
  `tools/eval_harness.py`, seed 12345, CPU. ~3-8 min per character.
- Meshes (read from `games/hll`, never written):
  - **Andras rehearsal**: genforge rehearsal run A's repair-topology
    output (5,766 verts), the exact mesh behind the 22 / 37.2 / 17.7
    numbers; rig stripped for the ML runs.
  - **Andras 13k**: `andras_rig.glb`'s mesh (13,610 verts).
  - **Andras casual**: the Tripo casual body (`andras_casual.blend`,
    `Andras_Body_0`, 97,456 verts; byte-identical to the mesh UniRig's
    bake-off used).
  - **Stalker**: `stalker.glb` with its primitives joined into one mesh.
- Every humanoid rig went through the same path: structural Mixamo naming
  (`skintokens/naming.py`; UniRig emits `bone_N` too), motionforge
  `adapter standardize` to the HLL 22-bone `DEF-*` skeleton, then
  weightforge 0.1.0 `weights check` (31 humanoid ROM poses) and
  `weights fix` (auto). Creatures: standardize `--class quadruped`,
  generic per-bone poses.
- "Game rig" = what the rehearsal fed as the rig: `andras_rig.glb`, the
  engine bake-off skeleton (itself UniRig output, hand-mapped) with its
  weights transferred onto the retopo mesh. **No Tripo-rigged Andras
  exists on this machine**, so "Tripo's own weights" could not be scored;
  the game rig is the rehearsal's candidate 0.

## Andras (weightforge score /100, higher is better; pass = no failing finding)

| Rig | Rehearsal mesh raw | after `weights fix` | 13k mesh raw | after fix |
|---|---|---|---|---|
| Game rig (UniRig bake-off weights) | 22.0 (22 fails) | 37.2 (12) | 17.0 (29) | 37.4 (10) |
| UniRig, fresh run, same mesh | 10.1 (37) | 33.6 (9) | 16.5 (32) | 35.0 (10) |
| rigforge rerig (rehearsal) | 17.7 (16) | 17.7 | - | - |
| **SkinTokens rig** (seed 0) | **47.9 (6)** | **61.6 (6)** | **46.7 (7)** | **62.8 (4)** |
| SkinTokens, 3 more runs (s1, s2, s0 again) | 44.2 / 46.2 / 47.2 | 62.0 / 58.8 / 61.4 | | |
| SkinTokens skin-only on the HLL skeleton | 40.2 (6) | 51.1 (7) | 43.1 (6) | 51.8 (5) |
| Game rig + SkinTokens skin-only as `--candidate` | | 48.6 (6) | | 49.8 (4) |
| Game rig + SkinTokens rig as `--candidate` | | 49.8 (5) | | |

Casual Tripo body (97k verts, the mesh UniRig's bake-off used):
SkinTokens 50.9 raw (14 fails; weightforge fix leaves it unchanged).
UniRig's own triage routed it to rigforge (root collapse predicted, stem
6.75 < 7.0), so it produced no skin at seed 12345.

End to end through the wrapper: `skintokens rig --class humanoid` on the
rehearsal mesh (48.8 s) -> standardize -> 46.2; and `weights fix
--skintokens` on the game rig: 22.0 -> 49.8 in 34 s (37.2 without it),
with `external:skintokens` chosen for some regions.

Reading:

- SkinTokens' own rig roughly doubles the raw score of UniRig and the
  game rig, and after weightforge's fix lands at ~61 vs ~35. Its raw
  output already beats every other rig *after* fixing.
- Zero `D_BLEED` findings in any SkinTokens run (game rig 7-8, UniRig
  8-10): no hand-to-spine or foot-to-thigh weights. Knees and elbows at
  140 deg no longer tear (see `cmp_unirig_vs_skintokens.png`).
- What still fails, in every SkinTokens run: `D_STRETCH` / `D_VOLUME` on
  the upper arms and thighs in `arm_up`, `arm_forward` and `hip_forward`
  (90 deg+ swings), 3-22 verts per region. That is the armpit/groin
  limit of linear blend skinning on this mesh, not weight noise; it needs
  twist/helper bones or corrective shapes, or a gate tolerance for those
  poses. So SkinTokens does not unblock the gate by itself.
- Run-to-run spread is small (44-48 raw, 59-62 fixed). Same-seed reruns
  are not bit-identical on MPS (joints within ~2 cm).
- Skin-only (SkinTokens weights on the existing HLL skeleton) scores
  lower than SkinTokens' own skeleton (40-43 vs 46-48): the model skins
  its own joints best. As a weightforge candidate it lifts the game rig's
  fix from 37 to ~49.

## Skeletons

Joint heads after standardizing (22 HLL joints, cm):

| | vs SkinTokens mean / max |
|---|---|
| UniRig (fresh) | 2.0 / 4.8 (thighs) |
| Game rig | 2.4 / 5.3 (forearms) |
| rigforge rerig | 10.6 / 22.9 (its shoulders and elbows sit 20+ cm low on this mesh) |

- Bone count: SkinTokens 28 raw (spine x4, neck, head, shoulder-arm-
  forearm-hand + 3 hand segments per side, thigh-shin-foot-toe), same
  topology as UniRig; 30 on a second run. No fingers, no face, no twists,
  like UniRig.
- genforge's standardize adapter maps it to the HLL skeleton once
  `skintokens rig --class humanoid` names the joints (28 -> 22 `DEF-*`,
  the 6 hand segments folded into the hands). Raw `bone_N` output does
  not map. TokenRig's VRoid class token (`--cls vroid`) did not help: the
  model still emitted unnamed joints and an odd extra branch at the hips.
- See `skel_andras.png`: game rig, UniRig and SkinTokens agree within a
  few cm; rigforge's rerig is the outlier on this retopo mesh.

## Stalker (blockout of rigid primitives)

| Rig | raw | after fix |
|---|---|---|
| UniRig (fresh) | routed to rigforge by its own triage (root collapse predicted, stem 4.10), no skin | - |
| UniRig (saved 2026-09-30 result) | 1.6 (17 fails) | 77.6 (3) |
| SkinTokens | 31.9 (4) | 32.0 (3) |

Inconclusive: a box body with cylinder legs cannot bend, scores depend on
which generic poses each skeleton gets, and both models give one-bone
legs. rigforge's measured `hll_stalker` rig stays the creature path. As
rigforge hints, SkinTokens' 11 stalker joints: 0 trusted, 11
measurement-wins (UniRig's sample: 1 trusted, 1 supporting, 8
measurement-wins); the hinted fit still gates strict.

## Decision

SkinTokens replaces UniRig: better weights by a wide margin on every
humanoid mesh, 5-10x faster, one model instead of two stages, same
licenses (MIT code and weights, Michelangelo-derived encoder). It
becomes weightforge's ML candidate (`weights fix --skintokens`) and
rigforge's hint source (`skintokens joints`); `unirig-mac` is retired,
not deleted.

Open for the owner: whether genforge should use `skintokens rig` as the
free rig stage (instead of buying Tripo's auto-rig) or as a repair-rig
step before rigforge; and how to clear the remaining shoulder/hip
stretch (helper bones in the HLL skeleton, or a gate tolerance for the
90 deg+ ROM poses).

Sheets: `~/Downloads/skintokens-eval/`. Raw files: the session scratch
(not kept).
