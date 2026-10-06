# SkinTokens vs UniRig vs the game rig (2026-10-05)

The numbers below this first section were measured with the Python
wrapper (PyTorch on MPS), which the Rust port has since replaced; the
first section is how the port was proven equivalent.

## Rust port: how parity was proven (2026-10-05)

Reference: the Python wrapper at commit `76d7311` (upstream `273b691d` +
the Mac patch, official checkpoint). `parity/dump_ref.py` (at that
commit) recorded upstream's intermediates on a fixed input, both in fp32
on the CPU and as production runs (bf16 autocast on MPS); the Rust side
(`skintokens parity-net`, `parity-input`) ran each network piece on the
same inputs and the same sampled indices. Rust: candle 0.11, Metal, f32.

Deterministic pieces, Andras 13k (rig mode; skin mode on the game rig
gave the same picture), Rust vs Python fp32:

| Piece | max abs diff | rel | cosine |
|---|---|---|---|
| Shape encoder latents (512x512); FPS picks 512/512 identical | 3.4e-4 | 6e-5 | 0.9999999 |
| Mesh condition after output_proj (512x896) | 2.6e-4 | 6e-5 | 1.0000000 |
| VAE condition latents (384x512) | 1.1e-5 | 3e-5 | 1.0000000 |
| LM logits, 43-token fixed prefix (33036 vocab); argmax 43/43 (skin: 144/144) | 2.3e-5 | 1.2e-6 | 0.9999999 |
| VAE decoder weights for fixed tokens (54000 points x 28 joints) | 5.1e-6 | 5e-6 | 1.0000000 |
| Greedy decoding, 208 tokens (skin: 215) | identical | | |
| 10-beam search without sampling, 215 tokens (skin: 215) | identical | | |

For scale: Python against itself, bf16-MPS vs fp32, differs by 6e-2
(rel) in the encoder and up to 3.9 in the logits. Input pipeline vs
Blender + trimesh: same 14,731 faces (2 duplicates dropped like
Blender), normals within 2e-4, normalized vertices within 7e-7. In
skin mode 11-18 prompt tokens out of 85-103 land one bin apart: those
joints sit exactly on a token-bin edge (the game rig came from UniRig
tokens) and only bit-exact Blender float32 math rounds them the same;
running Rust with Python's exact prompt gave the same scores.

Upstream behaviours found and kept: the predict sampler uses 54,000
surface points only (its vertex-sample count is never passed); the
model-wide `.to(bfloat16)` also rounds the VAE embedder's frequencies and
Qwen's rotary `inv_freq`; the vocabulary switch forces the end token one
skin token early, so the last joint's fourth code is decoded from the
end-token id; Blender's exporter drops influences <= 1e-4.

End to end, weightforge 0.1.0 + motionforge standardize (current, with
the 4 twist helpers; 26 bones), same day, same chain:

| | Python (seeds) | Rust (seeds) |
|---|---|---|
| Andras 13k `rig`, raw | 48.1 / 49.6 / 46.9 / 48.9 | 48.9 / 49.9 / 51.5 / 49.7 |
| same, after `weights fix` | 61.0 / 64.9 / 64.6 / 62.7 | 65.1 / 65.3 / 60.6 / 60.8 |
| Rehearsal mesh (5,766 verts) `rig`, raw | 48.7 / 44.4 / 50.0 | 47.9 / 49.2 / 52.4 |
| same, after `weights fix` (genforge chain: 68.7) | 69.3 / 68.4 / 69.9 | 68.9 / 69.1 / 66.9 |
| genforge `--mode rehearsal` on Andras (rig -> standardize -> check -> fix) | 48.2 -> 68.7, 3 failing regions | 47.9 -> 68.9, the same 3 regions and vertex counts |
| `skin` on the standardized game rig, raw (8 seeds) | 43.5-45.8, mean 44.3 | 44.4-47.1, mean 45.8 |
| same as weightforge candidate: candidate score | 71.1-72.4 | 72.2-73.3 |
| same: `weights fix` result | 44.8-56.2, mean 52.2 | 44.6-47.2, mean 45.9 |

The last row is weightforge, not SkinTokens: its greedy region pick is
not monotonic in candidate quality. Rust's skin weights score higher on
their own and as a candidate, the per-joint weight mass matches Python
(Python-vs-Rust per-vertex difference 0.042 = Python seed-to-seed 0.041),
and degrading a Rust candidate with noise (raw 46.6 -> 38.3) raised the
fix result from 46.8 to 49.5. Better candidates get locked in early by
the smallest-edit-first rule and `optimize` then starts from that mix.

Fixed in weightforge the same day (exact region pick, weightforge
`docs/pick-and-bands.md`). Same 16 skins as candidates on the same rig:
Python now fixes to 46.4-58.3 (mean 49.9), Rust to 52.3-52.5 (mean 52.5).
Noise no longer raises the fix (amplitude 0.5 and up: 47.3 and below,
against 52.5 clean), and no candidate leaves the fix below the
no-candidate 46.4. On this 16-failing-region rig the search runs out of
budget, so a 0.25-noise copy still fixed better (58.5).

Speed and memory on the M4 (16 GB), same session: `rig` Andras 13k
31.5-32.7 s (Python 47.8-49.5 s), `skin` 20.5-21 s (Python 34-36 s);
peak RSS 3.7 GB (Python 3.0 GB; Rust keeps f32 weights,
`SKINTOKENS_DTYPE=bf16`: 2.4 GB, 18.9 s, same scores). Same seed gives
the same file (the Python wrapper did not). Joint docs (`skintokens
joints`) are identical to the Python ones (max diff 7e-16).

Sheets (Python left, Rust right): `~/Downloads/skintokens-rust/`.

## SkinTokens vs UniRig (Python wrapper era)

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
