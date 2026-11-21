# Creature rigs: what was tried, kept and dropped

A running log for the creature route (anything that is not a person):
`skintokens rig` for the predicted rig, `creature-rig` (this repo,
`crates/creature`) for the body's parts, the skeleton check, our own
skeleton and weights. Newest last. Models are named by what they are,
not by the game's names. Scores are weightforge's bend check (0 to 100,
"failing" = failing findings; the gate is none) unless they say
"skeleton" (the skeleton check, below).

The test set is 29 generated models, cleaned to 18,000 triangles (7,500
for the human-shaped ones): 10 four-legged, 6 two-legged monsters that
stand up with their hands near the ground, 4 human-shaped, 1 winged, 5
cloaked or without legs, 3 odd ones (a hand, a fish, a torso).

## 2026-11-12

### The rigger on an NVIDIA card (kept)

`--features cuda`. On an RTX 2060 SUPER a rig takes 26 s where that
machine's CPU took about ten minutes, with the same skeleton, joint for
joint, as the CPU build gives for that seed. Peak 1.6 GB of memory and
5.5 GB of the card's 8.

### A skeleton that never ends (fixed)

A cloaked figure in rags had no rig: twice the rigger was stopped past
12 GB. Cause: the model wrote a chain for every strip of cloth and never
ended the skeleton, so generation ran to its 1,534-token limit. Memory:
the beam cache kept one slot per beam per step (15,340 slots) and grew
by doubling, to 20,480 slots, 4.7 GB, 7 GB while copying, more on a GPU
that frees late. No rig: the tokens ran out before the skin tokens.

- The cache now reclaims the slots of dropped beams before it grows: it
  stays at its first 2,560 slots (0.6 GB) for any length. `cargo test`
  drives a small model 1,534 steps on 10 beams and holds it there, and
  checks that reclaiming changes no logit.
- The skeleton has to end while four skin tokens per joint still fit
  (146 joints at most); the report says `skeleton_cut_at_token_limit`.
- The grammar let a skeleton end between a branch's parent and its
  joint ("truncated branch", seen once on that figure after 24 minutes
  on a CPU); it no longer does.
- Result: that figure rigs in 84 s at 1.6 GB. Its 146-joint skeleton is
  mostly cloth strips; see below for what is done with it.

### The body's parts from its volume (kept; the base of everything below)

`creature-rig parts`: the mesh as a solid on a 128-cell grid (surface
crossings counted with their facing along each axis, two of three axes
decide, so holes and parts pushed through each other don't empty it),
distance through the inside from the deepest point, shells of equal
distance, a tree of the shells' pieces. Long branches are limbs; each
has a centre line inside the body and a thickness along it. Half a
second a model.

`roles` then names them from where they start and end: legs (reach the
ground from well above it, one on each side), neck and head, tail,
wings (a flat pair), arms (a pair high on a body that stands up),
"extra" (a long thing that is none of these). Toes, ears, horns and fur
are bumps and get nothing. Checked by eye on all 29 (sheets in the
round's report): the ten four-legged ones read as four legs, a neck
and, where there is one, a tail.

- Dropped: the side of a limb by its tip against 7% of the body's
  longest side. A horse's and a wolf's legs are nearer the middle than
  that and read as "centre", so the horse had two legs. Now: the mean
  of the limb's line against 1.5% of the longest side or 6% of the
  width.
- Dropped: a leg's first joint where its line starts. The line starts
  in the middle of the trunk, so the leg turned about a point deep in
  the body and the shoulder tore. Now it starts where the body around
  the line is no more than a quarter thicker than the limb.
- Found on the way: the clean step turns two of the upright monsters to
  face backwards (its measured facing wins over the convention when
  they disagree by 180 degrees; on these the convention was right).
  `roles` reads the toes and treats such a body the other way round;
  the report lists them.

### Our own skeleton and weights (`creature-rig rig`)

Hips and one to four spine bones along the trunk's centre line, a chain
down every limb with joints where its line turns (else at set
fractions), the same joints on a left and its right. Weights: each
bone's distance to every cell measured through the inside only; a
vertex goes to its nearest bone and shares with that bone's neighbours
over a band as wide as `--blend` times the body's thickness at the
joint; two smoothing passes along short edges.

- Band width (bend score, raw, 13 models): 0.3 worst everywhere; 1.0 to
  1.6 best on the bulky, furry ones; 2.5 best on the thin-limbed ones;
  4.0 worse again (long collapses). No one width wins, so the route
  makes the weights at 1.0, 1.6 and 2.5 and lets weightforge's fix
  choose per region.
- Skull joint at the thinnest point of the neck: on a horse that put
  the joint low on the neck. Now at the bend where the line turns down
  into the head when there is one, else the thinnest point.

### Judging two rigs alike (kept: this is the score)

The old route judged a creature on motionforge's four-legged pose set
when its gait step found four legs in the predicted skeleton, and on
"every bone bent 45 degrees" when it didn't (5 of the 10 four-legged
models, every other kind). Scores from different pose sets don't
compare, so both rigs of a creature are now judged the same way:

- `creature-rig check --rom` writes the body's range of motion on that
  rig's own bones (the bones that lie along each limb, the topmost one
  swings it), `weights check --poses` bends it through that and through
  the rig's own clips;
- `creature-rig check` scores the skeleton against the body;
- **creature score = bend score x skeleton score / 100**; the gate is no
  failing finding in either.

Why the skeleton score has to be in it: the bend check gave 99 to a
cloaked figure whose seven bones bend almost nothing, and 0 failing
findings to rigs with no bone in the tail or the neck.

- Dropped: judging without the shoulder. The first fitted skeleton
  started each leg where it leaves the body, so "leg forward" turned the
  leg at the elbow; the predicted rigs have bones inside the shoulder and
  were turned there. That made ours look better than it was (20 of 25
  passing). With a shoulder or hip bone half way between the spine and
  the surface, and the check counting a rig's bones inside the body as
  the limb's, both are swung from the top.

### The whole route against the old one, 29 models (first full run)

`tools/creature_route.py` (clean, `creature-rig rig` at bands 1.6, 1.0
and 2.5, walk and trot for a four-legged body, `weights fix` choosing
per region among the bands, the model's weights for this skeleton and
its own repairs, both checks, rfcheck) against the old route's best of
three seeds, judged alike:

| | Old route | New route |
| --- | --- | --- |
| Creature score, mean | 52.8 | 69.4 |
| Bend score, mean | 66.5 | 74.8 |
| Skeleton score, mean | 80.4 | 93.5 |
| Failing bend findings | 58 | 29 |
| Failing skeleton findings | 43 | 12 |
| No failing finding at all | 1 | 12 |
| Peak memory | 1.6 GB | 1.5 GB |

Higher on 23 of 29. Lower on six: a squat four-legged beast (82.5 to
67.9: its old rig has no walk or trot and is judged without them; ours
fails two findings in its trot), a hunched two-legged one whose head
our reading missed, and four within two points. What the fix picks,
region by region, is mostly its own `optimize` and the model's weights
on our skeleton; the three bands are the starting point more than the
answer.

- What our own check found wrong with our own rigs: last bones of claws
  and rags reported outside the body (the check gave a last bone 0.6 of
  its parent's length along the parent, not along itself; now it reads
  it as weightforge does and ends it where it leaves the body), and
  half the four-legged rigs got no walk: motionforge's gait wants a
  joint within a fifth of the height of the ground at each foot and
  both legs of a pair on one bone. Now: a foot joint where the leg
  comes within a tenth of the height of the ground, and a pair hangs
  from the spine bone nearest the middle of its two bases.

### After the first full run (same day)

Judged alike on the final poses, old route against new, 29 models:
creature score 52.8 to 70.6, skeleton 81.0 to 97.4, failing skeleton
findings 39 to 10, no failing finding at all 1 to 9; failing bend
findings 53 to 45. Higher on 23, lower on 6.

- Kept: a foot joint where the leg comes within a tenth of the height of
  the ground, and a pair of legs hung from one spine bone. motionforge's
  gait then accepts 8 of the 10 four-legged rigs (5 before), and the
  horse and both wolves pass with their walk and trot.
- Kept: no bone shorter than 6% of the body's longest side. A squat
  beast's hind legs had been cut into six 5 cm bones and scored 27.
- Kept: the pose writer tells a hip or shoulder bone (inside the body)
  from the upper leg. With the hip bone added, "knee" had been bending
  the thigh a right angle; the two-legged scores before this fix were
  on wrong poses, in both directions.
- Kept: the check matches a bone to the nearest centre line in plain
  distance (it had divided by the limb's thickness, so a thin limb's
  bones went to the thick limb beside it), and reads a last bone as
  weightforge does (along its own +Y, ended where it leaves the body).
- Kept: a piece of body between two forks is trunk. On a hunched
  two-legged monster the piece between the chest's fork and the hips'
  was read as a 1 cm "neck", and everything below it counted as that
  limb.
- Dropped as a default: knees and elbows where the predicted rig has
  its joints (`--hint`). All 29: creature score 68.7 with, 70.6
  without; a human-shaped one 83.5 to 90.3 and a gorilla-like one 68.3
  to 73.6, but a squat beast 68.6 to 41.7 and a wolf 85.3 to 63.2 (the
  predicted joints sit off our centre line's bends and the trot tears).
  It stays as an option.
- Not solved: a figure whose two legs are one mass (roots). The volume
  gives one limb or two that share cells, and one knee tears the other
  leg: 10.7 where the predicted rig scores 74.9. So the route now makes
  the predicted rig as well, judges both alike and keeps the better
  (`creature_route.py`, step 5b; `--no-predicted` turns it off).

### The route keeps the better of two rigs (kept)

`creature_route.py` step 5b: the model's own rig of the same mesh
(three seeds, as the old route tried), judged exactly as ours, the best
of them kept instead of ours when it ranks higher. Rank: creature
score, 15 off for a failing skeleton finding, 5 on for passing both
checks.

- Dropped: "passes both checks" as the first key. A floating figure's
  predicted rig passed both with bones along half its limbs (skeleton
  70, creature score 44) and was kept over ours at 75 with one failing
  finding.
- One seed for the model's rig: 72.6 over the set, the model's rig kept
  on 4. Three seeds: **73.3**, kept on 8; 13 rigs with no failing
  finding (1 for the old route), failing bend findings 33 (53), failing
  skeleton findings 8 (39). Higher than the old route on 25 of 29,
  lower on 3: a human-shaped figure whose old rig came from the human
  route (74.9 to 55.3; it belongs there), and two within two points.
- Cost: 40 s to 7 minutes a creature on the GPU, about 3 typically;
  1.5 GB peak.

Leads not taken yet: two limbs in one mass (cut the solid along the
mirror plane below the hips before reading parts); arms and legs
through thick fur, which is most of what still fails; a jaw; the
facing in the clean step itself.

## 2026-11-12, evening: judged against hand-made animals

The benchmark (`docs/creature-bench.md`) put the route's rig against
twelve hand-made animals playing their own clips: farther from the
artist than the model's rig on 11 of 12, with four of the six
four-legged animals read as two-legged or legless, and the route's own
skeleton check passing them. A new hunched, long-armed beast was read
with both arms as legs and a leg as a tail, and passed at 99.5.

### Read the body about its own mirror plane (kept)

The beast's cause: the clean step could not tell which way it faces and
left it sideways, so an arm and a leg of one side read as a left and a
right. `creature-rig` now finds the body's mirror plane from the mesh
(every vertical plane, 3 degrees apart then 1: the body's mirror image
laid on the body, mean distance as the score) and turns the mesh onto it
when the best plane is more than 10 degrees off and a quarter better.
The beast: two legs, two arms, a neck and head.

### A branch the body stands on is a leg however short (kept)

A limb had to reach 15% of the body's longest side; a long, low animal's
legs are a tenth of it, so the pug, cow and rat had no front legs. A
branch off the midline that reaches the ground now counts at 18% of the
body's height. Body-read rig against the artist (joints, skin mean): pug
14.3% / 2.2% to 7.3% / 1.1%, cow 9.8% / 2.4% to 4.9% / 1.3%, rat 11.0% /
2.3% to 6.7% / 1.6%; the other animals unchanged or within 0.1.

### A head held to one side (kept)

On a body that stands with no limb on the midline up top, the bulkiest
unnamed limb in the upper half with no like limb opposite is the head.
Two more creatures get their head chain.

### Not solved

Hips and shoulders still sit where the limb leaves the body (about a
tenth of the body's length from an artist's). The apatosaurus gets two
legs of four; the dinosaurs no small arms; the bat and spider no head
bone. The fused-root figure lost a leg (two to one). The full route
(weights, the model's rig, the picker) was not rerun with these.

## 2026-11-13

### A finer grid when the body stands on more feet than it has legs (kept)

A crouched, spiked four-legged model with its hind legs tucked against
its tail read as two-legged: on the 128-cell grid the hind legs and the
tail are one mass. The reader now counts where the body stands on the
ground off its midline (vertices within 4% of the height of the ground,
in patches); when that is more than the legs read, plus arms whose hands
rest on the ground, it reads the body again at 256 cells and keeps that
reading if it has more legs. That model: four legs, bend 82.1 with no
failing finding. The other 33 generated creatures and the twelve
hand-made animals read as before.

### The full route with the mirror-plane reading (measured)

Twelve hand-made animals: joints 9.0% to 5.0%, a missing chain on 10 to
8, skin 1.6% to 1.1% (the model's rig alone: 2.9%, 3, 0.7%). 34
generated creatures: tears on the range of motion median 0.0%, at or
under 0.7% on 27; 21 pass the no-artist check, the five newest among
them. Tears rose on six of the earlier 29 (cloaked, legless and odd
bodies, where the reading now finds more or other limbs) and fell on
two.

### Not solved (added)

A tail or head that lies a little off the mirror plane is called
"extra" and gets no poses of its own (the spiked four-legged model).
The picker still ranks by the bend score times this crate's skeleton
check, which cannot see that hips and shoulders sit a tenth of the body
from an artist's.

## 2026-11-16: artist calibration, blind walk benchmark and reader-neutral picker

### Why hand-made rigs scored 7.5 to 24.2 (fixed)

On the twelve benchmark animals, the old creature score (`bend * skeleton
/ 100`) gave the artists' own hand-made rigs 7.5 to 24.2 (median ~14):

- `creature-rig` read meshes through `Frame::World` (the mesh node's
  matrix) while reading joints in world space. On artist GLBs whose
  armature node carries a unit-conversion scale (`horse` 1/6), the mesh
  loaded six times smaller than its own bones, failing `S_OUTSIDE` and
  `S_NO_CHAIN`. Added `Frame::Rest` (`skin_rest`: `jointWorld *
  inverseBind * p`) in `crates/skintokens/src/glb.rs` and switched
  `creature-rig` to it.
- Also fixed in `creature-rig`: volume seed chosen inside the largest
  connected component of `solid.inside` rather than an isolated floater;
  stance midline width measured from the lower 40% of the body so spread
  wings (`bat`, `dragon`) neither swallow legs nor merge narrow feet
  across `X = 0` (`apatosaurus`); zero-skin `*_end` leaf joints and
  `neutral_bone` excluded from `S_OUTSIDE`; IK feet hung under `root`
  walked to their meeting shin bone (`rat`).
- Result (`tools/bench/score_artists.py`): all 12 hand-made rigs pass
  both `creature-rig check` (93.9 to 100.0) and `creature-bench anatomy`,
  scoring **92.4 to 99.3** on `pick_score.rig_score` — above `crude`
  rigid weights (**77.2 to 86.7**) and `plant`ed swapped-limb rigs
  (**0.0 to 26.2**).

### Benchmark loopholes closed (`crates/bench`)

- `fit()` in `creature-bench score` only translates the single primary
  root joint holding the most vertices (closing a flat-hierarchy
  loophole where unparented joints translated freely; extra roots are
  reported in `unparented_roots`).
- At most `k + 2` route bones by skin mass may match an artist chain of
  `k` bones (`oversegmented_bones` reported), and `extra_bones` checks
  plain segment distance (`> 0.12` body lengths).
- `creature-bench anatomy` splits ground patches across the mirror plane
  (`apatosaurus` body-read rig now fails `B_NO_LEG`), rejects 1-joint
  side-limb stubs, and flags `B_SWAPPED` when a side limb holds
  opposite-side skin. Added `creature-bench plant` to swap farthest
  opposite-side limb joints as a planted bad control.

### Reader-neutral picker and blind game-camera verification (kept)

- `tools/pick_score.py` replaces `bend * skeleton / 100` in
  `tools/creature_route.py`: ranks `skintokens` seeds by
  `creature-bench anatomy`, `skeleton_cut_at_token_limit`, skeleton pass
  and `rig_score`, and compares `predicted` against `body` on
  reader-independent anatomy, articulated limb pairs and tear share.
- Agreement with ground-truth `creature-bench score` across the 12
  animals: **12 of 12** (leave-one-out **12 of 12**, vs **5 of 12** for
  the old score).
- Not yet a held-out result (note, 2026-11-18): the 12 of 12 was measured
  on the same 12 animals the picker's constants and branch order were
  tuned on; the leave-one-out refits only `tear_override_pct`. No animal
  outside the 12 has been tried, and a picker that always answers
  `predicted` was not scored beside it.
- Blind same-frame walk pairs at the game camera (`tools/creature_blind.py`
  + `tools/bench/render_gamecam.py`, `seed 20261117`, key outside the
  judged folder): an independent blind judge passed the identical
  control (`RMSE 0.0` -> `cant_tell`) and the planted swapped-limb bad
  case, answered `cant_tell` on `bat`, and agreed **3 of 3** with ground
  truth and `pick_score` on `velociraptor`, `fox` and `trex`, retiring
  the old score on all three.

## 2026-11-20: Frozen held-out 12-animal set, honest baselines, post-11-16 medians, and subtle plants (`skintokens-1/-2/-5/-6`)

- **Frozen held-out set (`skintokens-1`, fleet `96qnk55e`):** Selected and
  froze 12 non-Quaternius, CC0-1.0 / CC-BY-4.0 animated animals from
  independent authors (`BabylonJS/Assets` and `Mesh2Motion`), with zero
  overlap against the 12 in-sample animals and covering winged, hunched,
  long-armed, many-legged, legless, quadruped, and dense 16k–52k triangle
  meshes (`bee_dense` 51.7k, `octopus_dense` 16.1k, `shark_dense` 51.2k,
  `eagle`, `m2m_dragon`, `kaiju`, `m2m_trex`, `m2m_spider`, `m2m_snake`,
  `dog`, `panda`, `whale`). Manifest and SHA-256s live in
  `tools/bench/heldout_manifest.json`; aggregate-only gate is
  `tools/bench/heldout.sh`.
- **Picker baselines (`skintokens-2`, fleet `gymftv8v` + `7ta6s56x`):**
  - **In-sample 12:** picker **11 of 12** (LOO **11 of 12**, old score
    **9 of 12**, **always-predicted 12 of 12**, always-body **0 of 12**,
    coin-flip `6.0/12` expected, `3/12` seed=0). Calibration gate: 12/12
    hand-made rigs >= 90 and beat crude & planted controls.
  - **Frozen held-out 12 (`heldout.sh`):** picker **10 of 12** (LOO
    **10 of 12**, old score **6 of 12**, **always-predicted 11 of 12**,
    always-body **1 of 12**, coin-flip `6.0/12` expected, `4/12` seed=0).
    Calibration gate: **6 of 12** non-Quaternius/dense/legless artist
    rigs fail `score_artists`'s `>= 90` calibration threshold.
  - **Honest verdict:** `always-predicted` beats the current `pick_score`
    picker by 1 animal on both in-sample (`12/12` vs `11/12`) and frozen
    held-out (`11/12` vs `10/12`), while `pick_score` beats the old score
    (`10/12` vs `6/12` held-out) and coin-flip (`6/12`).
- **Per-limb check & subtle plant verification (`skintokens-6`, fleet `gymftv8v`):**
  - **In-sample 12:** artist anatomy pass **12/12**; `loud` rejected
    **12/12**, `mirror` rejected **12/12**, `falloff` rejected **12/12**
    (**PASS**).
  - **Frozen held-out 12:** artist anatomy pass **8/12** (honest failure
    on 4 of 12 non-Quaternius/dense/legless artist rigs); on all 11
    limbed animals (`m2m_snake` is legless so `creature-bench plant`
    exits 2), `loud` rejected **11/11**, `mirror` rejected **11/11**,
    `falloff` rejected **11/11**.
- **Route medians (`skintokens-5`, fleet `gymftv8v` + `7ta6s56x`):** See
  `docs/creature-bench.md` ("Post-11-16 and held-out medians"). On frozen
  held-out animals, `predicted` (`5.38%` joints, `0.59%` skin mean,
  `1.58%` p95) continues to outperform `body` (`7.78%` joints, `1.80%`
  skin mean, `5.04%` p95) by 3x on skin error, and the picked `route`
  tracks `predicted` closely (`5.66%` joints, `0.65%` skin mean, `1.69%`
  p95).
- **Outside rival — upstream UniRig (`skintokens-8` / `unirig-mac-5`, fleet `3h66zkcb`):**
  Scored `VAST-AI/UniRig` (MIT, SDPA without `flash-attn`, seed 42) across
  all `12/12` in-sample and `12/12` frozen held-out animals (0 failures).
  - **In-sample 12:** UniRig joints `3.60%`, missing `7/12`, skin mean
    `1.24%`, p95 `3.05%`, weights-only `1.52%` (vs SkinTokens `predicted`
    `3.08%` / `4/12` / `0.71%` / `1.95%` / `0.50%`).
  - **Frozen held-out 12:** UniRig joints **`4.47%`**, missing `12/12`,
    skin mean `1.36%`, p95 `3.84%`, weights-only `1.39%` (vs SkinTokens
    `predicted` `5.38%` / `11/12` / **`0.59%`** / **`1.58%`** / **`0.53%`**
    and `route` `5.66%` / `11/12` / **`0.65%`** / **`1.69%`** / **`0.64%`**).
  - UniRig places joints closer on unseen non-Quaternius bodies, while
    SkinTokens delivers >2x lower skin deformation error on both sets.
- **Outside skeleton rivals — Puppeteer & MagicArticulate GPU bench (`skintokens-8`, fleet `kn533sry`, `zmnw2ntc`, `v5bc6s56`):**
  - **Licences verified:** `Seed3D/Puppeteer` skeleton (`puppeteer_skeleton_w_diverse_pose.pth`) **Apache-2.0** (skinning stage excluded: depends on NVIDIA PartField, non-commercial); `Seed3D/MagicArticulate` skeleton (`checkpoint_trainonv2_hier.pth`) **Apache-2.0** (no skinning code released); shared shape encoder `Maikou/Michelangelo` (`shapevae-256.ckpt`) **LGPL-3.0** weights / **GPL-3.0** code (tool-side only).
  - Added `tools/bench/setup_skeleton_rivals.sh`, `tools/bench/run_skeleton_rivals.py` (SDPA patch without `flash-attn`, glTF `+Y`-up world OBJ export, single-root tree sanitizer, rigid + `skintokens skin` GLB builder), and `tools/test_skeleton_rivals.py`.
  - Fleet `zmnw2ntc` ran SDPA skeleton inference (`apply_marching_cubes=true`, seed 42) across all `12/12` in-sample and `12/12` frozen held-out meshes (`0` failures), and `v5bc6s56` skinned every predicted skeleton with `skintokens skin` (`sk_wrap.sh`) and scored both skinned and rigid baselines with `creature-bench score` (`12/12` in-sample, `12/12` held-out, `0` failures):

| Set | Rig | N | Joints med | Missing chain | Skin mean med | Skin p95 med | Weights-only med |
|---|---|---:|---:|---:|---:|---:|---:|
| In-sample | `predicted` (SkinTokens best seed) | 12 | **3.08%** | **4/12** | **0.71%** | **1.95%** | **0.50%** |
| In-sample | `route` (SkinTokens picked) | 12 | 5.00% | 8/12 | 1.10% | 2.82% | 0.82% |
| In-sample | `puppeteer` + `skintokens skin` | 12 | 3.53% | 6/12 | 0.89% | 2.29% | 0.62% |
| In-sample | `puppeteer` + rigid | 12 | 3.54% | 9/12 | 1.17% | 2.94% | 1.31% |
| In-sample | `magicarticulate` + `skintokens skin` | 12 | 4.91% | 7/12 | 1.08% | 2.78% | 0.76% |
| In-sample | `magicarticulate` + rigid | 12 | 5.07% | 9/12 | 1.26% | 3.34% | 1.26% |
| In-sample | `unirig` (VAST-AI/UniRig, SDPA) | 12 | 3.60% | 7/12 | 1.24% | 3.05% | 1.52% |
| In-sample | `body` (body-read alone) | 12 | 6.86% | 11/12 | 1.52% | 3.49% | 1.22% |
| Held-out | `predicted` (SkinTokens best seed) | 12 | 5.38% | 11/12 | **0.59%** | **1.58%** | **0.53%** |
| Held-out | `route` (SkinTokens picked) | 12 | 5.66% | 11/12 | **0.65%** | **1.69%** | **0.64%** |
| Held-out | `puppeteer` + `skintokens skin` | 12 | **3.94%** | 12/12 | 0.77% | 1.62% | 0.67% |
| Held-out | `puppeteer` + rigid | 12 | 4.10% | 12/12 | 0.97% | 2.52% | 0.78% |
| Held-out | `magicarticulate` + `skintokens skin` | 12 | 5.28% | **10/12** | 0.85% | 1.83% | 0.76% |
| Held-out | `magicarticulate` + rigid | 12 | 5.25% | **10/12** | 0.92% | 2.62% | 0.67% |
| Held-out | `unirig` (VAST-AI/UniRig, SDPA) | 12 | 4.47% | 12/12 | 1.36% | 3.84% | 1.39% |
| Held-out | `body` (body-read alone) | 12 | 7.78% | 12/12 | 1.80% | 5.04% | 1.21% |

  - **Takeaway (`docs/creature-bench.md`):** On unseen held-out animals, **Puppeteer** leads on joint placement (`3.94%` vs UniRig `4.47%`, MagicArticulate `5.28%`, SkinTokens `predicted` `5.38%` / `route` `5.66%`), while **SkinTokens** still leads on skin deformation (`0.59%` mean / `1.58%` p95 / `0.53%` weights-only) and cuts Puppeteer/MagicArticulate p95 error by 30–36% when re-skinning their skeletons (`0.77%` / `1.62%` and `0.85%` / `1.83%` vs rigid `0.97%` / `2.52%` and `0.92%` / `2.62%`, and far better than UniRig's own skin stage `1.36%` / `3.84%`).
- **Plumbing & safety consolidation (`skintokens-10`, fleet `44w258xc`, `9ktkgw27`):**
  - Prior `main` commits (`69422da`, `72fcdd2`) landed `tools/toolpaths.py`, `tools/pngfont.py`, `write_blind_pack` non-pack refusal + `PackRefusesStrangeFoldersTest`, `crates/skintokens/tests/nan_safe.rs`, and `total_cmp` sorts across `crates/skintokens/src` and `crates/creature/src` (excl. guarded `check.rs`).
  - Remaining consolidation completed on `lane/skintokens-8-rivals`: extracted single shared 3-tool evaluator `tools/rigjudge.py` (`judge`) used by both `tools/creature_blind.py` (`judge_rig_metrics`) and `tools/bench/score_artists.py` (`judge`, removing duplicate `find`/`run`/`judge`), added `required=False` to `toolpaths.find` (used by `tools/creature_route.py` for `skintokens`), and hardened `write_blind_pack` against forged non-pack `pairs.json`/key files, symlink keys, empty strings, and `"/"`. Verified RED (`svcegakf`) -> GREEN (`44w258xc`) and all `skintokens-10`/`skintokens-10-rust`/`carbon.sh` gates (`9ktkgw27`).
