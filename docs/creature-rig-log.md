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
