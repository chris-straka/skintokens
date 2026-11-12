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
