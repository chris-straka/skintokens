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
