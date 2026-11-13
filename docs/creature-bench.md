# The creature rig benchmark

What a creature rig route is judged by. Two parts:

- **A. Against hand-made animals.** An artist's rig and clips are the
  truth. The mesh is stripped bare, a route rigs it, and the route's rig
  is compared with the artist's: where its joints are, and how far its
  skin lands from the artist's when both play the artist's clips.
- **B. Against a reference video.** A generated creature has no artist's
  rig. Its rig is made to follow a video of the same creature seen from
  the side, and is judged on how well its outline can lie on the video's
  and whether its skin tears while it does.

It replaces the "creature score" (the bend checker's score times a
skeleton check) as the number a route is judged by. That score was
designed beside the route it scored, and the route's last step repairs
exactly the findings it counts; see "Why not the creature score" below.

Code: `crates/bench` (`creature-bench`, Rust), `tools/creature_bench.py`
(part A), `tools/creature_follow.py` (part B), `tools/bench/*`
(Blender scripts and the shell around the routes).

## Part A: how to run it

```sh
B=/path/to/bench                      # a folder of its own, outside any repo

# 1. The truth and the bare mesh of each animal (Blender).
blender -b --factory-startup -P tools/bench/prep_truth.py -- Horse.fbx $B/animals/horse --name horse
#    $B/animals.tsv: name <tab> kind <tab> source file, one animal a line (order of the table)

# 2. Each route on each bare mesh (GPU for the rigger; see the script's head).
BENCH=$B tools/bench/run_routes.sh [names...]

# 3. Scores, the table, pictures.
BENCH=$B tools/bench/build.sh         # creature-bench into $B/bin
CREATURE_BENCH_BIN=$B/bin/creature-bench WEIGHTS_BIN=.../weights \
  python3 tools/creature_bench.py $B --routes new,old --pictures
#    -> $B/out/table.md, summary.json, <animal>.png, <animal>.mp4, all.jpg
```

`prep_truth.py` keeps the artist's rig as it is (every bone, weight and
clip, each clip sampled on every frame) and puts a turn and a scale on a
node above it: the animal faces +Z, stands on the ground, and its
longest side is 1.5 m. `bare.glb` is the same mesh in its rest pose with
its materials and nothing else. A route is any tool that turns
`bare.glb` into a rigged GLB at `animals/<name>/<route>/<name>.glb`; it
may turn the mesh about the up axis, scale it or move it (the scorer
lines the two meshes up again), but it must keep the vertices.

## Part A: what each number means

All lengths are shares of the **body's length**: the longest side of the
rest pose's bounding box.

**Joints.** For each of the artist's bones that hold skin, the distance
from its joint to the route's joint there. "There" needs a rule for
which of the route's bones stands for which of the artist's:

- A route's bone *lies along* the artist's bone whose line (joint to
  child joint; for a last bone, joint to the middle of the skin it holds)
  is nearest to the middle half of the route's bone, among lines that
  point the same way (an opposite direction costs 0.3 body lengths, a
  right angle 0.15). Only the artist's bones that hold skin, or carry a
  bone that does, can be followed: an IK handle cannot.
- The joint's error is the distance to the nearest of the route's joints
  whose bone lies along that bone, the bone above it or a bone below it.
  If there is none, the rig stops short of that joint: its error is the
  distance back to the nearest joint on the last bone the rig does have
  on the way up, and it is counted as "without a joint".
- The table's number is the mean over the **leg, neck, head, jaw, tail
  and wing** joints (what a bone is comes from its name; `--roles` or a
  `roles.json` beside `truth.glb` corrects one). The trunk's joints are
  reported apart (`trunk_joint_error_pct`): where an artist puts a body
  or hips pivot is a habit, and half the animals here hang the whole
  trunk from one bone in the middle of the body.

**Missing.** Chains of the artist's (bones of one kind joined through
the hierarchy: a tail, a leg, a jaw) that none of the route's bones lies
along. `short_chains` in the JSON lists chains the route covers with
fewer bones; that is not counted as a fault. **Extra**: the route's
skin-holding bones that lie along nothing of the artist's (farther than
0.12 body lengths from every line).

**Skin.** The artist's clips are played on the artist's rig: that is
where every vertex truly goes. The route's rig then follows each frame
as well as its bones allow, in two steps:

1. *Rotations carried.* Each of the route's bones is turned as the
   artist's bone it lies along turns (the world rotation from rest to
   the frame). A joint stays on its parent; a joint with no parent moves
   as its artist's bone moves. Where the artist slides a bone along its
   parent (a body bobbing on a root, a foot on an IK handle) and the
   route's bone hangs from the bone that follows that parent, it slides
   the same way. This is a plain retarget.
2. *Fitted.* From there, six times over, each joint in turn (parents
   first) is rotated about its own place, everything below it turning
   with it, to the rotation that leaves the whole skin nearest to the
   artist's (least squares; a joint with no parent may also move). A
   step is kept only if the skin gets nearer.

What is left after step 2 is the **skin error**: per frame the mean and
the 95th percentile over the vertices of the distance between the two
skins; then the mean over a clip's frames (at most 48 a clip, spread
evenly), then over the clips, so a long idle counts as much as a short
attack. `worst_frame` is the frame with the highest 95th percentile and
the artist's bone under which the skin is farthest; the pictures are
taken there.

Why fitted and not carried: carrying rotations charges a rig for
splitting a limb differently from the artist (a hip bone inside the
body where the artist has a thigh). The fit asks the fairer question:
*can this rig get there at all?* It starts from the carried pose and
only improves on it, so the artist's own rig scores exactly 0. The
carried number is in the JSON (`skin_rotations_carried`) as what a
naive retarget of the artist's clips would look like.

**Weights only.** Each of the route's bones is carried bodily (rotation
and place) by the artist's bone it lies along, so where its joints are
stops counting. What is left is what the weights alone cost.

**Tears.** `weights check` (weightforge) on the route's rig carrying the
fitted frames, with no pose set: the vertices in failing stretch
(`D_STRETCH`), volume (`D_VOLUME`) and bleed (`D_BLEED`) findings as a
share of all vertices. `D_NOISE` (speckled weights) is left out: it
flags the hand-painted weights of these low-poly animals on 5 to 15% of
their vertices and says nothing about motion. The artist's own rig is
checked on the same frames: that row is the line to beat.

**Feet.** Skin held by a leg bone that stands within 5% of the ground in
the rest pose, gathered into feet. While the artist's foot is on the
ground (within 2%), how far across the ground the route's foot is from
it (mean), and how far under it at most. This stands in for a foot-slide
check: the clips are played in place, so a planted foot is not a still
foot, and `motionforge motion check` reads human ankles.

**Controls**, scored like routes:

- *the artist's rig*: must be 0 everywhere but tears. If it is not, the
  truth has something the method cannot carry (a bone scaled in a clip).
- *the artist's skeleton, crude weights*: every vertex wholly on its
  nearest bone (`creature-bench rigid`). A perfect skeleton with the
  worst sane weights. A route that scores worse than this on skin has a
  skeleton problem.

## Part A: what the first run says (2026-11-12)

Twelve animals (four packs by Quaternius and the Khronos fox, all CC0
or CC-BY; sources and checksums beside the downloads): horse, pug, cow,
fox, tyrannosaur, velociraptor, apatosaurus, rat, dragon, bat, spider,
wasp. Medians:

| Rig | Joints | Animals with a missing chain | Skin mean | Skin 95th | Weights only | Tears |
|---|---|---|---|---|---|---|
| the artist's rig | 0 | 0 of 12 | 0 | 0 | 0 | 0.8% |
| artist's skeleton, crude weights | 0 | 0 of 12 | 0.6% | 1.9% | 0.6% | 6.8% |
| the body-read route (`creature_route.py`) | 9.0% | 10 of 12 | 1.6% | 3.5% | 1.3% | 0.0% |
| the predicted-skeleton route (before it) | 2.9% | 3 of 12 | 0.7% | 1.8% | 0.4% | 0.2% |

(The table of record is the run's own `table.md`; these are the numbers
of the day this was written.)

- The body-read route is farther from the artist than the route it
  replaced on 11 of 12 animals and level on one (where it kept the
  predicted rig). It is also farther than crude weights on the artist's
  skeleton: its problem is the skeleton, not the weights.
- It reads short-legged bodies wrongly and passes its own check anyway:
  the cow and the pug as two-legged (no front legs), the apatosaurus as
  legless (no leg bones at all; skeleton check 92.6, "passes"). On every
  four-legged animal its leg starts where the limb leaves the body, so
  the thigh and the shoulder blade inside the body have no bone, and its
  hip joints are about a tenth of the body's length from the artist's.
- Its tears are 0%: its last step repairs exactly what the bend checker
  flags. The artist's own rigs tear 0 to 6.7% by that checker in their
  own clips. Low tears do not mean a rig moves like the animal.
- The predicted skeleton's lead here is partly unfair: see the first
  limit below.

## Part A: limits

- **The old route's model has probably seen these animals.** They are
  public CC0 models of the kind its training set was gathered from. On
  the fox it returns the artist's 22 bones in the artist's order, each
  within 0.6% of the body's length. A probe (`tools/bench/leak_probe.sh`:
  five animals made 35% taller and 20% shorter, rigged again) shows it
  is not replaying stored coordinates: the joints move with the body
  (fox 0.5% to 0.7%, tyrannosaur 3.8% to 4.1%, bat 1.9% to 1.5%, horse
  5.1% to 5.9%; the cow breaks, 3.0% to 11.1%). It cannot show that the
  model would do as well on an animal it has never met. The old route's
  numbers here are an upper bound; the body-read route's are honest,
  since it has no memory.
- **Low-poly, clean bodies.** 600 to 4,000 triangles, no fur, no
  fused limbs, no cloaks. The owner's creatures are 18,000-triangle
  generated meshes. A route can do well here and badly there (part B is
  for that), and crude weights look better here than they would on a
  dense mesh (few vertices sit between two bones).
- **One artist's habits.** Ten of the twelve are by one author, with
  IK feet under the root and a trunk hung from one middle bone. The
  joint number leaves the trunk out for that reason.
- **Bone names decide roles.** A bone with an odd name lands in "other"
  and drops out of the joint mean; `creature-bench info` prints what
  each bone was taken for.
- **No jaws, toes or ears in the truth** of most of these animals, so
  the benchmark cannot yet say a route lacks them.
- **Motion checks are relative.** Feet are measured against the
  artist's feet, not against the ground under a travelling body.

## Part B: how to run it

```sh
D=$B/video/wolf                       # one creature: rig.glb is the route's rig
blender -b --factory-startup -P tools/bench/render_side.py -- $D/rig.glb --out $D/start.png --camera-out $D/camera.json
# start.png is the first frame given to a video model (image to video,
# still camera, plain background): save each result as $D/<clip>.mp4
CREATURE_BENCH_BIN=$B/bin/creature-bench WEIGHTS_BIN=.../weights python3 tools/creature_follow.py $D walk attack
#    -> $D/<clip>.followed.glb (the rig with the followed motion as a clip),
#       <clip>.follow.json, <clip>.loop.mp4 (video | rig | outlines), follow.json
```

## Part B: what each number means

- **The video's outline**: per frame, what differs from the background
  (the median colour of the picture's border) by more than 48 of 255,
  largest piece, at a quarter of the picture's size.
- **Lining up**: a video model does not hold the first picture: it zooms
  and moves it a little and the creature is already moving. So on the
  first frame the camera's zoom and place and the joints' turns are
  fitted together, the rig standing where it stands. The zoom and move
  found are in the report (`camera_lined_up`).
- **Following**: the rig's pose is one turn per joint in the picture's
  plane (about the camera's axis, relative to its parent) and the
  body's place. From the frame before, each number is nudged up and
  down by a step that halves (12, 6, 3, 1.5 degrees), and a nudge is
  kept when the outlines get nearer: the mean distance from each
  outline's pixels to the other outline, plus a small charge for
  turning far from the frame before and a charge for going under the
  ground. Every second video frame is used. The poses are then smoothed
  over time (weights 1 2 1) and lifted so the lowest point is on the
  ground.
- **Overlap**: pixels in both outlines over pixels in either, per frame;
  the mean and the worst frame. The rest pose laid on the first frame
  is given for scale.
- **Tears while following**: as in part A, on the followed clip.
- **The followed clip** is a first real clip for that creature, with
  the limits below.

## Part B: limits

- **One camera sees no depth.** Nothing turns out of the picture's
  plane; a left and a right leg are told apart only by which fits; a
  creature that turns toward the camera in the video cannot be
  followed, and the overlap says so.
- **The outline is all that is matched.** A limb in front of the body
  is invisible to it. Fur and rags that the video moves and the rig
  cannot will always cost overlap.
- **The video is a made-up reference**, not a truth: its creature may
  change shape. Nothing from it ships.
- **The search is local** (nudges from the frame before): a fast limb
  can be lost and found again late.

## Why not the creature score

- It multiplies the bend checker's 0 to 100 score (half of which is the
  single worst region, so it counts faults more than it measures
  quality: hand-made rigs score 3 to 70) by a skeleton check written
  for the route under test.
- The route's fix step minimises the bend findings, and its skeleton
  check passes rigs with no leg bones (above).
- Its before and after were bent through different poses: each rig's
  own clips were in the set. With the clips left out, 29 creatures go
  55.9 to 74.3 (with them, 52.8 to 73.3), still on pose sets that
  differ on 23 of 29 because a rig with no bones along a limb gets no
  poses for it.

Tears (a plain share of vertices) is kept from that checker; the score
is not.
