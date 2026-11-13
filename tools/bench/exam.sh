#!/bin/bash
# The video-to-motion exam on a creature's own skeleton: motionforge's
# two-camera exam (docs/creature.md there), with the outline follower as
# the baseline row.
#
#   BENCH=/path/to/bench tools/bench/exam.sh NAME RIG.glb [NAME RIG.glb ...]
#
# Per creature, in $BENCH/exam/NAME (a step whose output is there is not
# run again):
#   1. creature-profile: the skeleton the reader will solve, its limits,
#      the two cameras.
#   2. creature-poses + render.py: 3,000 random poses filmed by both
#      cameras, half of them blurred by motion.
#   3. train.py: the joint finder for this creature (a DINOv2 ViT-S/14
#      backbone, Apache-2.0, and a small head).
#   4. exam.py: three made-up motions (every joint moving at random
#      within its limits) filmed, read back (finder, two rays crossed,
#      solve) and compared joint by joint with the motion that was made.
#   5. The baseline on the same three motions: the motion is filmed again
#      from the side by one camera on a plain background, and
#      `creature-bench follow` turns the rig until its outline lies on
#      the film's. `creature-bench joints` then measures the finder's
#      answer, the outline follower's and a rig that does not move at
#      all against the same truth.
#
# Needs the GPU (training), Blender, ffmpeg, motionforge's binary
# ($MOTIONFORGE_BIN) and checkout ($MOTIONFORGE_REPO), and a Python with
# torch, transformers and OpenCV ($EXAM_PYTHON).
set -uo pipefail
BENCH=${BENCH:?set BENCH}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
MF=${MOTIONFORGE_BIN:-/home/f/rigwork/creatures/bin/motionforge}
MFREPO=${MOTIONFORGE_REPO:-$HOME/Games/_blender/motionforge}
PY=${EXAM_PYTHON:-$HOME/.genforge/local/venv-triposg/bin/python}
CB=${CREATURE_BENCH_BIN:-$BENCH/bin/creature-bench}
C=$MFREPO/python/creature
step() { echo "[$(date +%H:%M:%S)] $*"; }
while [ $# -ge 2 ]; do
  NAME=$1; RIG=$2; shift 2
  D=$BENCH/exam/$NAME; mkdir -p "$D"
  step "$NAME: profile"
  [ -f "$D/creature.json" ] || "$MF" creature-profile --rig "$RIG" --output "$D/creature.json" --clean-glb "$D/clean.glb" --report "$D/profile.json" --name "$NAME" > "$D/profile.log" 2>&1 || { echo "$NAME: no profile: $(tail -2 "$D/profile.log")"; continue; }
  step "$NAME: poses and renders"
  [ -f "$D/poses.json" ] || "$MF" creature-poses --creature "$D/creature.json" --output "$D/poses.json" --count "${COUNT:-3000}" --seed 1 > /dev/null
  [ -f "$D/renders/truth.json" ] || blender --background --factory-startup --python "$C/render.py" -- --glb "$D/clean.glb" --creature "$D/creature.json" --clip "$D/poses.json" --out "$D/renders" --mode poses --samples 16 > "$D/render.log" 2>&1 || { echo "$NAME: renders failed: $(tail -3 "$D/render.log")"; continue; }
  step "$NAME: joint finder"
  [ -f "$D/finder.pt" ] || "$PY" "$C/train.py" --renders "$D/renders" --out "$D" --epochs "${EPOCHS:-12}" --batch 16 > "$D/train.log" 2>&1 || { echo "$NAME: training failed: $(tail -3 "$D/train.log")"; continue; }
  step "$NAME: exam"
  "$PY" "$C/exam.py" --creature "$NAME=$D" --out "$D/exam" --seeds 1,2,3 --frames 48 --video --motionforge "$MF" --python "$PY" > "$D/exam.log" 2>&1
  tail -n 2 "$D/exam.log"
  step "$NAME: the outline follower on the same motions"
  [ -f "$D/camera.json" ] || blender -b --factory-startup -P "$HERE/render_side.py" -- "$D/clean.glb" --out "$D/start.png" --camera-out "$D/camera.json" > /dev/null 2>&1
  for s in 1 2 3; do
    E=$D/exam/$NAME/seed$s
    [ -f "$E/truth.json" ] || continue
    # the made-up motion as a file of the rig: perfect detections solved, feet not pinned
    "$MF" creature-project --creature "$D/creature.json" --input "$E/truth.json" --output "$E/det_perfect.json" > /dev/null
    "$MF" creature-solve --creature "$D/creature.json" --detections "$E/det_perfect.json" --output "$E/truth_resolved.json" --glb-in "$D/clean.glb" --glb-out "$E/truth.glb" --clip-name truth --no-pin --smooth-sigma 0 --quat-sigma 0 > "$E/truth_solve.log" 2>&1
    "$MF" creature-compare --creature "$D/creature.json" --truth "$E/truth.json" --solved "$E/truth_resolved.json" --output "$E/truth_glb_error.json" > /dev/null 2>&1
    rm -rf "$E/side_film" "$E/overlays"
    blender -b --factory-startup -P "$HERE/render_side.py" -- "$E/truth.glb" --camera "$D/camera.json" --clip truth --frames "$E/side_film" --fps 24 > /dev/null 2>&1
    "$CB" follow "$D/clean.glb" --camera "$D/camera.json" --frames "$E/side_film" --out "$E/followed.glb" --json "$E/follow.json" --name followed --fps 24 --step 1 --overlays "$E/overlays" | cut -c1-160
    "$CB" joints "$E/truth.glb" "$E/solved.glb" --clip truth --other-clip "exam_$s" --json "$E/joints_finder.json" | cut -c1-200
    "$CB" joints "$E/truth.glb" "$E/followed.glb" --clip truth --other-clip followed --json "$E/joints_outline.json" | cut -c1-200
    "$CB" joints "$E/truth.glb" "$D/clean.glb" --clip truth --rest yes --json "$E/joints_rest.json" | cut -c1-200
  done
done
