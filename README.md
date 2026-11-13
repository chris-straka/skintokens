# skintokens

Auto-rigging for the game character pipeline with SkinTokens / TokenRig
(VAST-AI-Research, arXiv 2602.04805, MIT code and weights), the successor
to UniRig. A mesh goes in; a skeleton and skin weights come out, as GLB.
It is:

- weightforge's **ML candidate** (`weights fix --skintokens` runs
  `skintokens skin`; scored like every other candidate, never trusted
  blindly),
- rigforge's **joint-hint source** (`skintokens joints` writes
  `skintokens-joints/1`, the unirig-joints/1 schema under a new name), and
- genforge's default **rig stage** (`skintokens rig`).

Rust (candle; Metal on Apple Silicon, CUDA on an NVIDIA card with the `cuda`
feature, CPU elsewhere), no Python at runtime: the official checkpoint is read directly and GLBs are read and
written in pure Rust. It replaced the Python/PyTorch wrapper on
2026-10-05 after matching it tensor for tensor; how parity was proven,
with numbers and sheets: [docs/evaluation.md](docs/evaluation.md).
Licenses and provenance: [PROVENANCE.md](PROVENANCE.md). Public repo:
github.com/chris-straka/skintokens.

## Setup

    cargo build --release      # target/release/skintokens; bin/skintokens runs it
                               # Linux: add --no-default-features (CPU; Metal is macOS-only)
                               # NVIDIA: --no-default-features --features cuda (below)
    ./fetch-weights.sh         # once: official grpo_1400.ckpt (1.1 GB, pinned, checksummed)
    bin/skintokens doctor

The weights are looked up in `$SKINTOKENS_CKPT`, then
`$SKINTOKENS_HOME/weights/grpo_1400.ckpt` (`SKINTOKENS_HOME` defaults to
`~/.local/share/skintokens`), then the Hugging Face cache at the pinned
revision. One checkpoint holds everything: TokenRig, the shape encoder
and the FSQ-CVAE skin decoder.

## Use

    skintokens rig    IN.glb OUT.glb [--report R.json] [--class humanoid|quadruped|custom] [--seed N]
                      [--transfer straight|surface|facing]
    skintokens skin   IN.glb OUT.glb [--report R.json] [--seed N] [--transfer ...]
    skintokens joints IN.glb OUT.json [--subject NAME]
    skintokens doctor

- **rig**: skeleton + weights for an unrigged mesh. The input GLB is kept
  as is (meshes, materials, textures, node placement); an `Armature` node
  with the joints and a skin are added. Bones follow Blender's
  convention (local Y along the bone, roll 0), like the files the Python
  wrapper got from Blender. For `--class humanoid` the joints are named
  by structure with Mixamo names, so motionforge's `adapter standardize`
  maps them onto the standard humanoid 22-bone skeleton (`DEF-*`); a skeleton without
  the humanoid core fails (`ok: false`, exit 1). Other classes keep
  TokenRig's `bone_N` names (standardize adds the `DEF-` prefix). An
  already rigged input is re-rigged (its skin and animations dropped).
- **skin**: keeps IN's skeleton and rewrites only its JOINTS_0/WEIGHTS_0
  bytes in place (TokenRig's skin-only mode on the given bones). This is
  the weightforge candidate. Twist/helper bones (motionforge's
  `DEF-*_twist.*`, marked by `extras.helper_bone`) share their driver's
  joint, so the model would see two joints in one place; `skin` runs the
  model without them and gives them zero weight; weightforge's fix
  weights them.
- **joints**: rigforge hints from any rigged GLB (no model run).
- **--transfer**: how the weights the model predicts on 54,000 surface
  points reach the vertices (each vertex blends its 8 nearest points).
  `straight` (default, upstream's) measures through space; `surface`
  walks over the mesh, welded across UV seams, so pieces that nearly touch
  but aren't joined keep their own weights; `facing` keeps the straight
  line but drops points on another piece or facing the other way.
  `SKINTOKENS_TRANSFER` sets it when the flag isn't given. On generated
  humans the choice barely matters (docs/evaluation.md, "Weights along
  the surface"); `facing` scores worse and stays only for comparison.

`R.json` is genforge's adapter contract: `{"ok", "outputs" (relative to
R.json's folder), "tool": "skintokens", "version", "stage", "<stage>":
{...}, "reason"?}`; exit 0 ok, 1 ran and failed (report written), 2 usage
or setup error (no report). Inputs are never modified. A genforge
`local-tools.json` entry looks like:

    "rig": ["skintokens", "rig", "{input}", "{out}", "--report", "{report}", "--class", "{class}"]

One inference runs at a time per machine (a file lock in
`$SKINTOKENS_HOME`). On the M4 (Metal, f32): ~32 s per `rig` and ~21 s
per `skin` on a 13.6k-vert hero mesh, ~3.7 GB peak; the Python wrapper took
~48 s and ~35 s. Same `--seed`, same machine, same output (the Python
wrapper was not reproducible); different seeds vary like the original
(raw weightforge 48.9-51.5 on that hero over four seeds).

**CUDA.** `cargo build --release --no-default-features --features cuda`
needs a CUDA toolkit at build time (`nvcc`; `CUDA_ROOT` names it) and its
libraries at run time (`LD_LIBRARY_PATH=$CUDA_ROOT/lib`). nvcc 12.8 refuses
gcc 15: point it at a gcc it accepts with
`NVCC_PREPEND_FLAGS="-ccbin /path/to/g++"`. On an RTX 2060 SUPER (8 GB) a
`rig` of an 18,000-triangle creature takes 26 s (ten minutes on that
machine's CPU), 1.6 GB of memory and 5.5 GB of the card, and gives the same
skeleton as the CPU, joint for joint.

**A skeleton that doesn't end.** Generation has 1,534 tokens (2,048 less
the prompt), and a skeleton is followed by four skin tokens per joint. On a
shape the model can't read (a cloaked figure in rags: a chain for every
strip) the skeleton used to run to the limit, so no skin was written and
there was no rig; and the beam cache, one slot per beam per step, doubled
its way past 12 GB on the way. Now (`--report`'s
`skeleton_cut_at_token_limit`): the skeleton has to end while its skin
still fits (146 joints at most), the cache reclaims dropped beams' slots
(0.6 GB whatever the length; `cargo test` holds it there), and a branch
can't be left half written. That figure rigs in 84 s at 1.6 GB.

Environment: `SKINTOKENS_DEVICE=cpu|metal|cuda` (default: the GPU the build has),
`SKINTOKENS_DTYPE=bf16` (default f32; bf16 peaks at 2.4 GB, is a little faster, scores
the same), `SKINTOKENS_VERBOSE=1` (stage timings), `SKINTOKENS_TRANSFER`
(see `--transfer`), `SKINTOKENS_PRED_CACHE=<prefix>` (developer hook: keep
the model's prediction in `<prefix>.ids.npy`/`.skin.npy` and reuse it, so
transfers are compared on one prediction).

## Creatures: `creature-rig` and `tools/creature_route.py`

The model was trained on people and common animals and gives a shaggy
beast, a moth or a cloaked giant a skeleton that misses a tail, has a
chain for every strip of cloth, or pivots a leg in the wrong place.
`creature-rig` (`crates/creature`, no model, half a second a mesh) reads
the body itself:

    creature-rig parts MESH.glb [--json P.json] [--sheet P.png]
    creature-rig rig   MESH.glb OUT.glb [--rom ROM.ron] [--json R.json] [--sheet S.png] [--blend 1.6]
    creature-rig check RIG.glb [--json C.json] [--rom ROM.ron] [--sheet S.png]

- **parts**: the mesh as a solid on a 128-cell grid (holes and parts
  pushed through each other tolerated), distance through the inside from
  its deepest point, and the tree of that distance's shells: its long
  branches are the limbs, each with a centre line inside the body. They
  are named by where they start and end: legs, arms, neck and head,
  tail, wings, and what kind of body it is (four-legged, two-legged,
  winged, many-limbed, legless). Toes, ears, horns and fur get nothing.
- **rig**: hips and spine along the trunk, a chain down every limb with
  joints where its line turns (a shoulder or hip inside the body first),
  the same joints left and right, bones named `DEF-<part>.<nnn>.<side>`;
  weights from each bone's distance through the solid, shared across a
  joint over a band `--blend` body thicknesses wide; the body's range of
  motion as a weightforge pose set (`--rom`). A body the clean step left
  facing -Z (read from its toes) is turned to face +Z.
- **check**: the skeleton check for any rig of a creature, whoever made
  it: every leg, arm, tail, wing and the neck and head has a chain of
  bones, the bones lie inside the body, a left and its right match
  (0 to 100, findings `S_NO_CHAIN`, `S_SHORT_CHAIN`, `S_OUTSIDE`,
  `S_LOPSIDED`; exit 1 when one fails). `--rom` writes the same range of
  motion on that rig's bones, so two rigs are bent the same way.

`tools/creature_route.py MODEL.fbx OUT --name NAME` is the whole route
(motionforge's `tripo_rig.py` calls it for a creature): clean,
`creature-rig rig` at three band widths, walk and trot for a four-legged
body, weightforge's fix choosing per region among those widths and the
model's own weights for this skeleton (`skintokens skin`), then the bend
gate, the skeleton check and rfcheck. It also makes the model's own rig
(`skintokens rig`, three seeds), judges it the same way, and keeps it
instead where it is the better one, so a creature is never rigged worse
than by the model alone.

On 29 generated creatures (four-legged, two-legged, winged, cloaked, a
hand, a fish), judged alike (the body's range of motion on each rig's own
bones and its clips, times the skeleton check): 52.8 for the model's rig
alone (best of three seeds), 73.3 for this route; rigs with no failing
finding in either check 1 and 13; failing skeleton findings 39 and 8;
failing bend findings 53 and 33. Higher on 25, lower on 3.
What was tried and what it scored:
[docs/creature-rig-log.md](docs/creature-rig-log.md).

## How it is built

- `crates/skintokens` (MIT): CLI, GLB I/O (`gltf` crate to read; edits on
  the raw JSON/binary so untouched data stays byte for byte), mesh
  pipeline (upstream's trim/normalize/surface sampling, Blender's
  armature frame and duplicate-face drop), TokenRig's Qwen3 LM, the
  skeleton tokenizer, transformers' beam sampling with TokenRig's
  vocabulary switch, the FSQ-CVAE decoder, humanoid naming.
- `crates/skintokens-encoder` (**GPL-3.0**): the point-cloud shape
  encoder, ported from Michelangelo-derived code. The binary links it,
  so the built binary is GPL-3.0; callers only run it as a separate
  process.
- `crates/skintokens-nn` (MIT): shared layers, attention, FPS.
- `crates/creature` (MIT): `creature-rig`; uses the skintokens crate for
  GLB I/O only (it links it, so its binary is GPL-3.0 as well).

Upstream details kept on purpose (all verified against the Python run):
the 54k-point cloud is surface samples only; the VAE embedder and rotary
frequencies use bf16-rounded constants (upstream casts those buffers);
the generation stops one skin token early and decodes the last joint's
fourth code from the end-of-sequence id; influences <= 1e-4 are dropped
as Blender's exporter does. Dropped: `--postprocess` (upstream's voxel
skin pass, unused by any caller).

## Checks

    cargo test --release

Fast tests, no model: GLB I/O, humanoid naming (both facings, rejects
non-humanoids), helper-bone detection, joints doc, CLI exit codes,
tokenizer round trip, k-d tree, trim, top-4 weights, and the weight
transfer (two tubes 4 mm apart, each on its own bone: the straight copy
mixes them, the surface and facing copies must not). Model-level parity
was checked against the Python reference with `skintokens parity-net`
(see docs/evaluation.md); the real-model numbers are there too.

## Layout

- `crates/` - the three crates above.
- `bin/skintokens` - shim to `target/release/skintokens`.
- `fetch-weights.sh` - pinned checkpoint download.
- `tools/joints_samples/` - a `skintokens-joints/1` sample rigforge's tests read.
- `tools/transfer_bench.sh`, `tools/transfer_bench_summary.py` - the rig
  route (rig, motionforge standardize, weightforge check/fix/check) over a
  folder of meshes, one transfer at a time, and its table.
- `docs/evaluation.md` - SkinTokens vs UniRig vs the game rig; Rust vs Python parity.
