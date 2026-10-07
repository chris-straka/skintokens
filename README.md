# skintokens

Auto-rigging for the HLL character pipeline with SkinTokens / TokenRig
(VAST-AI-Research, arXiv 2602.04805, MIT code and weights), the successor
to UniRig. A mesh goes in; a skeleton and skin weights come out, as GLB.
It is:

- weightforge's **ML candidate** (`weights fix --skintokens` runs
  `skintokens skin`; scored like every other candidate, never trusted
  blindly),
- rigforge's **joint-hint source** (`skintokens joints` writes
  `skintokens-joints/1`, the unirig-joints/1 schema under a new name), and
- genforge's default **rig stage** (`skintokens rig`).

Rust (candle; Metal on Apple Silicon, CPU elsewhere, no CUDA), no Python at
runtime: the official checkpoint is read directly and GLBs are read and
written in pure Rust. It replaced the Python/PyTorch wrapper on
2026-10-05 after matching it tensor for tensor; how parity was proven,
with numbers and sheets: [docs/evaluation.md](docs/evaluation.md).
Licenses and provenance: [PROVENANCE.md](PROVENANCE.md). Public repo:
github.com/chris-straka/skintokens.

## Setup

    cargo build --release      # target/release/skintokens; bin/skintokens runs it
                               # Linux: add --no-default-features (CPU; Metal is macOS-only)
    ./fetch-weights.sh         # once: official grpo_1400.ckpt (1.1 GB, pinned, checksummed)
    bin/skintokens doctor

The weights are looked up in `$SKINTOKENS_CKPT`, then
`$SKINTOKENS_HOME/weights/grpo_1400.ckpt` (`SKINTOKENS_HOME` defaults to
`~/.local/share/skintokens`), then the Hugging Face cache at the pinned
revision. One checkpoint holds everything: TokenRig, the shape encoder
and the FSQ-CVAE skin decoder.

## Use

    skintokens rig    IN.glb OUT.glb [--report R.json] [--class humanoid|quadruped|custom] [--seed N]
    skintokens skin   IN.glb OUT.glb [--report R.json] [--seed N]
    skintokens joints IN.glb OUT.json [--subject NAME]
    skintokens doctor

- **rig**: skeleton + weights for an unrigged mesh. The input GLB is kept
  as is (meshes, materials, textures, node placement); an `Armature` node
  with the joints and a skin are added. Bones follow Blender's
  convention (local Y along the bone, roll 0), like the files the Python
  wrapper got from Blender. For `--class humanoid` the joints are named
  by structure with Mixamo names, so motionforge's `adapter standardize`
  maps them onto the HLL 22-bone skeleton (`DEF-*`); a skeleton without
  the humanoid core fails (`ok: false`, exit 1). Other classes keep
  TokenRig's `bone_N` names (standardize adds the `DEF-` prefix). An
  already rigged input is re-rigged (its skin and animations dropped).
- **skin**: keeps IN's skeleton and rewrites only its JOINTS_0/WEIGHTS_0
  bytes in place (TokenRig's skin-only mode on the given bones). This is
  the weightforge candidate. Twist/helper bones (motionforge's
  `DEF-*_twist.*`, marked by `extras.hll_helper`) share their driver's
  joint, so the model would see two joints in one place; `skin` runs the
  model without them and gives them zero weight; weightforge's fix
  weights them.
- **joints**: rigforge hints from any rigged GLB (no model run).

`R.json` is genforge's adapter contract: `{"ok", "outputs" (relative to
R.json's folder), "tool": "skintokens", "version", "stage", "<stage>":
{...}, "reason"?}`; exit 0 ok, 1 ran and failed (report written), 2 usage
or setup error (no report). Inputs are never modified. A genforge
`local-tools.json` entry looks like:

    "rig": ["skintokens", "rig", "{input}", "{out}", "--report", "{report}", "--class", "{class}"]

One inference runs at a time per machine (a file lock in
`$SKINTOKENS_HOME`). On the M4 (Metal, f32): ~32 s per `rig` and ~21 s
per `skin` on Andras (13.6k verts), ~3.7 GB peak; the Python wrapper took
~48 s and ~35 s. Same `--seed`, same machine, same output (the Python
wrapper was not reproducible); different seeds vary like the original
(raw weightforge 48.9-51.5 on Andras over four seeds).

Environment: `SKINTOKENS_DEVICE=cpu|metal` (default Metal if present),
`SKINTOKENS_DTYPE=bf16` (default f32; bf16 peaks at 2.4 GB, is a little faster, scores
the same), `SKINTOKENS_VERBOSE=1` (stage timings).

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
tokenizer round trip, k-d tree, trim, top-4 weights. Model-level parity
was checked against the Python reference with `skintokens parity-net`
(see docs/evaluation.md); the real-model numbers are there too.

## Layout

- `crates/` - the three crates above.
- `bin/skintokens` - shim to `target/release/skintokens`.
- `fetch-weights.sh` - pinned checkpoint download.
- `tools/joints_samples/` - a `skintokens-joints/1` sample rigforge's tests read.
- `docs/evaluation.md` - SkinTokens vs UniRig vs the game rig; Rust vs Python parity.
