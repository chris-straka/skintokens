# skintokens

Auto-rigging for the HLL character pipeline with SkinTokens / TokenRig
(VAST-AI-Research, arXiv 2602.04805, MIT code and weights), the successor
to UniRig. A mesh goes in; a skeleton and skin weights come out, as GLB.
Replaces `~/SWE/blender/unirig-mac` (retired 2026-10-05) as:

- weightforge's **ML candidate** (`weights fix --skintokens` runs
  `skintokens skin`; scored like every other candidate, never trusted
  blindly), and
- rigforge's **joint-hint source** (`skintokens joints` writes
  `skintokens-joints/1`, the unirig-joints/1 schema under a new name).

Local-only repo (no remote). Python because the model is PyTorch with no
other runtime; everything else calls it as a CLI. Why it replaced UniRig,
with numbers and sheets: [docs/evaluation.md](docs/evaluation.md).
Licenses and provenance: [PROVENANCE.md](PROVENANCE.md).

## Setup

    ./setup.sh          # once: pinned upstream + Mac patch, venv, weights (~1.6 GB)
    bin/skintokens doctor

`setup.sh` builds `$SKINTOKENS_HOME` (default `~/.local/share/skintokens`):
the upstream checkout at `273b691d` with `patches/apple-silicon.patch`
applied, a Python 3.11 venv from `requirements.lock.txt`, and the
checkpoints in the Hugging Face cache (pinned revisions) linked in.
Nothing big lives in this repo.

## Use

    skintokens rig    IN.glb OUT.glb [--report R.json] [--class humanoid|quadruped|custom] [--seed N]
    skintokens skin   IN.glb OUT.glb [--report R.json] [--seed N]
    skintokens joints IN.glb OUT.json [--subject NAME]
    skintokens doctor

- **rig**: skeleton + weights for an unrigged mesh (texture and scale
  kept). For `--class humanoid` the joints are named by structure with
  Mixamo names, so motionforge's `adapter standardize` maps them onto the
  HLL 22-bone skeleton (`DEF-*`); a skeleton without the humanoid core
  fails (`ok: false`, exit 1). Other classes keep TokenRig's `bone_N`
  names (standardize adds the `DEF-` prefix).
- **skin**: keeps IN's skeleton and writes only new JOINTS_0/WEIGHTS_0
  (TokenRig's skin-only mode on the given bones). This is the weightforge
  candidate.
- **joints**: rigforge hints from any rigged GLB.

`R.json` is genforge's adapter contract: `{"ok", "outputs" (relative to
R.json's folder), "tool": "skintokens", "version", "stage", "<stage>":
{...}, "reason"?}`; exit 0 ok, 1 ran and failed (report written), 2 usage
or setup error (no report). Inputs are never modified. A genforge
`local-tools.json` entry looks like:

    "rig": ["skintokens", "rig", "{input}", "{out}", "--report", "{report}", "--class", "{class}"]

One inference runs at a time per machine (a file lock in
`$SKINTOKENS_HOME`). On the M4 (MPS): ~35-55 s per character, ~4 GB.
Sampling is stochastic and MPS kernels are not bit-deterministic: the
same `--seed` gives joints within ~2 cm and locally different weights.
Over four Andras runs the weightforge score stayed in 44-48 raw and
59-62 after `weights fix`, so pick-best over seeds buys little.

## Checks

    $SKINTOKENS_HOME/.venv/bin/python -m unittest discover -s tests

Fast tests, no model: GLB IO, humanoid naming (both facings, rejects
non-humanoids), skin-only weight mapping (skeleton untouched, only
weights change), joints doc, report schema, exit 2 without a report.
The real-model numbers are in `docs/evaluation.md`.

## Port notes (patches/apple-silicon.patch)

Upstream targets CUDA + flash-attn. The patch: device pick
(`SKINTOKENS_DEVICE`, else cuda > mps > cpu) for autocast and model
placement; an SDPA stand-in for `flash_attn_interface`; Qwen3 with
`sdpa` attention off CUDA; no dataloader worker processes off CUDA (MPS
tensors cannot be pickled); `--cls` and `--seed` on `demo.py`. Two
upstream bugs fixed on the way: predicted joint names were dropped from
the asset, and skin-only transfer crashed when the input armature was not
named `Armature`.

## Layout

- `skintokens/` — CLI (`cli.py`), GLB access (`glb.py`), naming (`naming.py`). MIT.
- `bin/skintokens` — shim into the pinned venv.
- `patches/`, `setup.sh`, `requirements.lock.txt` — the pinned runtime.
- `tests/` — fast tests + synthetic GLB fixture.
- `docs/evaluation.md` — SkinTokens vs UniRig vs Tripo/game weights.
