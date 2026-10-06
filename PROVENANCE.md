# Provenance and licenses

Checked 2026-10-05. This repo ships source only; `fetch-weights.sh`
downloads the official checkpoint (or the Hugging Face cache already has
it). No Python, PyTorch or Blender is used at runtime.

| Part | Source (pinned) | License |
|---|---|---|
| `crates/skintokens` (CLI, GLB I/O, mesh pipeline, Qwen3 LM, tokenizer, generation, FSQ-CVAE decoder, naming), `crates/skintokens-nn`, `bin/`, `fetch-weights.sh`, tests | this repo; ports of SkinTokens' MIT code, transformers' beam search (Apache-2.0), diffusers' DiT block (Apache-2.0), vector-quantize-pytorch's FSQ (MIT) | MIT (`LICENSE`) |
| `crates/skintokens-encoder` (point-cloud shape encoder) | port of SkinTokens' `src/model/michelangelo/` (ShapeAsLatentPerceiverEncoder, attention blocks, FourierEmbedder), derived from NeuralCarver/Michelangelo | **GPL-3.0-only** (`crates/skintokens-encoder/COPYING`, SPDX header in the source) |
| The built `skintokens` binary | links the GPL-3.0 encoder crate | GPL-3.0 as a whole (MIT parts stay MIT as source) |
| SkinTokens / TokenRig reference code | github.com/VAST-AI-Research/SkinTokens @ `273b691d` | MIT (upstream `LICENSE`) |
| TokenRig checkpoint `grpo_1400.ckpt` (also holds the FSQ-CVAE and encoder weights) | huggingface.co/VAST-AI/SkinTokens @ `79736cad`, sha256 `f4e4706a...5692` | MIT (model card) |
| Qwen3-0.6B architecture (layer shapes, rope theta; hard-coded, no files fetched) | huggingface.co/Qwen/Qwen3-0.6B @ `c1899de2` | Apache-2.0 |
| candle, gltf, serde_json, anyhow (crates.io) | see `Cargo.lock` | MIT / Apache-2.0 |

What that means here: everything runs locally and only the rigged GLBs
leave the machine. The rigs (skeleton positions and weights) are model
output, not GPL code, so shipping them in a game is fine. The GPL-3.0
encoder is kept in its own crate; the MIT tools that use SkinTokens
(weightforge, rigforge, genforge, motionforge) only run the `skintokens`
CLI as a separate process and link none of it. Distributing the built
binary means distributing it under GPL-3.0 (source: this repo).

## History

Until 2026-10-05 this repo was a Python wrapper around the upstream
PyTorch code with an Apple Silicon patch and a `bpy` helper (GPL-3.0)
for GLB I/O. It was replaced after the Rust port matched it (see
docs/evaluation.md, "Rust port"); the last Python version is commit
`76d7311` (wrapper, `patches/apple-silicon.patch`, `setup.sh`,
`requirements.lock.txt`) and the parity dump scripts are in `parity/` at
that commit.

## Conversions considered and not used

| Conversion | License | Why not |
|---|---|---|
| mlx-community/SkinTokens-bf16 | MIT (+ Apache-2.0 Qwen) | Swift-MLX only; not the official weights file, no parity numbers |
| LocalAI-io/SkinTokens-GGUF + localai-org/skin-tokens.cpp | MIT weights, Apache-2.0 code | third-party C++/GGML port; would need its own parity check |
| fernandotonon/QtMeshEditor-skintokens-onnx | MIT | ONNX export for one app; same |

The Rust port loads the official `.ckpt` itself (candle reads the torch
zip pickle), so there is no converted weights file to trust.

## Asset provenance for rigs made with this tool

Log a rig produced here as: "skeleton + skin weights by SkinTokens
(VAST-AI, MIT code and weights) via `skintokens rig` vX (Rust), seed N".
The mesh keeps its own provenance (Tripo plan, retopoforge, etc.).
