# Provenance and licenses

Checked 2026-10-05. The wrapper ships no model code or weights; `setup.sh`
fetches them into `$SKINTOKENS_HOME` (default `~/.local/share/skintokens`)
and the Hugging Face cache.

| Part | Source (pinned) | License |
|---|---|---|
| Wrapper (`skintokens/`, `bin/`, `setup.sh`, tests) | this repo | MIT (`LICENSE`) |
| `patches/apple-silicon.patch` | written here against upstream | MIT (same as upstream) |
| SkinTokens / TokenRig code | github.com/VAST-AI-Research/SkinTokens @ `273b691d` | MIT (upstream `LICENSE`) |
| TokenRig + FSQ-CVAE checkpoints (`grpo_1400.ckpt`, `last.ckpt`) | huggingface.co/VAST-AI/SkinTokens @ `79736cad` | MIT (model card) |
| Qwen3-0.6B architecture config (tokenizer/config only; the fine-tuned weights live in the TokenRig checkpoint) | huggingface.co/Qwen/Qwen3-0.6B @ `c1899de2` | Apache-2.0 |
| Shape encoder code (`src/model/michelangelo/`) | derived from NeuralCarver/Michelangelo | upstream SkinTokens ships it without a header; Michelangelo itself is GPL-3.0 (same caveat as unirig-mac) |
| `bpy` (helper process) | PyPI `bpy` 5.0.1 | GPL-3.0 |

What that means here: everything runs locally and only the rigged GLBs
leave the machine. The rigs (skeleton positions and weights) are model
output, not GPL code, so shipping them in a game is fine. Distributing this
tool's runtime (the upstream checkout with the Michelangelo-derived encoder,
or bpy) would need GPL-3.0 compliance; this repo distributes neither.

## Conversions considered and not used

| Conversion | License | Why not |
|---|---|---|
| mlx-community/SkinTokens-bf16 | MIT (+ Apache-2.0 Qwen) | Swift-MLX only (needs the third-party `mlx-engine-swift` runtime); not the official code path, no parity numbers published |
| LocalAI-io/SkinTokens-GGUF + localai-org/skin-tokens.cpp | MIT weights, Apache-2.0 code | third-party C++/GGML port (CPU/Vulkan); would need its own parity check |
| fernandotonon/QtMeshEditor-skintokens-onnx | MIT | ONNX export for one app; same |

The official PyTorch code runs on the M4's GPU (MPS) with four small
patches (see `patches/apple-silicon.patch`), so no conversion is needed and
there is nothing to prove faithful: the weights are the upstream bf16
checkpoints, loaded by the upstream model code.

## Asset provenance for rigs made with this tool

Log a rig produced here as: "skeleton + skin weights by SkinTokens
(VAST-AI, MIT code and weights) via `skintokens rig` vX, seed N". The mesh
keeps its own provenance (Tripo plan, retopoforge, etc.).
