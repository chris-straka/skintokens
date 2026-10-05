#!/bin/bash
# Build the pinned SkinTokens runtime outside the repo:
#   $SKINTOKENS_HOME/upstream  VAST-AI-Research/SkinTokens @ UPSTREAM + patches/apple-silicon.patch
#   $SKINTOKENS_HOME/.venv     Python 3.11 from requirements.lock.txt (uv)
#   weights                    HF cache (VAST-AI/SkinTokens @ HF_REV, Qwen/Qwen3-0.6B config), linked in
# Idempotent. Needs git, uv, network on first run (~1.6 GB of weights).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
home="${SKINTOKENS_HOME:-$HOME/.local/share/skintokens}"
UPSTREAM=273b691d35989d71cd17ff2895fdc735097b92d1
HF_REV=79736cad0fd84de384d5eede659b4ebd24effe33
QWEN_REV=c1899de289a04d12100db370d81485cdf75e47ca
mkdir -p "$home"
if [ ! -d "$home/upstream/.git" ]; then
  git clone -q https://github.com/VAST-AI-Research/SkinTokens.git "$home/upstream"
fi
cd "$home/upstream"
if ! git rev-parse -q --verify refs/heads/mac >/dev/null; then
  git checkout -q -b mac "$UPSTREAM"
  git apply "$here/patches/apple-silicon.patch"
  git -c user.name=skintokens -c user.email=skintokens@localhost commit -qam "Apple Silicon port (skintokens wrapper)" 
fi
if [ ! -x "$home/.venv/bin/python" ]; then
  uv venv -q --python 3.11 "$home/.venv"
  VIRTUAL_ENV="$home/.venv" uv pip install -q -r "$here/requirements.lock.txt"
fi
"$home/.venv/bin/python" - "$HF_REV" "$QWEN_REV" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download, snapshot_download
rev, qrev = sys.argv[1:3]
for f in ("experiments/skin_vae_2_10_32768/last.ckpt",
          "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"):
    src = hf_hub_download("VAST-AI/SkinTokens", f, revision=rev)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    if not os.path.lexists(f):
        os.symlink(src, f)
q = snapshot_download("Qwen/Qwen3-0.6B", revision=qrev, ignore_patterns=["*.bin", "*.safetensors"])
os.makedirs("models", exist_ok=True)
if not os.path.lexists("models/Qwen3-0.6B"):
    os.symlink(q, "models/Qwen3-0.6B")
PY
"$here/bin/skintokens" doctor
