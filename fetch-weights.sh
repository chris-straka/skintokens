#!/bin/bash
# Fetch the official SkinTokens checkpoint (MIT) at the pinned revision.
# grpo_1400.ckpt holds TokenRig, the shape encoder and the FSQ-CVAE (~1.1 GB).
# Skipped if it is already in the Hugging Face cache or $SKINTOKENS_HOME/weights.
set -euo pipefail
REV=79736cad0fd84de384d5eede659b4ebd24effe33
SHA=f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692
REL=experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt
home="${SKINTOKENS_HOME:-$HOME/.local/share/skintokens}"
hf="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}/models--VAST-AI--SkinTokens/snapshots/$REV/$REL"
dst="$home/weights/grpo_1400.ckpt"
for f in "$hf" "$dst"; do
  if [ -f "$f" ]; then echo "weights present: $f"; exit 0; fi
done
mkdir -p "$home/weights"
curl -fL --retry 3 -o "$dst.part" "https://huggingface.co/VAST-AI/SkinTokens/resolve/$REV/$REL"
got=$(shasum -a 256 "$dst.part" | cut -d' ' -f1)
if [ "$got" != "$SHA" ]; then echo "checksum mismatch: $got" >&2; rm -f "$dst.part"; exit 1; fi
mv "$dst.part" "$dst"
echo "weights: $dst"
