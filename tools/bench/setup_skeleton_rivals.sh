#!/usr/bin/env bash
# Set up Apache-2.0 skeleton rivals (Puppeteer + MagicArticulate) in /data/rigwork on GPU host.
# Strictly excludes Puppeteer's skinning stage and NVIDIA PartField (non-commercial).
set -euo pipefail

PUPPETEER_DIR="${PUPPETEER_DIR:-/data/rigwork/puppeteer}"
VENV_DIR="${VENV_DIR:-/data/rigwork/venv-puppeteer}"
UNIRIG_VENV="${UNIRIG_VENV:-/data/rigwork/venv-unirig}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$(dirname "$PUPPETEER_DIR")"

# 1. Clone Seed3D/Puppeteer (Apache-2.0) WITHOUT recursive submodules, then init ONLY Michelangelo.
if [ ! -d "$PUPPETEER_DIR/.git" ]; then
  git clone --depth 1 https://github.com/Seed3D/Puppeteer.git "$PUPPETEER_DIR"
fi
git -C "$PUPPETEER_DIR" submodule update --init --depth 1 skeleton/third_partys/Michelangelo

# Double-check PartField was NOT fetched
if [ -d "$PUPPETEER_DIR/skinning/third_partys/PartField/partfield" ]; then
  echo "ERROR: non-commercial PartField submodule present; aborting" >&2
  exit 1
fi

# 2. Prepare Python venv (isolated from venv-unirig)
if [ ! -x "$VENV_DIR/bin/python" ]; then
  PY_BIN="$(command -v python3.10 || command -v python3.11 || command -v python3)"
  "$PY_BIN" -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  "$PIP" install --upgrade pip wheel setuptools
  "$PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

# Minimal skeleton-only dependencies (zero flash-attn, zero nvdiffrast, zero PartField, zero pyrender)
"$PIP" install -q \
  "numpy<2" \
  "transformers==4.46.1" \
  "accelerate>=0.28.0" \
  "trimesh>=4.2.0" \
  "scipy" \
  "scikit-image" \
  "omegaconf>=2.3.0" \
  "einops>=0.7.0" \
  "huggingface_hub" \
  "tqdm" \
  "opencv-python-headless"

# Optional mesh2sdf for --apply_marching_cubes (non-fatal if no wheel/compiler)
if ! "$PY" -c "import mesh2sdf.core" 2>/dev/null; then
  "$PIP" install -q mesh2sdf || echo "WARN: mesh2sdf install failed; will fall back to --no-marching-cubes"
fi

# 3. Download ONLY Apache-2.0 skeleton weights + LGPL-3.0 Michelangelo shape encoder
"$PY" - "$PUPPETEER_DIR" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download

root = sys.argv[1]
skel_dir = os.path.join(root, "skeleton")

# Michelangelo shape encoder (LGPL-3.0 weights)
hf_hub_download(
    repo_id="Maikou/Michelangelo",
    filename="checkpoints/aligned_shape_latents/shapevae-256.ckpt",
    local_dir=os.path.join(skel_dir, "third_partys", "Michelangelo"),
)

# Puppeteer skeleton checkpoint (Apache-2.0)
hf_hub_download(
    repo_id="Seed3D/Puppeteer",
    filename="skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth",
    local_dir=skel_dir,
)

# MagicArticulate skeleton checkpoint (Apache-2.0)
ma_path = hf_hub_download(
    repo_id="Seed3D/MagicArticulate",
    filename="skeleton_ckpt/checkpoint_trainonv2_hier.pth",
    local_dir=skel_dir,
)
dst = os.path.join(skel_dir, "skeleton_ckpts", "checkpoint_trainonv2_hier.pth")
os.makedirs(os.path.dirname(dst), exist_ok=True)
if not os.path.exists(dst):
    os.symlink(os.path.abspath(ma_path), dst)

# Pre-cache facebook/opt-350m config.json only (architecture config, no weights downloaded)
from transformers import OPTConfig
OPTConfig.from_pretrained("facebook/opt-350m")
print("OK all skeleton checkpoints & OPT config ready")
PY

# 4. Patch Puppeteer skeleton code for SDPA (no flash-attn) + PyTorch 2.6 weights_only=False
"$PY" - "$REPO_ROOT" "$PUPPETEER_DIR" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(sys.argv[1], "tools", "bench"))
import run_skeleton_rivals as rsr
changed = rsr.patch_puppeteer_sdpa(os.path.join(sys.argv[2], "skeleton"))
print("Patched files:", changed)
PY

"$PY" -c "
import torch, transformers, accelerate, trimesh, skimage, omegaconf, einops
try:
    import mesh2sdf.core
    m2s = True
except Exception as e:
    m2s = f'missing ({e})'
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'transformers:', transformers.__version__, 'mesh2sdf:', m2s)
"
