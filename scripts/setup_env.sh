#!/usr/bin/env bash
# One-shot environment setup for OV-CBMR on a CUDA GPU box. Idempotent.
#
#   bash scripts/setup_env.sh [cuda_tag]     # cuda_tag: cu121 (default) | cu118 | cu124 | cpu
#
# Run from the repo root (the folder containing pyproject.toml).
set -euo pipefail

CUDA_TAG="${1:-cu121}"
ENV_NAME="ovcbmr"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] conda not found. Install Miniconda first, or create a venv and skip to pip steps." >&2
  exit 1
fi

echo "[setup] conda env: $ENV_NAME (python 3.10)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install --upgrade pip

echo "[setup] torch (CUDA tag: $CUDA_TAG)"
if [ "$CUDA_TAG" = "cpu" ]; then
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
else
  pip install torch==2.4.1 torchvision==0.19.1 --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

echo "[setup] project + requirements"
pip install -r requirements.txt
pip install -e .

echo "[setup] verify"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "from ovcbmr.models.encoder import load_biomedclip; print('encoder import: ok')"
python tests/test_smoke.py

echo "[setup] DONE. Next: bash scripts/download_data.sh   (see SERVER_SETUP.md §6)"
