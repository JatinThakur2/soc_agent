#!/usr/bin/env bash
# Cluster version of setup_env.sh: Python 3.11 from the shared conda install,
# PyTorch cu121 (matches the 525 driver on jiit-gpu01). Run once on the login node.
set -euo pipefail
ENV_DIR="${ENV_DIR:-$HOME/soc-agent-env}"
HERE="$(cd "$(dirname "$0")" && pwd)"
/cm/shared/apps/anaconda3/envs/py311/bin/python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r "$HERE/requirements.txt"
python -m ipykernel install --user --name soc-agent --display-name "SOC-Agent (local)"
python -c "import torch, transformers, xgboost, bitsandbytes; print('torch', torch.__version__, '| transformers', transformers.__version__)"
echo "ENV OK"
