#!/usr/bin/env bash
# One-time setup for the SOC-Agent local notebook.
#   bash setup_env.sh                 (environment in ~/soc-agent-env)
#   ENV_DIR=/path bash setup_env.sh   (elsewhere)
set -euo pipefail
ENV_DIR="${ENV_DIR:-$HOME/soc-agent-env}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== 1/4 NVIDIA driver"
if ! nvidia-smi; then
  echo "nvidia-smi failed: install the NVIDIA driver first, e.g.  sudo ubuntu-drivers install  , reboot, then re-run."
  exit 1
fi

echo "== 2/4 Python virtual environment: $ENV_DIR"
if ! python3 -m venv "$ENV_DIR" 2>/dev/null; then
  echo "python3-venv is missing:  sudo apt install python3-venv  (then re-run)"; exit 1
fi
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

echo "== 3/4 PyTorch (CUDA 12.4 build) + packages"
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r "$HERE/requirements.txt"
python -m ipykernel install --user --name soc-agent --display-name "SOC-Agent (local)"

echo "== 4/4 Check"
python - <<'PY'
import torch, bitsandbytes, transformers, xgboost
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name, f"{p.total_memory/1024**3:.1f} GB", "| bf16:", torch.cuda.is_bf16_supported())
    x = torch.randn(1024, 1024, device="cuda"); print("CUDA matmul OK:", float((x @ x).sum()) == float((x @ x).sum()))
print("transformers", transformers.__version__, "| bitsandbytes", bitsandbytes.__version__, "| xgboost", xgboost.__version__)
PY
echo
echo "Done. Next:  tmux new -s soc   then   bash $HERE/run_notebook.sh"
