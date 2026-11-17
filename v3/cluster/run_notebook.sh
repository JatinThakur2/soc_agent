#!/usr/bin/env bash
# Runs the whole notebook headless (use inside tmux). Safe to re-run: cached
# steps are loaded and finished agent runs are skipped.
set -uo pipefail
ENV_DIR="${ENV_DIR:-$HOME/soc-agent-env}"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$ENV_DIR/bin/activate"
# API keys (HF_TOKEN for gated Llama, Kaggle for dataset downloads)
[ -f "$HOME/.soc_secrets" ] && source "$HOME/.soc_secrets"
cd "$HERE"
STAMP=$(date +%Y%m%d_%H%M)
mkdir -p logs
echo "Running SOC_Agent_Local_v3.ipynb (${SOC_LLM_MODEL:-auto}) — log: $HERE/logs/run_$STAMP.log"
echo "Cell outputs appear in the executed notebook when it finishes. To watch progress meanwhile:"
echo "  tail -f ${SOC_AGENT_HOME:-$HOME/SOC_Agent}/workers/worker_0.log   (LLM agent runs)"
echo "  nvidia-smi -l 5                                                  (GPU use)"
TAG="${SOC_LLM_MODEL:-auto}"; TAG="${TAG##*/}"
# papermill saves the output notebook after every cell (open it in Jupyter and
# reload to follow progress) and streams cell output into the log.
papermill SOC_Agent_Local_v3.ipynb "SOC_Agent_Local_v3.executed_${TAG}_$STAMP.ipynb" \
  -k soc-agent --log-output --request-save-on-cell-execute --autosave-cell-every 30 \
  --progress-bar 2>&1 | tee "logs/run_${TAG}_$STAMP.log"
echo "Finished. Executed notebook: $HERE/SOC_Agent_Local_v3.executed_${TAG}_$STAMP.ipynb"
echo "Results: ${SOC_AGENT_HOME:-$HOME/SOC_Agent}/results/"
