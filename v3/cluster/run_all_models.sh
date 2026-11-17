#!/usr/bin/env bash
# Runs the notebook once per LLM so every model is tested on the SAME alerts and
# the SAME controlled conditions (v3). Use inside tmux. Safe to re-run: finished
# alerts are skipped, so an interrupted model resumes where it stopped.
#
# The first model also extends the original v2 runs and runs the full four-agent
# pipeline; the others run the v3 control grid only (SOC_V3_ONLY=1).
# Gated models (Llama) need HF_TOKEN and the licence accepted on huggingface.co;
# if a model cannot be loaded the script logs it and moves on.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PRIMARY="Qwen/Qwen2.5-14B-Instruct"
OTHERS=(
  "microsoft/phi-4"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "microsoft/Phi-4-mini-instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"          # slowest (batch 4); remove this line to skip
)
run_one () {
  local model="$1" v3only="$2"
  local batch=""
  [[ "$model" == *32B* ]] && batch=4
  echo "================ $model (v3 only: $v3only) ================"
  SOC_LLM_MODEL="$model" SOC_V3_ONLY="$v3only" SOC_LLM_BATCH="$batch" bash "$HERE/run_notebook.sh" \
    || echo "!!! $model failed — see the log above; continuing with the next model"
}
run_one "$PRIMARY" 0
for m in "${OTHERS[@]}"; do run_one "$m" 1; done
echo "All models done. The cross-model tables are in ~/SOC_Agent/results/tables/v3/ (models_*.csv);"
echo "the last notebook run (Cell 16f) compares every model on the common alerts."
