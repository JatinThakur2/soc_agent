#!/usr/bin/env bash
# Submits one PBS job per model, chained so they run one after another
# (same order as run_all_models.sh). Each job resumes if re-run.
cd "$(dirname "$0")"
mkdir -p logs
PRIMARY="Qwen/Qwen2.5-14B-Instruct"
OTHERS=(
  "microsoft/phi-4"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "microsoft/Phi-4-mini-instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-32B-Instruct"
)
submit () {   # $1 model  $2 v3only  $3 dependency job id (optional)
  local model="$1" v3="$2" dep="$3" batch="" tag="${1##*/}"
  [[ "$model" == *32B* ]] && batch=4
  qsub -N "soc_${tag:0:10}" -o "logs/pbs_${tag}.log" \
       -v MODEL="$model",V3ONLY="$v3",BATCH="$batch" \
       ${dep:+-W depend=afterany:$dep} soc_job.pbs
}
prev=$(submit "$PRIMARY" 0 "$1"); echo "$PRIMARY -> $prev"
for m in "${OTHERS[@]}"; do
  prev=$(submit "$m" 1 "$prev"); echo "$m -> $prev"
done
echo "Check with: qstat -u \$USER"
