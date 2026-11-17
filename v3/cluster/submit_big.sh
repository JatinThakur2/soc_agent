#!/usr/bin/env bash
# Queues the larger models after an existing job (default: the last job of
# submit_all.sh). 70B+ models get two chained jobs because one 24 h job is
# probably not enough; the second resumes where the first stopped.
#   ./submit_big.sh <job id to wait for>
cd "$(dirname "$0")"
prev="$1"
BIG=(
  "mistralai/Mistral-Small-24B-Instruct-2501|8|1"
  "Qwen/Qwen2.5-72B-Instruct|4|2"
  "meta-llama/Llama-3.3-70B-Instruct|4|2"     # gated: accept the licence on huggingface.co
)
for entry in "${BIG[@]}"; do
  IFS='|' read -r model batch parts <<< "$entry"
  tag="${model##*/}"
  for part in $(seq 1 "$parts"); do
    prev=$(qsub -N "soc_${tag:0:10}" -o "logs/pbs_${tag}_part${part}.log" \
         -v MODEL="$model",V3ONLY=1,BATCH="$batch" \
         ${prev:+-W depend=afterany:$prev} soc_job.pbs)
    echo "$model (part $part) -> $prev"
  done
done
