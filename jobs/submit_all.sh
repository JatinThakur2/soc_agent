#!/bin/bash
# Submit all SOC-Agent jobs with dependencies
# Usage: bash jobs/submit_all.sh

set -e
cd /home/deepak.g/Downloads/soc_agent

echo "Submitting SOC-Agent pipeline jobs..."

J1=$(qsub jobs/01_engineer.pbs)
echo "  [1] Feature engineering : $J1"

J2=$(qsub -W depend=afterok:$J1 jobs/02_train.pbs)
echo "  [2] Model training      : $J2"

J3=$(qsub -W depend=afterok:$J2 jobs/03_build_kb.pbs)
echo "  [3] Build knowledge base: $J3"

J4=$(qsub -W depend=afterok:$J3 jobs/04_pipeline.pbs)
echo "  [4] Agent pipeline      : $J4"

J5=$(qsub -W depend=afterok:$J4 jobs/05_evaluate.pbs)
echo "  [5] Evaluation          : $J5"

echo ""
echo "All jobs queued. Monitor with:"
echo "  qstat -u $USER"
echo ""
echo "Tail live logs with:"
echo "  tail -f jobs/logs/01_engineer.out"
