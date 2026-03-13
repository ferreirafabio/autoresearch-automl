#!/bin/bash
#SBATCH --job-name=ar-benchmark
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --array=0-4
#SBATCH --time=12:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/bench_%A_%a.log
#SBATCH --requeue

set -euo pipefail

source /work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/.venv/bin/activate
cd /work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl

export OPENAI_API_BASE="http://${VLLM_NODE}:8000/v1"
export OPENAI_API_KEY="dummy"

python -m autoresearch_automl.cli benchmark \
    --method "${METHOD}" \
    --scenario "${SCENARIO}" \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --results-dir "/work/dlclarge1/ferreira-autoresearch-automl/results"
