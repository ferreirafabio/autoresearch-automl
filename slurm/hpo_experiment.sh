#!/bin/bash
#SBATCH --job-name=ar-automl-hpo
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/hpo_%j.log
#SBATCH --requeue

set -euo pipefail

source /work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/.venv/bin/activate
cd /work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl

# Point to self-hosted LLM (vLLM server running on another node)
export OPENAI_API_BASE="http://${VLLM_NODE}:8000/v1"
export OPENAI_API_KEY="dummy"

# Run HPO — override BACKEND, TRIALS, SEED via environment
python -m autoresearch_automl.cli run \
    --backend "${BACKEND:-optuna}" \
    --trials "${TRIALS:-100}" \
    --budget-max 300 \
    --seed "${SEED:-${SLURM_ARRAY_TASK_ID:-0}}" \
    --results-dir "/work/dlclarge1/ferreira-autoresearch-automl/results/${BACKEND:-optuna}/seed_${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
