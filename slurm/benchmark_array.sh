#!/bin/bash
#SBATCH --job-name=ar-benchmark
#SBATCH --partition=alldlc2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --time=12:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/bench_%A_%a.log

source /work/dlclarge1/ferreira-autoresearch-automl/venvs/ar-automl/bin/activate
cd /work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl

export OPENAI_API_BASE="http://${VLLM_NODE}:8000/v1"
export OPENAI_API_KEY="dummy"

python -m autoresearch_automl.cli benchmark \
    --method "${METHOD}" \
    --scenario "${SCENARIO}" \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --results-dir "/work/dlclarge1/ferreira-autoresearch-automl/results"
