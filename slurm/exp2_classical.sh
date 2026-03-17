#!/bin/bash
#SBATCH --job-name=exp2-classical
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/exp2_%x_%j.log
#SBATCH --requeue

set -euo pipefail

# Experiment 2: Classical AutoML backends (no vLLM needed)
#
# Usage:
#   sbatch slurm/exp2_classical.sh random 0
#   sbatch slurm/exp2_classical.sh optuna 0
#   sbatch slurm/exp2_classical.sh smac 0
#   sbatch slurm/exp2_classical.sh dehb 0
#   sbatch slurm/exp2_classical.sh bohb 0
#   sbatch slurm/exp2_classical.sh cma_es 0

BACKEND="${1:?Usage: sbatch exp2_classical.sh <backend> <seed>}"
SEED="${2:?Usage: sbatch exp2_classical.sh <backend> <seed>}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/exp2_benchmark/${BACKEND}/seed_${SEED}"
TRIALS=9999

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

echo "=============================================="
echo "Experiment 2: ${BACKEND} (seed ${SEED})"
echo "=============================================="
echo "Backend:   $BACKEND"
echo "Seed:      $SEED"
echo "Trials:    $TRIALS"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo "Results:   $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

python -m autoresearch_automl.cli run \
    --backend "$BACKEND" \
    --trials $TRIALS \
    --budget-max 300 \
    --seed "$SEED" \
    --results-dir "$RESULTS_DIR" \
    --resume \
    --time-budget 86400

echo ""
echo "=============================================="
echo "Experiment 2 complete: ${BACKEND} seed ${SEED}"
echo "=============================================="
