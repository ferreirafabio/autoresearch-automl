#!/bin/bash
#SBATCH --job-name=exp2-classical
#SBATCH --partition=accelerated-h200
#SBATCH --account=hk-project-p0024002
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/fr_ff1042-autoresearch/logs/exp2_%x_%j.log
#SBATCH --requeue

set -euo pipefail

# Experiment 2: Classical AutoML backends (no vLLM needed) — HoreKa
#
# Usage:
#   sbatch slurm/exp2_classical_horeka.sh random 0
#   sbatch slurm/exp2_classical_horeka.sh optuna 0
#   sbatch slurm/exp2_classical_horeka.sh smac 0
#   sbatch slurm/exp2_classical_horeka.sh cma_es 0

BACKEND="${1:?Usage: sbatch exp2_classical_horeka.sh <backend> <seed>}"
SEED="${2:?Usage: sbatch exp2_classical_horeka.sh <backend> <seed>}"
PROJECT_DIR="/hkfs/work/workspace/scratch/fr_ff1042-autoresearch/autoresearch-automl"
RESULTS_DIR="/hkfs/work/workspace/scratch/fr_ff1042-autoresearch/results/exp2_benchmark/${BACKEND}/seed_${SEED}"
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

# Cap GPU memory to 80GB (matches H100 VRAM, ensures fair comparison across all backends)
export CUDA_MEM_FRACTION=0.57

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
