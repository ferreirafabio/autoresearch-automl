#!/bin/bash
#SBATCH --job-name=exp2-test-budget
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/exp2_test_budget_%j.log
#SBATCH --requeue

set -euo pipefail

# Test: verify time-budget enforcement with random seed 0
# random/seed_0 has ~1.58h (5688s) already. Set budget to 6300s (~1.75h)
# so it runs ~2-3 more trials then stops.

BACKEND="random"
SEED="0"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/exp2_benchmark/${BACKEND}/seed_${SEED}"
TRIALS=9999
TIME_BUDGET=6300  # ~1.75h — existing 1.58h + ~10min more

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

echo "=============================================="
echo "TEST: time-budget enforcement"
echo "=============================================="
echo "Backend:     $BACKEND"
echo "Seed:        $SEED"
echo "Time budget: ${TIME_BUDGET}s"
echo "Node:        $(hostname)"
echo "=============================================="

# Show current state before run
echo ""
echo "Current trials.jsonl line count:"
wc -l "$RESULTS_DIR/trials.jsonl" 2>/dev/null || echo "no file"
echo ""

python -m autoresearch_automl.cli run \
    --backend "$BACKEND" \
    --trials $TRIALS \
    --budget-max 300 \
    --seed "$SEED" \
    --results-dir "$RESULTS_DIR" \
    --resume \
    --time-budget $TIME_BUDGET

echo ""
echo "=============================================="
echo "After run trials.jsonl line count:"
wc -l "$RESULTS_DIR/trials.jsonl" 2>/dev/null || echo "no file"
echo "=============================================="
