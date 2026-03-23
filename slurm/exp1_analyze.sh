#!/bin/bash
set -euo pipefail

PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/exp1_model_size"
PLOTS_DIR="${PROJECT_DIR}/experiments/plots"

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

echo "Analyzing Experiment 1 results..."
python -m autoresearch_automl.cli analyze \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$PLOTS_DIR"

# Rename plots with exp1 prefix
for f in "${PLOTS_DIR}/convergence.png" "${PLOTS_DIR}/anytime.png" "${PLOTS_DIR}/pareto.png" "${PLOTS_DIR}/hp_importance.png"; do
    if [ -f "$f" ]; then
        base=$(basename "$f")
        mv "$f" "${PLOTS_DIR}/exp1_${base}"
    fi
done

echo "Plots saved to ${PLOTS_DIR}/"
ls -la "${PLOTS_DIR}/"
