#!/bin/bash
# Monitor exp2 job progress: check running jobs and training time status
set -euo pipefail

PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
cd "$PROJECT_DIR"

echo "=== $(date) ==="
echo ""
echo "--- Running jobs ---"
squeue -u ferreira -o "%.10i %.20j %.2t %.10M %.6D %R" 2>/dev/null || echo "squeue failed"
echo ""
echo "--- Training time status ---"
.venv/bin/python scripts/analyze_time_budgets.py
