#!/bin/bash
#SBATCH --job-name=ar-baseline
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/baseline_%j.log
#SBATCH --requeue

set -euo pipefail

source /work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/.venv/bin/activate
cd /work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/autoresearch

echo "=============================================="
echo "BASELINE TRAIN.PY RUN"
echo "=============================================="
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================="

python train.py 2>&1

echo ""
echo "Baseline run complete"
