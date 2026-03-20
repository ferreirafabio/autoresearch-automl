#!/bin/bash
#SBATCH --job-name=test-vram-cap
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/test_vram_cap_%j.log

set -euo pipefail

PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

echo "=== VRAM Cap Test ==="
python scripts/test_vram_cap.py
echo "=== Done ==="
