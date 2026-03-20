#!/bin/bash
#SBATCH --job-name=vllm-qwen
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/vllm_%j.log
#SBATCH --requeue

set -euo pipefail

# Usage:
#   sbatch slurm/vllm_server.sh                           # default: Qwen3.5-9B
#   sbatch slurm/vllm_server.sh Qwen3.5-0.8B              # small
#   sbatch slurm/vllm_server.sh Qwen3.5-35B-A3B           # large (H200 only)
#   MODEL_DIR=/path/to/model sbatch slurm/vllm_server.sh   # custom path

MODELS_BASE="/work/dlclarge1/ferreira-autoresearch-automl/models"
MODEL_NAME="${1:-Qwen3.5-9B}"
MODEL_DIR="${MODEL_DIR:-${MODELS_BASE}/${MODEL_NAME}}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model not found at $MODEL_DIR"
    echo "Available models:"
    ls -1 "$MODELS_BASE/" 2>/dev/null || echo "  (no models directory)"
    exit 1
fi

source /work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/.venv/bin/activate

echo "=============================================="
echo "vLLM Server"
echo "=============================================="
echo "Model:     $MODEL_NAME"
echo "Path:      $MODEL_DIR"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=============================================="

vllm serve "$MODEL_DIR" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 32768
