#!/bin/bash
#SBATCH --job-name=exp1-hpo
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/exp1_%j.log
#SBATCH --requeue

set -euo pipefail

# Experiment 1: LLM Model Size Comparison for HPO
# Runs vLLM server + HPO on the same node (2 GPUs: 1 for vLLM, 1 for training)
#
# Usage:
#   sbatch slurm/exp1_vllm_and_hpo.sh Qwen3.5-0.8B    # small
#   sbatch slurm/exp1_vllm_and_hpo.sh Qwen3.5-9B       # medium
#   sbatch slurm/exp1_vllm_and_hpo.sh Qwen3.5-35B-A3B  # large

MODELS_BASE="/work/dlclarge1/ferreira-autoresearch-automl/models"
MODEL_NAME="${1:?Usage: sbatch exp1_vllm_and_hpo.sh <model-name>}"
MODEL_DIR="${MODELS_BASE}/${MODEL_NAME}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/exp1_model_size/${MODEL_NAME}"
TRIALS=30
VLLM_PORT=8000

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model not found at $MODEL_DIR"
    exit 1
fi

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

echo "=============================================="
echo "Experiment 1: LLM Model Size Comparison"
echo "=============================================="
echo "Model:     $MODEL_NAME"
echo "Trials:    $TRIALS"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo "Results:   $RESULTS_DIR"
echo "=============================================="

# Start vLLM server in background (shares GPU with training)
echo "Starting vLLM server..."
vllm serve "$MODEL_DIR" \
    --host 127.0.0.1 --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 4096 &
VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for vLLM server to start..."
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server died unexpectedly"
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
    echo "vLLM server failed to start within 120s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Run HPO (shares GPU with vLLM — they alternate, not concurrent)
export OPENAI_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_API_KEY="dummy"

echo ""
echo "Starting HPO with llm_greedy backend..."
python -m autoresearch_automl.cli run \
    --backend llm_greedy \
    --trials $TRIALS \
    --budget-max 300 \
    --seed 0 \
    --llm-model "$MODEL_NAME" \
    --results-dir "$RESULTS_DIR" \
    --resume

echo ""
echo "=============================================="
echo "Experiment 1 complete: $MODEL_NAME"
echo "=============================================="

# Cleanup vLLM
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
