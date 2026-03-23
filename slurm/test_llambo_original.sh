#!/bin/bash
#SBATCH --job-name=test-llambo-orig
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/test_llambo_orig_%j.log

set -euo pipefail

# Quick test of llambo_original backend with Qwen3.5-0.8B
# 1h time limit, seed 0, uses same setup as exp2_llm.sh

BACKEND="llambo_original"
SEED=0
MODEL_NAME="Qwen3.5-0.8B"
MODELS_BASE="/work/dlclarge1/ferreira-autoresearch-automl/models"
MODEL_DIR="${MODELS_BASE}/${MODEL_NAME}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/test_llambo_original/seed_${SEED}"
TRIALS=9999
VLLM_PORT=$((8100 + RANDOM % 900))
VLLM_GPU_UTIL=0.15
AVAILABLE_VRAM="~118GB"

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

echo "=============================================="
echo "Test: llambo_original (Qwen3.5-0.8B, seed 0)"
echo "=============================================="
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo "Results:   $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

# Start vLLM (0.8B = no thinking)
echo "Starting vLLM server (port ${VLLM_PORT})..."
vllm serve "$MODEL_DIR" \
    --host 127.0.0.1 --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization $VLLM_GPU_UTIL \
    --default-chat-template-kwargs '{"enable_thinking":false}' &
VLLM_PID=$!

echo "Waiting for vLLM server to start..."
for i in $(seq 1 600); do
    STATUS=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM server died unexpectedly"
        exit 1
    fi
    sleep 1
done

STATUS=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
if [ "$STATUS" != "200" ]; then
    echo "vLLM server failed to start within 600s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_API_KEY="dummy"
export AVAILABLE_VRAM="${AVAILABLE_VRAM}"

echo ""
echo "Starting HPO with llambo_original backend..."
python -m autoresearch_automl.cli run \
    --backend "$BACKEND" \
    --trials $TRIALS \
    --budget-max 300 \
    --seed "$SEED" \
    --llm-model "$MODEL_DIR" \
    --results-dir "$RESULTS_DIR" \
    --resume

echo ""
echo "Test complete!"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
