#!/bin/bash
#SBATCH --job-name=exp2-llm
#SBATCH --partition=accelerated-h200
#SBATCH --account=hk-project-p0024002
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/fr_ff1042-autoresearch/logs/exp2_%x_%j.log
#SBATCH --requeue

set -euo pipefail

# Experiment 2: LLM-based backends (need vLLM with Qwen3.5-0.8B)
#
# Usage:
#   sbatch slurm/exp2_llm_horeka.sh llm_greedy 0
#   sbatch slurm/exp2_llm_horeka.sh llambo 0
#   sbatch slurm/exp2_llm_horeka.sh llambo 0 Qwen3.5-27B nothink   # thinking off comparison

BACKEND="${1:?Usage: sbatch exp2_llm_horeka.sh <backend> <seed> [model_name] [nothink]}"
SEED="${2:?Usage: sbatch exp2_llm_horeka.sh <backend> <seed> [model_name] [nothink]}"
MODELS_BASE="/hkfs/work/workspace/scratch/fr_ff1042-autoresearch/models"
MODEL_NAME="${3:-Qwen3.5-0.8B}"
THINKING_OVERRIDE="${4:-}"  # pass "nothink" to force thinking off
MODEL_DIR="${MODELS_BASE}/${MODEL_NAME}"
PROJECT_DIR="/hkfs/work/workspace/scratch/fr_ff1042-autoresearch/autoresearch-automl"
# Include model name in results dir when non-default model is specified
if [ "${MODEL_NAME}" = "Qwen3.5-0.8B" ]; then
    RESULTS_DIR="/hkfs/work/workspace/scratch/fr_ff1042-autoresearch/results/exp2_benchmark/${BACKEND}/seed_${SEED}"
else
    MODEL_TAG=$(echo "${MODEL_NAME}" | tr '.' '_' | tr '-' '_')
    NOTHINK_SUFFIX=""
    if [ "${THINKING_OVERRIDE}" = "nothink" ]; then
        NOTHINK_SUFFIX="_nothink"
    fi
    RESULTS_DIR="/hkfs/work/workspace/scratch/fr_ff1042-autoresearch/results/exp2_benchmark/${BACKEND}_${MODEL_TAG}${NOTHINK_SUFFIX}/seed_${SEED}"
fi
TRIALS=9999
VLLM_PORT=$((8100 + RANDOM % 900))

# Adjust vLLM GPU memory, available VRAM, and extra flags based on model size
# Values measured via slurm/test_vram_usage.sh on H200 (140 GiB)
# Small models (<= 4B) can't do proper thinking (infinite <think> loop, never produce answer)
# Large models (>= 9B) produce proper <think>...</think> + answer with reasoning parser
VLLM_EXTRA_ARGS=""
# Thinking disabled for all models — thinking traces waste tokens on structured HP output
ENABLE_THINKING="false"
case "${MODEL_NAME}" in
    Qwen3.5-0.8B)  VLLM_GPU_UTIL=0.15; AVAILABLE_VRAM="~118GB" ;;
    Qwen3.5-4B)    VLLM_GPU_UTIL=0.15; AVAILABLE_VRAM="~112GB" ;;
    Qwen3.5-9B)    VLLM_GPU_UTIL=0.20; AVAILABLE_VRAM="~108GB"; VLLM_EXTRA_ARGS="--enable-prefix-caching" ;;
    Qwen3.5-27B)   VLLM_GPU_UTIL=0.45; AVAILABLE_VRAM="~76GB"; VLLM_EXTRA_ARGS="--enforce-eager --enable-prefix-caching" ;;
    Qwen3.5-35B*)  VLLM_GPU_UTIL=0.55; AVAILABLE_VRAM="~62GB"; VLLM_EXTRA_ARGS="--enforce-eager --enable-prefix-caching" ;;
    Qwen3-32B-AWQ) VLLM_GPU_UTIL=0.20; AVAILABLE_VRAM="~112GB" ;;
    *)             VLLM_GPU_UTIL=0.15; AVAILABLE_VRAM="~120GB" ;;
esac

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model not found at $MODEL_DIR"
    exit 1
fi

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

# Bypass cluster HTTP proxy for localhost connections
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

echo "=============================================="
echo "Experiment 2: ${BACKEND} (seed ${SEED})"
echo "=============================================="
echo "Backend:   $BACKEND"
echo "Seed:      $SEED"
echo "Trials:    $TRIALS"
echo "Model:     $MODEL_NAME"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo "Results:   $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

# Start vLLM server in background
REASONING_ARGS=""
if [ "$ENABLE_THINKING" = "true" ]; then
    REASONING_ARGS="--reasoning-parser qwen3 --default-chat-template-kwargs {\"enable_thinking\":true}"
else
    # Explicitly disable thinking for models that support it (Qwen3.5-9B+)
    # Without this, large models still "think" in plain text inside content
    REASONING_ARGS="--default-chat-template-kwargs {\"enable_thinking\":false}"
fi

echo "Starting vLLM server (port ${VLLM_PORT}, gpu_util=${VLLM_GPU_UTIL}, thinking=${ENABLE_THINKING})..."
vllm serve "$MODEL_DIR" \
    --host 127.0.0.1 --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization $VLLM_GPU_UTIL \
    $REASONING_ARGS \
    $VLLM_EXTRA_ARGS &
VLLM_PID=$!

# Wait for vLLM to be ready
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
    echo "vLLM server failed to start within 600s (status: $STATUS)"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Run HPO
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_API_KEY="dummy"
export AVAILABLE_VRAM="${AVAILABLE_VRAM}"

echo ""
echo "Starting HPO with ${BACKEND} backend..."
python -m autoresearch_automl.cli run \
    --backend "$BACKEND" \
    --trials $TRIALS \
    --budget-max 300 \
    --seed "$SEED" \
    --llm-model "$MODEL_DIR" \
    --results-dir "$RESULTS_DIR" \
    --resume \
    --time-budget 86400

echo ""
echo "=============================================="
echo "Experiment 2 complete: ${BACKEND} seed ${SEED}"
echo "=============================================="

# Cleanup vLLM
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
