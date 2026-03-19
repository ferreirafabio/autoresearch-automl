#!/bin/bash
#SBATCH --job-name=exp2-centaur
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/exp2_%x_%j.log
#SBATCH --requeue

set -euo pipefail

# Experiment 2: CMA-ES + LLM — CMA-ES guided LLM optimization
#
# Usage:
#   sbatch slurm/exp2_centaur.sh 0 Qwen3.5-0.8B
#   sbatch slurm/exp2_centaur.sh 0 Qwen3.5-27B
#   sbatch slurm/exp2_centaur.sh 0 Qwen3.5-27B withC   # append suffix to results dir

SEED="${1:?Usage: sbatch exp2_centaur.sh <seed> <model_name> [suffix]}"
MODEL_NAME="${2:?Usage: sbatch exp2_centaur.sh <seed> <model_name> [suffix]}"
SUFFIX="${3:-}"
MODELS_BASE="/work/dlclarge1/ferreira-autoresearch-automl/models"
MODEL_DIR="${MODELS_BASE}/${MODEL_NAME}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"

MODEL_TAG=$(echo "${MODEL_NAME}" | tr '.' '_' | tr '-' '_')
SUFFIX_TAG=""
if [ -n "$SUFFIX" ]; then
    SUFFIX_TAG="_${SUFFIX}"
fi
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/exp2_benchmark/centaur_${MODEL_TAG}${SUFFIX_TAG}/seed_${SEED}"
TRIALS=9999
VLLM_PORT=$((8100 + RANDOM % 900))

# Adjust vLLM GPU memory based on model size
VLLM_EXTRA_ARGS=""
ENABLE_THINKING="false"
case "${MODEL_NAME}" in
    Qwen3.5-0.8B)  VLLM_GPU_UTIL=0.15; AVAILABLE_VRAM="~118GB" ;;
    Qwen3.5-4B)    VLLM_GPU_UTIL=0.15; AVAILABLE_VRAM="~112GB" ;;
    Qwen3.5-9B)    VLLM_GPU_UTIL=0.20; AVAILABLE_VRAM="~108GB"; VLLM_EXTRA_ARGS="--enable-prefix-caching" ;;
    Qwen3.5-27B)   VLLM_GPU_UTIL=0.45; AVAILABLE_VRAM="~76GB"; VLLM_EXTRA_ARGS="--enforce-eager --enable-prefix-caching" ;;
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
echo "Experiment 2: centaur (seed ${SEED})"
echo "=============================================="
echo "Backend:   centaur"
echo "Seed:      $SEED"
echo "Trials:    $TRIALS"
echo "Model:     $MODEL_NAME"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo "Results:   $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

# Start vLLM server in background
REASONING_ARGS="--default-chat-template-kwargs {\"enable_thinking\":false}"

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
# Cap training subprocess to ~76GB (matches 27B vLLM gpu_util=0.45)
export CUDA_MEM_FRACTION=0.543

echo ""
echo "Starting HPO with centaur backend..."
python -m autoresearch_automl.cli run \
    --backend "centaur" \
    --trials $TRIALS \
    --budget-max 300 \
    --seed "$SEED" \
    --llm-model "$MODEL_DIR" \
    --results-dir "$RESULTS_DIR" \
    --resume \
    --time-budget 86400

echo ""
echo "=============================================="
echo "Experiment 2 complete: centaur seed ${SEED}"
echo "=============================================="

# Cleanup vLLM
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
