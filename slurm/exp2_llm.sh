#!/bin/bash
#SBATCH --job-name=exp2-llm
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/zelaa/autoresearch-automl-private/logs/exp2_%x_%j.log
#SBATCH --requeue

set -euo pipefail

# Experiment 2: LLM-based backends (need vLLM with Qwen3.5-0.8B)
#
# Usage:
#   sbatch slurm/exp2_llm.sh llm_greedy 0
#   sbatch slurm/exp2_llm.sh llambo 0
#   sbatch slurm/exp2_llm.sh llambo 0 Qwen3.5-27B nothink   # thinking off comparison

BACKEND="${1:?Usage: sbatch exp2_llm.sh <backend> <seed> [model_name] [nothink]}"
SEED="${2:?Usage: sbatch exp2_llm.sh <backend> <seed> [model_name] [nothink]}"
MODELS_BASE="/work/dlclarge1/ferreira-autoresearch-automl/models"
MODEL_NAME="${3:-Qwen3.5-0.8B}"
THINKING_OVERRIDE="${4:-}"  # pass "nothink" to force thinking off
MODEL_DIR="${MODELS_BASE}/${MODEL_NAME}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
# Include model name in results dir when non-default model is specified
if [ "${MODEL_NAME}" = "Qwen3.5-0.8B" ]; then
    RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/exp2_benchmark/${BACKEND}/seed_${SEED}"
else
    MODEL_TAG=$(echo "${MODEL_NAME}" | tr '.' '_' | tr '-' '_')
    NOTHINK_SUFFIX=""
    if [ "${THINKING_OVERRIDE}" = "nothink" ]; then
        NOTHINK_SUFFIX="_nothink"
    fi
    RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/exp2_benchmark/${BACKEND}_${MODEL_TAG}${NOTHINK_SUFFIX}/seed_${SEED}"
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
USE_GEMINI_API="false"
case "${MODEL_NAME}" in
    gemini-*) USE_GEMINI_API="true"; AVAILABLE_VRAM="~140GB" ;;
    Qwen3.5-0.8B)  VLLM_GPU_UTIL=0.15; AVAILABLE_VRAM="~118GB" ;;
    Qwen3.5-4B)    VLLM_GPU_UTIL=0.15; AVAILABLE_VRAM="~112GB" ;;
    Qwen3.5-9B)    VLLM_GPU_UTIL=0.20; AVAILABLE_VRAM="~108GB"; VLLM_EXTRA_ARGS="--enable-prefix-caching" ;;
    Qwen3.5-27B)   VLLM_GPU_UTIL=0.45; AVAILABLE_VRAM="~76GB"; VLLM_EXTRA_ARGS="--enforce-eager --enable-prefix-caching" ;;
    Qwen3.5-35B*)  VLLM_GPU_UTIL=0.55; AVAILABLE_VRAM="~62GB"; VLLM_EXTRA_ARGS="--enforce-eager --enable-prefix-caching" ;;
    Qwen3-32B-AWQ) VLLM_GPU_UTIL=0.20; AVAILABLE_VRAM="~112GB" ;;
    *)             VLLM_GPU_UTIL=0.15; AVAILABLE_VRAM="~120GB" ;;
esac

if [ "$USE_GEMINI_API" = "true" ]; then
    if [ -z "${GEMINI_API_KEY:-}" ]; then
        echo "Error: GEMINI_API_KEY environment variable is not set."
        echo "       Get a key at https://aistudio.google.com/apikey and export it before submitting."
        exit 1
    fi
    LLM_MODEL_ARG="gemini-2.5-flash"
else
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Error: Model not found at $MODEL_DIR"
        exit 1
    fi
    LLM_MODEL_ARG="$MODEL_DIR"
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
echo "GPU Mem:   $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | tr '\n' ', ')"
echo "Results:   $RESULTS_DIR"
echo "VRAM Cap:  CUDA_MEM_FRACTION=0.543 (~76GB on H200)"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

if [ "$USE_GEMINI_API" = "false" ]; then
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

    export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
    export OPENAI_API_KEY="dummy"
    # Cap training subprocess to ~76GB (matches 27B vLLM gpu_util=0.45)
    export CUDA_MEM_FRACTION=0.543
else
    # Gemini API — no local server needed, full GPU is free for training
    export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
    export OPENAI_API_KEY="${GEMINI_API_KEY}"
    # Cap training to ~76GB for fair comparison (same as vLLM-based methods)
    export CUDA_MEM_FRACTION=0.543
fi

# Run HPO
export AVAILABLE_VRAM="${AVAILABLE_VRAM}"

# Verify VRAM cap is set
echo ""
echo "VRAM cap verification:"
echo "  CUDA_MEM_FRACTION=${CUDA_MEM_FRACTION}"
echo "  AVAILABLE_VRAM=${AVAILABLE_VRAM}"
echo "  USE_GEMINI_API=${USE_GEMINI_API}"
python3 -c "
import torch, os
frac = float(os.environ.get('CUDA_MEM_FRACTION', '1.0'))
if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU total: {total:.1f} GB, fraction: {frac}, effective cap: {total*frac:.1f} GB')
    torch.cuda.set_per_process_memory_fraction(frac)
    print(f'  VRAM cap applied successfully')
else:
    print(f'  No GPU available (fraction={frac})')
" 2>&1 || echo "  WARNING: Could not verify VRAM cap"
echo ""
echo "Starting HPO with ${BACKEND} backend..."
python -m autoresearch_automl.cli run \
    --backend "$BACKEND" \
    --trials $TRIALS \
    --budget-max 300 \
    --seed "$SEED" \
    --llm-model "$LLM_MODEL_ARG" \
    --results-dir "$RESULTS_DIR" \
    --resume \
    --time-budget 86400

echo ""
echo "=============================================="
echo "Experiment 2 complete: ${BACKEND} seed ${SEED}"
echo "=============================================="

# Cleanup vLLM
if [ "$USE_GEMINI_API" = "false" ]; then
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null || true
fi
