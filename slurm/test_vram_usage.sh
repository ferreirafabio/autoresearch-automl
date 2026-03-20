#!/bin/bash
#SBATCH --job-name=test-vram
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/test_vram_%j.log

set -euo pipefail

# Measure actual VRAM usage of vLLM models at their configured gpu-memory-utilization.
# Usage: sbatch slurm/test_vram_usage.sh <model_name>
#   e.g. sbatch slurm/test_vram_usage.sh Qwen3.5-0.8B
#        sbatch slurm/test_vram_usage.sh Qwen3.5-35B-A3B

MODEL_NAME="${1:?Usage: sbatch test_vram_usage.sh <model_name>}"
MODELS_BASE="/work/dlclarge1/ferreira-autoresearch-automl/models"
MODEL_DIR="${MODELS_BASE}/${MODEL_NAME}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
VLLM_PORT=$((8100 + RANDOM % 900))

# Use same gpu-memory-utilization as exp2_llm.sh
VLLM_EXTRA_ARGS=""
case "${MODEL_NAME}" in
    Qwen3.5-0.8B)  VLLM_GPU_UTIL=0.15 ;;
    Qwen3.5-4B)    VLLM_GPU_UTIL=0.15 ;;
    Qwen3.5-9B)    VLLM_GPU_UTIL=0.20 ;;
    Qwen3.5-27B)   VLLM_GPU_UTIL=0.45; VLLM_EXTRA_ARGS="--enforce-eager" ;;
    Qwen3.5-35B*)  VLLM_GPU_UTIL=0.55; VLLM_EXTRA_ARGS="--enforce-eager" ;;
    Qwen3-32B-AWQ) VLLM_GPU_UTIL=0.20 ;;
    *)             VLLM_GPU_UTIL=0.15 ;;
esac

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model not found at $MODEL_DIR"
    exit 1
fi

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

echo "=============================================="
echo "VRAM Usage Test: ${MODEL_NAME}"
echo "=============================================="
echo "Model:           $MODEL_NAME"
echo "GPU util:        $VLLM_GPU_UTIL"
echo "Node:            $(hostname)"
echo "GPU:             $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "vLLM port:       $VLLM_PORT"
echo "=============================================="

# Determine which physical GPU we got from Slurm
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"  # first GPU if multiple
if [ -z "$GPU_ID" ]; then GPU_ID=0; fi

# Baseline: VRAM before vLLM
echo ""
echo "=== GPU memory BEFORE vLLM (physical GPU ${GPU_ID}) ==="
nvidia-smi --id="$GPU_ID" --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
VRAM_BEFORE=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
echo "Used: ${VRAM_BEFORE} MiB"

# Start vLLM
echo ""
echo "=== Starting vLLM server (gpu_util=${VLLM_GPU_UTIL}) ==="
vllm serve "$MODEL_DIR" \
    --host 127.0.0.1 --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization $VLLM_GPU_UTIL \
    $VLLM_EXTRA_ARGS &
VLLM_PID=$!

# Wait for vLLM to be ready
for i in $(seq 1 600); do
    STATUS=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo "vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM died unexpectedly"
        exit 1
    fi
    sleep 1
done

STATUS=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
if [ "$STATUS" != "200" ]; then
    echo "vLLM failed to start within 600s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Measure VRAM after vLLM is loaded (idle, no requests)
echo ""
echo "=== GPU memory AFTER vLLM loaded (idle) ==="
nvidia-smi --id="$GPU_ID" --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
VRAM_AFTER_IDLE=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
echo "Used: ${VRAM_AFTER_IDLE} MiB"

# Send a warmup request to populate KV cache
echo ""
echo "=== Sending warmup request ==="
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_API_KEY="dummy"

curl --noproxy '*' -s "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_DIR"'",
        "messages": [{"role": "user", "content": "You are an ML researcher. Suggest hyperparameters for a transformer: DEPTH, ASPECT_RATIO, HEAD_DIM, DEVICE_BATCH_SIZE, TOTAL_BATCH_SIZE, EMBEDDING_LR, UNEMBEDDING_LR, MATRIX_LR, SCALAR_LR, WEIGHT_DECAY, WARMUP_RATIO, WARMDOWN_RATIO, FINAL_LR_FRAC, WINDOW_PATTERN. Return JSON only."}],
        "temperature": 0.7,
        "max_tokens": 512
    }' | python3 -c "import sys,json; r=json.load(sys.stdin); print('Response:', r['choices'][0]['message']['content'][:200])"

# Small delay to let memory allocation settle
sleep 3

# Measure VRAM after request (KV cache populated)
echo ""
echo "=== GPU memory AFTER warmup request ==="
nvidia-smi --id="$GPU_ID" --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
VRAM_AFTER_REQ=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
echo "Used: ${VRAM_AFTER_REQ} MiB"

# Summary
TOTAL_VRAM=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
VLLM_USAGE=$((VRAM_AFTER_REQ - VRAM_BEFORE))
AVAILABLE=$((TOTAL_VRAM - VRAM_AFTER_REQ))

echo ""
echo "=============================================="
echo "RESULTS: ${MODEL_NAME} (gpu_util=${VLLM_GPU_UTIL})"
echo "=============================================="
echo "Total GPU VRAM:        ${TOTAL_VRAM} MiB  ($((TOTAL_VRAM / 1024)) GiB)"
echo "VRAM before vLLM:      ${VRAM_BEFORE} MiB"
echo "VRAM after vLLM idle:  ${VRAM_AFTER_IDLE} MiB"
echo "VRAM after warmup req: ${VRAM_AFTER_REQ} MiB"
echo "vLLM total usage:      ${VLLM_USAGE} MiB  ($((VLLM_USAGE / 1024)) GiB)"
echo "Available for training: ${AVAILABLE} MiB  ($((AVAILABLE / 1024)) GiB)"
echo "=============================================="

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
