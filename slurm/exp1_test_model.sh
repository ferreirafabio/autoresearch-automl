#!/bin/bash
#SBATCH --job-name=exp1-memtest
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/exp1_memtest_%j.log
#SBATCH --requeue

set -euo pipefail

MODELS_BASE="/work/dlclarge1/ferreira-autoresearch-automl/models"
MODEL_NAME="${1:?Usage: sbatch exp1_test_model.sh <model-name>}"
MODEL_DIR="${MODELS_BASE}/${MODEL_NAME}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/exp1_memtest_${MODEL_NAME}"
VLLM_PORT=8000

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# Set vLLM GPU memory fraction based on model size
case "$MODEL_NAME" in
    *0.8B*)  VLLM_MEM=0.10 ;;
    *9B*)    VLLM_MEM=0.20 ;;
    *35B*)   VLLM_MEM=0.55 ;;
    *)       VLLM_MEM=0.30 ;;
esac

echo "=== Memory Test: $MODEL_NAME ==="
echo "vLLM GPU memory: ${VLLM_MEM} ($(python -c "print(f'{140*${VLLM_MEM}:.0f}GB')"))"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"

# Start vLLM
vllm serve "$MODEL_DIR" \
    --host 127.0.0.1 --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization $VLLM_MEM \
    --enforce-eager &
VLLM_PID=$!

echo "Waiting for vLLM..."
for i in $(seq 1 300); do
    STATUS=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo "vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "FAIL: vLLM died"
        exit 1
    fi
    sleep 1
done

STATUS=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
if [ "$STATUS" != "200" ]; then
    echo "FAIL: vLLM didn't start (status: $STATUS)"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Check GPU memory after vLLM startup
echo ""
echo "=== GPU Memory after vLLM ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader 2>/dev/null

# Query LLM
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_API_KEY="dummy"

RESPONSE=$(curl --noproxy '*' -s "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$MODEL_DIR"'","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10}')

if echo "$RESPONSE" | python -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d" 2>/dev/null; then
    echo "PASS: LLM query works"
else
    echo "FAIL: LLM query failed: $RESPONSE"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Run 1 HPO trial
echo ""
echo "=== Running 1 HPO trial ==="
python -m autoresearch_automl.cli run \
    --backend llm_greedy \
    --trials 1 \
    --budget-max 300 \
    --seed 0 \
    --llm-model "$MODEL_DIR" \
    --results-dir "$RESULTS_DIR"

echo ""
echo "=== GPU Memory after training ==="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader 2>/dev/null

# Check result
if [ -f "$RESULTS_DIR/trials.jsonl" ]; then
    cat "$RESULTS_DIR/trials.jsonl" | python -c "
import sys, json
d = json.load(sys.stdin) if False else json.loads(sys.stdin.readline())
print(f'val_bpb: {d[\"val_bpb\"]}')
print(f'success: {d[\"success\"]}')
print(f'error: {d[\"error\"]}')
print(f'peak_memory_gb: {d[\"peak_memory_gb\"]}')
print(f'wall_time_seconds: {d[\"wall_time_seconds\"]}')
"
fi

echo ""
echo "=== Test complete: $MODEL_NAME ==="

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
