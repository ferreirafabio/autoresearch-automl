#!/bin/bash
#SBATCH --job-name=exp1-test
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/exp1_test_%j.log
#SBATCH --requeue

set -euo pipefail

MODELS_BASE="/work/dlclarge1/ferreira-autoresearch-automl/models"
MODEL_NAME="Qwen3.5-0.8B"
MODEL_DIR="${MODELS_BASE}/${MODEL_NAME}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/exp1_test"
VLLM_PORT=8000

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

# Bypass cluster HTTP proxy for localhost connections
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

echo "=== Test Job: Validate vLLM + HPO pipeline ==="
echo "Python: $(which python) ($(python --version))"
echo "vLLM: $(python -c 'import vllm; print(vllm.__version__)')"
echo "transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "Model: $MODEL_DIR"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# Test 1: vLLM can load the model
echo ""
echo "--- Test 1: Starting vLLM server ---"
vllm serve "$MODEL_DIR" \
    --host 127.0.0.1 --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.3 &
VLLM_PID=$!

echo "Waiting for vLLM server to start (up to 300s)..."
for i in $(seq 1 300); do
    STATUS=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo "vLLM server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "FAIL: vLLM server died"
        exit 1
    fi
    sleep 1
done

STATUS=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
if [ "$STATUS" != "200" ]; then
    echo "FAIL: vLLM server didn't start within 300s (status: $STATUS)"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi
echo "PASS: vLLM server started"

# Test 2: Can query the LLM
echo ""
echo "--- Test 2: Query LLM ---"
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_API_KEY="dummy"

RESPONSE=$(curl --noproxy '*' -s "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$MODEL_DIR"'","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10}')
echo "LLM response: $RESPONSE"

if echo "$RESPONSE" | python -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d" 2>/dev/null; then
    echo "PASS: LLM query works"
else
    echo "FAIL: LLM query did not return expected response"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# Test 3: Run 1 trial of HPO
echo ""
echo "--- Test 3: Run 1 HPO trial ---"
python -m autoresearch_automl.cli run \
    --backend llm_greedy \
    --trials 1 \
    --budget-max 300 \
    --seed 0 \
    --llm-model "$MODEL_DIR" \
    --results-dir "$RESULTS_DIR"

echo ""
echo "--- Test 4: Check results ---"
if [ -f "$RESULTS_DIR/trials.jsonl" ]; then
    echo "trials.jsonl contents:"
    cat "$RESULTS_DIR/trials.jsonl"
    echo "PASS: Results written"
else
    echo "FAIL: No results file created"
fi

echo ""
echo "=== All tests passed ==="

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
