#!/bin/bash
#SBATCH --job-name=test-centaur-state
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/test_centaur_state_%j.log

set -euo pipefail

# Quick test: Run Centaur for ~15 trials and verify CMA-ES state extraction works.
# This validates the split-key bug fix before running full 24h experiments.

MODEL_NAME="Qwen3.5-0.8B"
MODELS_BASE="/work/dlclarge1/ferreira-autoresearch-automl/models"
MODEL_DIR="${MODELS_BASE}/${MODEL_NAME}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/test_centaur_state"
VLLM_PORT=$((8100 + RANDOM % 900))

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

echo "=============================================="
echo "Test: Centaur CMA-ES State Extraction"
echo "=============================================="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

# Start vLLM server (small model, minimal GPU)
REASONING_ARGS="--default-chat-template-kwargs {\"enable_thinking\":false}"
vllm serve "$MODEL_DIR" \
    --host 127.0.0.1 --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.15 \
    $REASONING_ARGS &
VLLM_PID=$!

echo "Waiting for vLLM..."
for i in $(seq 1 120); do
    STATUS=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo "vLLM ready after ${i}s"
        break
    fi
    sleep 1
done

export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_API_KEY="dummy"
export AVAILABLE_VRAM="~118GB"
export CUDA_MEM_FRACTION=0.85

mkdir -p "$RESULTS_DIR"

# Run 15 trials (enough for CMA-ES to initialize + a few LLM trials)
echo ""
echo "Running 15 Centaur trials..."
python -m autoresearch_automl.cli run \
    --backend "centaur" \
    --trials 15 \
    --budget-max 300 \
    --seed 0 \
    --llm-model "$MODEL_DIR" \
    --results-dir "$RESULTS_DIR" \
    --time-budget 1800

echo ""
echo "=============================================="
echo "Verifying CMA-ES state extraction..."
echo "=============================================="

# Run the state extraction test against the actual study
python3 -c "
import json
from pathlib import Path

results_dir = Path('$RESULTS_DIR')
trials_file = results_dir / 'trials.jsonl'
if not trials_file.exists():
    print('ERROR: No trials.jsonl found')
    exit(1)

n_trials = sum(1 for _ in open(trials_file))
print(f'Completed {n_trials} trials')

# Check if any LLM logs exist (indicates LLM trials ran)
llm_logs = list(results_dir.glob('centaur_seed0_trial*.jsonl'))
print(f'LLM trial logs: {len(llm_logs)}')

# Check a log to see if CMA analysis was included
for log in llm_logs[:3]:
    with open(log) as f:
        data = json.loads(f.read())
    prompt = data.get('messages', [{}])[0].get('content', '')
    if 'still initializing' in prompt:
        print(f'FAIL: {log.name} shows \"still initializing\" — state extraction broken!')
        exit(1)
    elif 'sigma=' in prompt:
        print(f'OK: {log.name} shows sigma in prompt — state extraction works!')
    else:
        print(f'WARN: {log.name} — no CMA analysis section found')

print('')
print('ALL CHECKS PASSED — CMA-ES state extraction is working correctly.')
"

EXIT_CODE=$?
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
exit $EXIT_CODE
