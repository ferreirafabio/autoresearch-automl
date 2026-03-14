#!/bin/bash
#SBATCH --job-name=test-llm-log
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/test_llm_logging_%j.log

set -euo pipefail

# Minimal test: feed LLAMBO n_initial_samples trials via suggest()+tell(),
# then call suggest() once more to trigger LLM calls. No training needed.
MODELS_BASE="/work/dlclarge1/ferreira-autoresearch-automl/models"
MODEL_NAME="Qwen3.5-0.8B"
MODEL_DIR="${MODELS_BASE}/${MODEL_NAME}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/test_llm_logging3"
VLLM_PORT=8000

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export PYTHONUNBUFFERED=1

echo "=== Test LLM Call Logging ==="
echo "Node: $(hostname)"

mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR/llm_calls.jsonl"

# Start vLLM
echo "Starting vLLM server..."
vllm serve "$MODEL_DIR" \
    --host 127.0.0.1 --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.30 &
VLLM_PID=$!

for i in $(seq 1 300); do
    STATUS=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || echo "000")
    if [ "$STATUS" = "200" ]; then
        echo "vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM died"
        exit 1
    fi
    sleep 1
done

export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_API_KEY="dummy"

# Use unquoted heredoc so $MODEL_DIR and $RESULTS_DIR expand
python3 << PYEOF
import json, logging
from pathlib import Path
from autoresearch_automl.backends.llambo_backend import LLAMBOBackend
from autoresearch_automl.core.search_space import SearchSpaceBuilder

logging.basicConfig(level=logging.INFO)

MODEL_DIR = "${MODEL_DIR}"
RESULTS_DIR = Path("${RESULTS_DIR}")

# Build search space from train.py
space_builder = SearchSpaceBuilder(Path("autoresearch/train.py"))
space = space_builder.build_configspace()
print(f"ConfigSpace: {list(space.keys())}")

N_INITIAL = 3  # must match n_initial_samples

backend = LLAMBOBackend(
    model=MODEL_DIR,
    n_initial_samples=N_INITIAL,
    log_dir=RESULTS_DIR,
)
backend.configure(
    space=space,
    objectives=["val_bpb"],
    budget_range=(60, 60),
    seed=99,
)

# Feed N_INITIAL trials through suggest()+tell() so LLAMBO transitions
# from random to LLM-guided mode. Use fake val_bpb values.
fake_results = [1.089, 1.150, 1.069]
for i in range(N_INITIAL):
    config, budget = backend.suggest()
    print(f"Trial {i}: config={config}")
    backend.tell(config, budget, {"val_bpb": fake_results[i]})
    print(f"  told val_bpb={fake_results[i]}")

print()
print("=== Now calling suggest() — this should trigger LLM calls ===")
config, budget = backend.suggest()
print(f"Suggested config: {config}")
print(f"Budget: {budget}")

# Check log
log_path = RESULTS_DIR / "llm_calls.jsonl"
if log_path.exists():
    lines = log_path.read_text().strip().split("\n")
    print(f"\nSUCCESS: {len(lines)} LLM calls logged to {log_path}")
    for line in lines[:3]:
        r = json.loads(line)
        prompt_len = sum(len(m.get("content", "") or "") for m in r.get("messages", []))
        resp_len = len(r.get("response", "") or "")
        print(f"  call {r['call_id']}: model={r['model']}, prompt={prompt_len} chars, response={resp_len} chars, {r['elapsed_s']}s")
    if len(lines) > 3:
        print(f"  ... and {len(lines) - 3} more")
else:
    print(f"\nFAILED: {log_path} not found")
PYEOF

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
