#!/bin/bash
#SBATCH --job-name=exp2-ka-hps-kimi
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/exp2_%x_%j.log
#SBATCH --requeue

set -euo pipefail

# Karpathy Agent (14 HPs, fixed search space) [Kimi K2.6]. Same SUGGEST_PROMPT
# as Qwen/Gemini variants — just a different OpenAI-compatible endpoint.
#
# Usage: sbatch slurm/exp2_karpathy_agent_hps_kimi.sh <seed>

SEED="${1:?Usage: sbatch exp2_karpathy_agent_hps_kimi.sh <seed>}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
ENDPOINT_FILE="/work/dlclarge1/ferreira-autoresearch-automl/kimi_k26_endpoint.txt"
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/kimi_k26_benchmark/karpathy_agent_hps_kimi_k2_6/seed_${SEED}"

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

echo "Waiting for Kimi K2.6 endpoint file..."
for i in $(seq 1 180); do
    if [ -f "$ENDPOINT_FILE" ]; then
        KIMI_ENDPOINT="$(cat "$ENDPOINT_FILE")"
        echo "Found endpoint: $KIMI_ENDPOINT"
        break
    fi
    sleep 10
done
if [ -z "${KIMI_ENDPOINT:-}" ]; then
    echo "Error: Kimi endpoint file not found after 30 min"
    exit 1
fi

export OPENAI_API_BASE="http://${KIMI_ENDPOINT}/v1"
export OPENAI_API_KEY="dummy"

export CUDA_MEM_FRACTION=0.543
export AVAILABLE_VRAM="76GB"

echo "=============================================="
echo "Experiment: karpathy_agent_hps [Kimi K2.6] seed ${SEED}"
echo "=============================================="
echo "Backend:     karpathy_agent_hps"
echo "LLM model:   moonshotai/Kimi-K2.6"
echo "API base:    $OPENAI_API_BASE"
echo "VRAM cap:    76 GB"
echo "Node:        $(hostname)"
echo "Results:     $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

curl -sf --max-time 30 "http://${KIMI_ENDPOINT}/v1/models" > /dev/null || {
    echo "Warning: Kimi server not yet responding"
}

python -m autoresearch_automl.cli run \
    --backend "karpathy_agent_hps" \
    --trials 9999 \
    --budget-max 300 \
    --seed "$SEED" \
    --llm-model "moonshotai/Kimi-K2.6" \
    --results-dir "$RESULTS_DIR" \
    --resume \
    --time-budget 86400

echo ""
echo "Done: karpathy_agent_hps [Kimi K2.6] seed ${SEED}"
