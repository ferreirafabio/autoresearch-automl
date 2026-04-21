#!/bin/bash
#SBATCH --job-name=exp2-ka-code-kimi
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/exp2_%x_%j.log
#SBATCH --requeue

set -euo pipefail

# Karpathy Agent (Code, free code editing) [Kimi K2.6]. Same AGENT_PROMPT
# as Qwen/Gemini variants — just a different OpenAI-compatible endpoint.
#
# Usage: sbatch slurm/exp2_karpathy_agent_kimi.sh <seed>

SEED="${1:?Usage: sbatch exp2_karpathy_agent_kimi.sh <seed>}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
ENDPOINT_FILE="/work/dlclarge1/ferreira-autoresearch-automl/kimi_k26_endpoint.txt"
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/kimi_k26_benchmark/karpathy_agent_kimi_k2_6/seed_${SEED}"

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost,dlc2gpu18,dlc2gpu19,dlc2gpu20,dlc2gpu21,10.5.166.0/24,.dlc2gpu"
export NO_PROXY="127.0.0.1,localhost,dlc2gpu18,dlc2gpu19,dlc2gpu20,dlc2gpu21,10.5.166.0/24,.dlc2gpu"

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

export OPENAI_BASE_URL="http://${KIMI_ENDPOINT}/v1"
export OPENAI_API_BASE="http://${KIMI_ENDPOINT}/v1"
export OPENAI_API_KEY="dummy"

export CUDA_MEM_FRACTION=0.543
export AVAILABLE_VRAM="76GB"

echo "=============================================="
echo "Experiment: karpathy_agent [Kimi K2.6] seed ${SEED}"
echo "=============================================="
echo "Backend:     karpathy_agent"
echo "LLM model:   moonshotai/Kimi-K2.6"
echo "API base:    $OPENAI_API_BASE"
echo "VRAM cap:    76 GB"
echo "Node:        $(hostname)"
echo "Results:     $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

# Block until Kimi server actually serves (not just endpoint file exists);
# prevents fail-hard patch from killing us if server is still loading.
echo "Probing Kimi server at http://${KIMI_ENDPOINT}/v1/models..."
for _i in $(seq 1 360); do
    if curl -sf --max-time 5 "http://${KIMI_ENDPOINT}/v1/models" > /dev/null 2>&1; then
        echo "Kimi server healthy after $((_i*10))s"
        break
    fi
    sleep 10
done || { echo "Kimi server never responded after 60 min"; exit 1; }

python -m autoresearch_automl.cli run \
    --backend "karpathy_agent" \
    --trials 9999 \
    --budget-max 300 \
    --seed "$SEED" \
    --llm-model "moonshotai/Kimi-K2.6" \
    --results-dir "$RESULTS_DIR" \
    --resume \
    --time-budget 86400

echo ""
echo "Done: karpathy_agent [Kimi K2.6] seed ${SEED}"
