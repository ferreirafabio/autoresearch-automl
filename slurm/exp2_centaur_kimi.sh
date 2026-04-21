#!/bin/bash
#SBATCH --job-name=exp2-centaur-kimi
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/exp2_%x_%j.log
#SBATCH --requeue

set -euo pipefail

# Experiment: Centaur [Kimi K2.6]. Uses the SAME centaur backend and SUGGEST_PROMPT
# as the Qwen and Gemini variants — the only change is that the OpenAI-compatible
# client points at a separately-hosted Kimi K2.6 vLLM server (see kimi_k26_server.sh).
#
# Requires: slurm/kimi_k26_server.sh already submitted and the endpoint file written.
#
# Usage:
#   sbatch slurm/exp2_centaur_kimi.sh 0

SEED="${1:?Usage: sbatch exp2_centaur_kimi.sh <seed>}"
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
ENDPOINT_FILE="/work/dlclarge1/ferreira-autoresearch-automl/kimi_k26_endpoint.txt"
RESULTS_DIR="/work/dlclarge1/ferreira-autoresearch-automl/results/kimi_k26_benchmark/centaur_kimi_k2_6/seed_${SEED}"

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost,dlc2gpu18,dlc2gpu19,dlc2gpu20,dlc2gpu21,10.5.166.0/24,.dlc2gpu"
export NO_PROXY="127.0.0.1,localhost,dlc2gpu18,dlc2gpu19,dlc2gpu20,dlc2gpu21,10.5.166.0/24,.dlc2gpu"

# Wait for Kimi endpoint file (server may still be loading; poll up to 30 min)
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
export OPENAI_API_KEY="dummy"  # vLLM doesn't check; OpenAI client needs it set

# VRAM cap on training (OPTIMIZEE) GPU — 76 GB (same as all other method variants)
export CUDA_MEM_FRACTION=0.543
export AVAILABLE_VRAM="76GB"

echo "=============================================="
echo "Experiment: centaur [Kimi K2.6] seed ${SEED}"
echo "=============================================="
echo "Backend:         centaur"
echo "LLM model:       moonshotai/Kimi-K2.6 (remote vLLM)"
echo "API base:        $OPENAI_API_BASE"
echo "Seed:            $SEED"
echo "VRAM cap:        76 GB (CUDA_MEM_FRACTION=$CUDA_MEM_FRACTION)"
echo "Node:            $(hostname)"
echo "Results:         $RESULTS_DIR"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

# Quick server reachability probe
curl -sf --max-time 30 "http://${KIMI_ENDPOINT}/v1/models" > /dev/null || {
    echo "Warning: Kimi server not yet responding — will keep retrying during run"
}

python -m autoresearch_automl.cli run \
    --backend "centaur" \
    --trials 9999 \
    --budget-max 300 \
    --seed "$SEED" \
    --llm-model "moonshotai/Kimi-K2.6" \
    --results-dir "$RESULTS_DIR" \
    --resume \
    --time-budget 86400

echo ""
echo "=============================================="
echo "Done: centaur [Kimi K2.6] seed ${SEED}"
echo "=============================================="
