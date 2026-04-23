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

export OPENAI_API_KEY="dummy"
export CUDA_MEM_FRACTION=0.543
export AVAILABLE_VRAM="76GB"

echo "=============================================="
echo "Experiment: karpathy_agent [Kimi K2.6] seed ${SEED}"
echo "=============================================="
echo "Backend:     karpathy_agent"
echo "LLM model:   moonshotai/Kimi-K2.6"
echo "Seed:        $SEED"
echo "VRAM cap:    76 GB (CUDA_MEM_FRACTION=$CUDA_MEM_FRACTION)"
echo "Node:        $(hostname)"
echo "Results:     $RESULTS_DIR"
echo "Endpoint file: $ENDPOINT_FILE"
echo "=============================================="

mkdir -p "$RESULTS_DIR"

# Retry loop: the HPO backend fails hard on APIConnectionError/Timeout to
# prevent silent random fallback. When the Kimi server chains to a new SLURM
# job its IP may change and there is a gap of a few minutes while the new
# server boots. This loop (a) waits for a healthy endpoint, (b) re-reads the
# endpoint file (server node may have changed), (c) re-exports OPENAI_BASE_URL,
# (d) restarts the HPO with --resume. It only gives up if the server stays
# unreachable for a full 60-min polling window.
KIMI_ENDPOINT=""
for attempt in $(seq 1 200); do
    # (a-c) wait for a healthy endpoint, re-reading the file every 10s
    echo "[$(date -u +%T)] Attempt $attempt: waiting for healthy Kimi endpoint..."
    probe_ok=0
    for _i in $(seq 1 360); do
        if [ -f "$ENDPOINT_FILE" ]; then
            NEW_EP="$(cat "$ENDPOINT_FILE" 2>/dev/null || true)"
            if [ -n "$NEW_EP" ] && [ "$NEW_EP" != "$KIMI_ENDPOINT" ]; then
                echo "  endpoint: ${KIMI_ENDPOINT:-<unset>} -> $NEW_EP"
                KIMI_ENDPOINT="$NEW_EP"
                export OPENAI_BASE_URL="http://${KIMI_ENDPOINT}/v1"
                export OPENAI_API_BASE="http://${KIMI_ENDPOINT}/v1"
            fi
        fi
        if [ -n "$KIMI_ENDPOINT" ] && curl -sf --max-time 5 "http://${KIMI_ENDPOINT}/v1/models" > /dev/null 2>&1; then
            probe_ok=1
            break
        fi
        sleep 10
    done
    if [ $probe_ok -ne 1 ]; then
        echo "Kimi server never responded within 60 min at attempt $attempt — exiting"
        exit 1
    fi
    echo "[$(date -u +%T)] Kimi healthy at $KIMI_ENDPOINT"

    # (d) run HPO; capture exit code without killing the script
    set +e
    python -m autoresearch_automl.cli run \
        --backend "karpathy_agent" \
        --trials 9999 \
        --budget-max 300 \
        --seed "$SEED" \
        --llm-model "moonshotai/Kimi-K2.6" \
        --results-dir "$RESULTS_DIR" \
        --resume \
        --time-budget 86400
    rc=$?
    set -e
    if [ $rc -eq 0 ]; then
        echo "[$(date -u +%T)] HPO completed cleanly (wall-time budget hit)"
        break
    fi
    echo "[$(date -u +%T)] HPO exited with code $rc — likely LLM server unreachable; retrying after probe"
    sleep 30
done

echo ""
echo "Done: karpathy_agent [Kimi K2.6] seed ${SEED}"
