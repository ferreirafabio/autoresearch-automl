#!/bin/bash
#SBATCH --job-name=kimi-k26-server
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/kimi_k26_server_%j.log
#SBATCH --requeue

set -euo pipefail

# Host Moonshot Kimi K2.6 (1T MoE, compressed-tensors quantization) as an
# OpenAI-compatible vLLM server on a single 8×H200 node (TP=8).
#
# Other SLURM jobs on OTHER nodes connect via OPENAI_API_BASE=http://<hostname>:<port>/v1
# The endpoint is written to a shared file so HPO jobs can discover it.
#
# Usage:
#   sbatch slurm/kimi_k26_server.sh
#
# For >24h coverage, chain another instance:
#   JOB1=$(sbatch --parsable slurm/kimi_k26_server.sh)
#   sbatch --dependency=afterany:$JOB1 slurm/kimi_k26_server.sh

PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
MODEL_DIR="/work/dlclarge1/ferreira-autoresearch-automl/models/Kimi-K2.6"
ENDPOINT_FILE="/work/dlclarge1/ferreira-autoresearch-automl/kimi_k26_endpoint.txt"
VLLM_PORT=8100
SERVED_MODEL_NAME="moonshotai/Kimi-K2.6"

source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
# Silence HF hub's telemetry and speed up loading from local dir
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

HOSTNAME_SHORT="$(hostname -s)"

echo "=============================================="
echo "Kimi K2.6 vLLM server"
echo "=============================================="
echo "Node:              $HOSTNAME_SHORT"
echo "Port:              $VLLM_PORT"
echo "Served model name: $SERVED_MODEL_NAME"
echo "Model dir:         $MODEL_DIR"
echo "Endpoint file:     $ENDPOINT_FILE"
echo "GPUs:              $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "=============================================="

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls $MODEL_DIR/*.safetensors 2>/dev/null)" ]; then
    echo "Error: Model not found at $MODEL_DIR or no safetensors files present"
    exit 1
fi

# Clean up endpoint file on any exit so HPO jobs stop trying to use a dead endpoint
cleanup() {
    echo "[server] cleaning up endpoint file"
    rm -f "$ENDPOINT_FILE"
}
trap cleanup EXIT INT TERM

# Publish endpoint before vLLM finishes loading so waiting jobs see it early;
# they'll retry on connection refused until the server is actually up.
echo "${HOSTNAME_SHORT}:${VLLM_PORT}" > "$ENDPOINT_FILE"
echo "Wrote endpoint to $ENDPOINT_FILE: ${HOSTNAME_SHORT}:${VLLM_PORT}"
echo ""
echo "Starting vLLM server (this may take 10-20 minutes to load 595 GB across 8 GPUs)..."

# --trust-remote-code: Kimi K2.6 uses a custom model class wrapping DeepseekV3
# --dtype auto: respect compressed-tensors storage format (don't upcast to bf16)
# --gpu-memory-utilization: leave ~15% headroom for KV cache scaling
# --max-model-len: Kimi default is 256K; we only need ~8K for HPO prompts (keeps KV cache small)
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --tensor-parallel-size 8 \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --disable-log-requests

echo ""
echo "=============================================="
echo "Kimi K2.6 vLLM server stopped"
echo "=============================================="
