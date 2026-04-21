#!/bin/bash
#SBATCH --job-name=kimi-k26-server
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/kimi_k26_server_%j.log
#SBATCH --requeue

set -euo pipefail

# Host Moonshot Kimi K2.6 on a single 8×H200 node (TP=8, no Ray needed).
# Uses .venv-kimi with vLLM 0.19.1 + torch 2.10+cu128 + local Marlin patch.

PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
MODEL_DIR="/work/dlclarge1/ferreira-autoresearch-automl/models/Kimi-K2.6"
ENDPOINT_FILE="/work/dlclarge1/ferreira-autoresearch-automl/kimi_k26_endpoint.txt"
VLLM_PORT=8100
SERVED_MODEL_NAME="moonshotai/Kimi-K2.6"

source /work/dlclarge1/ferreira-autoresearch-automl/.venv-kimi/bin/activate
cd "$PROJECT_DIR"

# CRITICAL: explicit LD_LIBRARY_PATH so multiprocess TP workers find bundled
# CUDA libs. Without this, spawned subprocesses hit cudaErrorInsufficientDriver
# because they dlopen the wrong libcudart.
VENV_SITE=/work/dlclarge1/ferreira-autoresearch-automl/.venv-kimi/lib/python3.12/site-packages
NV_LIBS=""
for d in $VENV_SITE/nvidia/*/lib; do
    NV_LIBS="$NV_LIBS:$d"
done
export LD_LIBRARY_PATH="$VENV_SITE/torch/lib${NV_LIBS}:${LD_LIBRARY_PATH:-}"

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Workarounds applied:
#  - VLLM_DISABLE_MARLIN_MOE=1: forces non-Marlin Triton WNA16 path
#    (local patch in compressed_tensors_moe.py; prebuilt Marlin PTX is
#    incompatible with the cluster driver)
export VLLM_DISABLE_MARLIN_MOE=1
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
# Disable flashinfer's RoPE kernel — its bundled tvm_ffi CUDA kernels require
# a newer driver than 570.211.01 (cudaErrorInsufficientDriver).
# Falls back to vLLM's native RoPE implementation.
export VLLM_DISABLE_FLASHINFER_ROPE=1

HOSTNAME_SHORT="$(hostname -s)"
# Use IP not hostname in the endpoint file to bypass cluster proxy/DNS issues
NODE_IP="$(hostname -I | awk '{print $1}')"

echo "=============================================="
echo "Kimi K2.6 vLLM server (single node, TP=8)"
echo "=============================================="
echo "Node:              $HOSTNAME_SHORT"
echo "Port:              $VLLM_PORT"
echo "Model dir:         $MODEL_DIR"
echo "Endpoint file:     $ENDPOINT_FILE"
echo "GPUs:              $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "VLLM_DISABLE_MARLIN_MOE: $VLLM_DISABLE_MARLIN_MOE"
echo "=============================================="

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls $MODEL_DIR/*.safetensors 2>/dev/null)" ]; then
    echo "Error: Model not found at $MODEL_DIR"
    exit 1
fi

cleanup() {
    echo "[server] cleanup"
    rm -f "$ENDPOINT_FILE"
}
trap cleanup EXIT INT TERM

echo "${NODE_IP}:${VLLM_PORT}" > "$ENDPOINT_FILE"
echo "Endpoint IP:${NODE_IP} (hostname=${HOSTNAME_SHORT}) port=${VLLM_PORT}"
echo "Wrote endpoint: ${HOSTNAME_SHORT}:${VLLM_PORT}"
echo ""

# Single-node TP=8 — no Ray, no distributed executor needed
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
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enforce-eager \
    --limit-mm-per-prompt '{"vision_chunk":0,"image":0,"video":0}'

echo ""
echo "=============================================="
echo "Kimi K2.6 vLLM server stopped"
echo "=============================================="
