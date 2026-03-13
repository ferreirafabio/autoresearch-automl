#!/bin/bash
#SBATCH --job-name=vllm-qwen
#SBATCH --partition=alldlc2
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/vllm_%j.log

source /work/dlclarge1/ferreira-autoresearch-automl/venvs/ar-automl/bin/activate

vllm serve /work/dlclarge1/ferreira-autoresearch-automl/models/Qwen3.5-35B-A3B \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768
