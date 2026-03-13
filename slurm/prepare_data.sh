#!/bin/bash
#SBATCH --job-name=ar-prepare
#SBATCH --partition=alldlc2_gpu-l40s
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/prepare_%j.log
#SBATCH --requeue

set -euo pipefail

source /work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/.venv/bin/activate
cd /work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/autoresearch

# Download 10 training shards (default) + pinned val shard, then train tokenizer
python prepare.py --num-shards 10 --download-workers 8

echo "prepare.py complete"
