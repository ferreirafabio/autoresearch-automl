#!/bin/bash
# Launch the complete Kimi K2.6 experiment:
#   - 1 vLLM server job + 2 chained continuations (3 × 24h = up to 72h coverage)
#   - 9 HPO jobs (3 methods × 3 seeds) each with 1 chained continuation
#     (2 × 24h SLURM wall ensures trial budget of 86400s / 24h trial wall
#      can actually accumulate despite per-call LLM overhead)
#
# Must be run AFTER the model download has completed and a smoke test has
# verified vLLM can load Kimi K2.6 on 8× H200.
#
# Usage:
#   bash slurm/launch_kimi_k26_experiment.sh

set -euo pipefail

cd "$(dirname "$0")/.."
SLURM=slurm

echo "=============================================="
echo "Launching Kimi K2.6 experiment"
echo "=============================================="

# 1) vLLM server — 3 chained 24h jobs
echo ""
echo "--- Kimi server chain ---"
SRV_1=$(sbatch --parsable $SLURM/kimi_k26_server.sh)
echo "server job 1: $SRV_1"
SRV_2=$(sbatch --parsable --dependency=afterany:$SRV_1 $SLURM/kimi_k26_server.sh)
echo "server job 2: $SRV_2  (after $SRV_1)"
SRV_3=$(sbatch --parsable --dependency=afterany:$SRV_2 $SLURM/kimi_k26_server.sh)
echo "server job 3: $SRV_3  (after $SRV_2)"

# 2) HPO jobs — each seed gets a parent (starts when possible) + 1 chained resume
#    The chain continues on the same results dir via --resume, so trial budget
#    accumulates across both jobs until it hits 86400s.
echo ""
echo "--- HPO job chains (parent + 1 chained resume per seed) ---"
declare -a PARENT_IDS=()
for method_script in \
    exp2_centaur_kimi.sh \
    exp2_karpathy_agent_hps_kimi.sh \
    exp2_karpathy_agent_kimi.sh
do
    for seed in 0 1 2; do
        parent=$(sbatch --parsable $SLURM/$method_script $seed)
        chain=$(sbatch --parsable --dependency=afterany:$parent $SLURM/$method_script $seed)
        echo "  $method_script seed=$seed: parent=$parent chain=$chain"
        PARENT_IDS+=($parent)
    done
done

echo ""
echo "=============================================="
echo "All jobs submitted."
echo "=============================================="
echo "Server chain:    $SRV_1 -> $SRV_2 -> $SRV_3"
echo "HPO parent jobs: ${PARENT_IDS[@]}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER --format=\"%.10i %.30j %.2t %.10M %R\""
echo ""
echo "Each HPO seed will resume across its parent+chain until --time-budget 86400s is reached."
