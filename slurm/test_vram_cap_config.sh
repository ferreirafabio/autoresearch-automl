#!/bin/bash
#SBATCH --job-name=test-vram-config
#SBATCH --partition=alldlc2_gpu-h200
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/work/dlclarge1/ferreira-autoresearch-automl/logs/test_vram_cap_config_%j.log

set -uo pipefail
# Note: not using -e because we expect test 1 to fail (OOM)

# Test a config that needs ~83GB VRAM:
# DEPTH=9, BATCH=128, ASPECT=89, HEAD=128, TOTAL_BATCH=262144
# This used 82.9GB in the legacy uncapped run.
# With 76GB cap (CUDA_MEM_FRACTION=0.543) it should OOM.
# Without cap it should succeed.

PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
source "${PROJECT_DIR}/.venv/bin/activate"
cd "$PROJECT_DIR"

# Create a test train.py with the 83GB config baked in (must stay in autoresearch/ dir)
TEST_TRAIN="autoresearch/train_test_vram.py"
cp autoresearch/train.py "$TEST_TRAIN"

# Inject the config using sed
sed -i 's/^DEPTH = .*/DEPTH = 9/' "$TEST_TRAIN"
sed -i 's/^DEVICE_BATCH_SIZE = .*/DEVICE_BATCH_SIZE = 128/' "$TEST_TRAIN"
sed -i 's/^ASPECT_RATIO = .*/ASPECT_RATIO = 89/' "$TEST_TRAIN"
sed -i 's/^HEAD_DIM = .*/HEAD_DIM = 128/' "$TEST_TRAIN"
sed -i 's/^TOTAL_BATCH_SIZE = .*/TOTAL_BATCH_SIZE = 262144/' "$TEST_TRAIN"

echo "=== Test 1: WITH 76GB cap (CUDA_MEM_FRACTION=0.543) ==="
echo "Expected: OOM (config needs ~83GB)"
CUDA_MEM_FRACTION=0.543 timeout 300 python "autoresearch/train_test_vram.py" 2>&1 | tail -10
EXIT1=$?
if [ $EXIT1 -ne 0 ]; then
    echo "Result: FAILED (exit code $EXIT1) — likely OOM as expected"
else
    echo "Result: SUCCESS (unexpected!)"
fi

echo ""
echo "=== Test 2: WITHOUT cap (unset CUDA_MEM_FRACTION) ==="
echo "Expected: SUCCESS (H200 has 140GB)"
timeout 300 python "autoresearch/train_test_vram.py" 2>&1 | tail -10
EXIT2=$?
if [ $EXIT2 -ne 0 ]; then
    echo "Result: FAILED (exit code $EXIT2)"
else
    echo "Result: SUCCESS — config runs fine without cap"
fi

echo ""
echo "=== Summary ==="
echo "Config: DEPTH=9 BATCH=128 ASPECT=89 HEAD=128 (~83GB VRAM)"
echo "With 76GB cap:    exit=$EXIT1 (expected: non-zero/OOM)"
echo "Without cap:      exit=$EXIT2 (expected: 0/success)"

rm -f "$TEST_TRAIN"
