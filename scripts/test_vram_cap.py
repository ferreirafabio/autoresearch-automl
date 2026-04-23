#!/usr/bin/env python3
"""Test: verify CUDA_MEM_FRACTION caps GPU memory to ~80GB on H200."""
import os
import torch

FRACTION = 0.57  # 80GB / 140GB H200

# Apply the cap exactly as train.py does
os.environ["CUDA_MEM_FRACTION"] = str(FRACTION)
torch.cuda.set_per_process_memory_fraction(FRACTION)

total = torch.cuda.get_device_properties(0).total_memory / 1024**3
allowed = total * FRACTION
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {total:.1f} GB")
print(f"Fraction: {FRACTION}")
print(f"Allowed VRAM: {allowed:.1f} GB")
print()

# Test 1: Allocate 70GB — should succeed (under 80GB cap)
print("Test 1: Allocate 70GB (should succeed)...")
try:
    x = torch.empty(int(70 * 1024**3 / 2), dtype=torch.bfloat16, device="cuda")
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"  PASS: allocated {allocated:.1f} GB")
    del x
    torch.cuda.empty_cache()
except torch.cuda.OutOfMemoryError:
    print(f"  FAIL: OOM at 70GB (unexpected — cap is {allowed:.1f} GB)")

# Test 2: Allocate 85GB — should OOM (over 80GB cap)
print("Test 2: Allocate 85GB (should OOM)...")
try:
    x = torch.empty(int(85 * 1024**3 / 2), dtype=torch.bfloat16, device="cuda")
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"  FAIL: allocated {allocated:.1f} GB (should have OOM'd!)")
    del x
    torch.cuda.empty_cache()
except torch.cuda.OutOfMemoryError:
    print(f"  PASS: OOM as expected (cap at {allowed:.1f} GB is working)")

# Test 3: Allocate 100GB — should definitely OOM
print("Test 3: Allocate 100GB (should OOM)...")
try:
    x = torch.empty(int(100 * 1024**3 / 2), dtype=torch.bfloat16, device="cuda")
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"  FAIL: allocated {allocated:.1f} GB (should have OOM'd!)")
    del x
    torch.cuda.empty_cache()
except torch.cuda.OutOfMemoryError:
    print(f"  PASS: OOM as expected (cap at {allowed:.1f} GB is working)")

# Test 4: No cap — verify 100GB works without the cap
print("\nTest 4: WITHOUT cap, allocate 100GB (should succeed on 140GB H200)...")
# Can't undo set_per_process_memory_fraction, so just report
print(f"  (skipped — can't undo memory fraction in same process)")
print(f"  Without cap, H200 has {total:.1f} GB available")

print("\n=== All tests done ===")
