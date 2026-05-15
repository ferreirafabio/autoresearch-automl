---
name: VRAM cap and legacy data
description: 76GB VRAM cap (CUDA_MEM_FRACTION=0.543) enforced since Mar 17; legacy dirs contain uncapped/broken runs
type: project
---

## VRAM cap

`CUDA_MEM_FRACTION=0.543` (76GB) added 2026-03-17 (commit f4e77cd), later adjusted from 0.57→0.543 (commit 784105f). All current runs use this cap. Results without it are in legacy dirs.

## Legacy directories (under results/exp2_benchmark/)

| Dir | Contents | Reason |
|-----|----------|--------|
| `_legacy_no_vram_cap/` | 22 runs, baseline ~1.008 | Throttled GPUs + no VRAM cap |
| `_legacy_centaur_broken_state_extraction/` | 7 round-1 + 4 round-2 Centaur runs | LLM never saw CMA-ES state |
| `_legacy_smac_v1_success_status_bug/` | 2 SMAC runs (s0,s1) | Wrong trial success status |
| `_legacy_smac_v2_wrong_facade_gp/` | 3 SMAC runs (s0,s1,s2) | Used GP surrogate instead of RF |

## Baseline difference explained

GPU power throttling (~1300K tok/s vs ~1750K tok/s), NOT the VRAM cap. Baseline config uses ~45-50GB (under 76GB cap). Fewer tok/s → fewer training steps in 300s → worse val_bpb.

**How to apply:** Only use current dirs (not _legacy*). Verify trial 0 baseline ~0.991 for any new run.
