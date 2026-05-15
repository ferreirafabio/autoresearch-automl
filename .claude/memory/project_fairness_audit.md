---
name: Fairness audit results (2026-03-20)
description: All methods verified fair — same baseline, VRAM cap, train.py, search space, training-only wall-time
type: project
---

## Audit Results (all PASS)

1. **Baselines**: All 29 runs have trial 0 val_bpb in [0.9906, 0.9933], wall_time 300.0-300.3s, tok/s 1.70-1.77M
2. **VRAM cap**: All scripts set CUDA_MEM_FRACTION=0.543. 27B gpu_util=0.45, 0.8B gpu_util=0.15
3. **train.py**: All fixed-search-space methods reset to original each trial via AST injection. Karpathy Agent (Code) intentionally edits full source (open search space — by design)
4. **Wall-time**: wall_time_seconds = training time only (not LLM inference). 24h budget counts only training seconds
5. **Search space**: All 13 fixed-search-space methods use identical 14-HP ConfigurationSpace. LLAMBO OptunaHub and CMA-ES sample WINDOW_PATTERN randomly (standard behavior for those optimizers)

## Known design differences (intentional, not bugs)

- Karpathy Agent (Code): open search space, edits entire train.py, hill-climbs on source
- LLAMBO OptunaHub: categorical HP delegated to random sampling
- CMA-ES/Centaur CMA component: categorical sampled independently (CMA-ES is continuous-only)
