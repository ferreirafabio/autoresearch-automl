# Exp2 Benchmark — Run Status

**Last updated**: 2026-03-20 17:30 UTC

## Cluster Setup

- **kislurm**: seeds 0,1 (all methods) — H200 GPUs, `CUDA_MEM_FRACTION=0.543` (76GB cap)
- **HoreKa**: seed 2 (all methods) — H200 GPUs, same VRAM cap
- **Cross-cluster comparability**: confirmed (tok/s ~1750K on both)
- **Baseline val_bpb**: ~0.990–0.993 on healthy H200s

## Current Run Status

| # | Method | Dir name | s0 (kislurm) | s1 (kislurm) | s2 (HoreKa) |
|---|--------|----------|:------------:|:------------:|:------------:|
| 1 | CMA-ES | `cma_es` | 67t RUNNING | 64t RUNNING | 355t DONE |
| 2 | TPE (Optuna) | `optuna` | 71t RUNNING | 80t RUNNING | 312t DONE |
| 3 | Random | `random` | 91t RUNNING | 111t RUNNING | 582t DONE |
| 4 | SMAC (RF) | `smac` | 356t DONE | 361t DONE | 365t DONE |
| 5 | LLAMBO Optuna [0.8B] | `llambo` | 546t DONE | 507t DONE | PENDING |
| 6 | LLAMBO Paper [0.8B] | `llambo_original` | 479t DONE | 468t DONE | PENDING |
| 7 | Karpathy Agent 14HP [0.8B] | `llm_greedy` | 62t RUNNING | 60t RUNNING | PENDING |
| 8 | LLAMBO Optuna [27B] | `llambo_Qwen3_5_27B_nothink` | 63t RUNNING | 60t RUNNING | PENDING |
| 9 | LLAMBO Paper [27B] | `llambo_original_Qwen3_5_27B_nothink` | 44t RUNNING | 42t RUNNING | PENDING |
| 10 | Karpathy Agent 14HP [27B] | `llm_greedy_Qwen3_5_27B_nothink` | 31t RUNNING | 31t RUNNING | PENDING |
| 11 | Centaur (CMA-ES+LLM) [0.8B] | `centaur_Qwen3_5_0_8B` | PENDING | PENDING | PENDING |
| 12 | Centaur (CMA-ES+LLM) [27B] | `centaur_Qwen3_5_27B` | PENDING | PENDING | PENDING |
| 13 | Karpathy Agent Code [0.8B] | `karpathy_agent_Qwen3_5_0_8B` | 62t RUNNING | 84t RUNNING | PENDING |
| 14 | Karpathy Agent Code [27B] | `karpathy_agent_Qwen3_5_27B` | 16t RUNNING | 27t RUNNING | PENDING |

Trial counts are live snapshots — RUNNING jobs accumulate more over time.

## Job IDs

### Kislurm
- Classical (cma_es, optuna, random s0,s1): 27795708–27795713
- LLM methods (all 27B + 0.8B s0,s1): 27795722–27795729, 27793757
- Karpathy agents (code-writing, s0,s1): 27795705–27795707
- Centaur fresh re-runs (s0,s1): **27796426–27796429** (PENDING)

### HoreKa
- All 14 seed_2 jobs: **3931289–3931302** (all PENDING, earliest ETA: Mar 23 19:50)
- Classical: 3931289–3931292
- LLM: 3931293–3931298
- Centaur: 3931299–3931300
- Karpathy agents: 3931301–3931302

## Issues & Fixes Applied

| Date | Issue | Fix | Commit |
|------|-------|-----|--------|
| Pre-Mar-17 | H200 GPU throttling (~1300K tok/s) → baseline ~1.008 | Re-ran on healthy H200s (~1750K tok/s) → baseline ~0.991 | — |
| Mar 17 | VRAM cap missing → unfair GPU usage | Added `CUDA_MEM_FRACTION=0.543` (76GB cap) | f4e77cd |
| Mar 17 | SMAC v1: wrong success status | Fixed status mapping | — |
| Mar 18 | SMAC v2: wrong facade (GP instead of RF) | Switched to `HyperparameterOptimizationFacade` (RF) | 7627695 |
| Mar 19 | Centaur: LLM never saw CMA-ES state (2 extraction bugs) | Fixed state extraction | 12a1a81 |
| Mar 20 | Centaur s0,s1 data poisoned (resumed from broken trials) | Moved to legacy, fresh re-run | — |

## Root Cause: Baseline Gap (~1.008 vs ~0.991)

GPU power throttling, NOT the VRAM cap. `train.py` baseline uses ~45-50GB (well under 76GB cap). Throttled H200s ran at ~1300K tok/s → fewer training steps in 300s → worse val_bpb. Healthy H200s at ~1750K tok/s → proper convergence.

## Legacy Data

All invalid/superseded data is preserved under `_legacy_*/` in the results dir:

| Legacy dir | Contents | Reason |
|------------|----------|--------|
| `_legacy_no_vram_cap/` | 22 runs (classical s0,s1; all 27B LLM; karpathy agents) | Throttled GPUs, no VRAM cap, baseline ~1.008 |
| `_legacy_centaur_broken_state_extraction/` | 7 round-1 + 4 round-2 Centaur runs | LLM never saw CMA-ES state |
| `_legacy_smac_v1_success_status_bug/` | 2 SMAC runs (s0,s1) | Wrong trial success status |
| `_legacy_smac_v2_wrong_facade_gp/` | 3 SMAC runs (s0,s1,s2) | Used GP surrogate instead of RF |

## What To Do Next

1. **Wait for kislurm jobs to finish** — classical s0,s1 re-runs, all LLM methods, karpathy agents, centaur fresh re-runs
2. **Wait for HoreKa jobs** — all 14 seed_2 jobs (ETA: ~Mar 23+)
3. **Rsync HoreKa seed_2 data** — after HoreKa jobs complete, rsync `trials.jsonl` files from HoreKa to kislurm results dirs
4. **Regenerate plots** — `python scripts/plot_convergence.py` (reads from results dir, outputs to `assets/`)
5. **Verify Centaur** — after ~30min of new Centaur runs, check trial 0 baseline (~0.991) and confirm CMA-ES state appears in LLM prompts

## Results Dir

```
/work/dlclarge1/ferreira-autoresearch-automl/results/exp2_benchmark/
├── cma_es/seed_{0,1,2}/trials.jsonl
├── optuna/seed_{0,1,2}/trials.jsonl
├── random/seed_{0,1,2}/trials.jsonl
├── smac/seed_{0,1,2}/trials.jsonl
├── llambo/seed_{0,1}/trials.jsonl
├── llambo_original/seed_{0,1}/trials.jsonl
├── llambo_Qwen3_5_27B_nothink/seed_{0,1}/trials.jsonl
├── llambo_original_Qwen3_5_27B_nothink/seed_{0,1}/trials.jsonl
├── llm_greedy/seed_{0,1}/trials.jsonl
├── llm_greedy_Qwen3_5_27B_nothink/seed_{0,1}/trials.jsonl
├── centaur_Qwen3_5_0_8B/                          (empty, re-run pending)
├── centaur_Qwen3_5_27B/                           (empty, re-run pending)
├── karpathy_agent_Qwen3_5_0_8B/seed_{0,1}/trials.jsonl
├── karpathy_agent_Qwen3_5_27B/seed_{0,1}/trials.jsonl
├── _legacy_no_vram_cap/
├── _legacy_centaur_broken_state_extraction/
├── _legacy_smac_v1_success_status_bug/
└── _legacy_smac_v2_wrong_facade_gp/
```

## HoreKa Access

```bash
# Check jobs
python3 /work/dlclarge2/ferreira-oellm/open-instruct/oellm/utils/horeka/run_remote.py "squeue -u fr_ff1042 --start"

# Rsync results back (after jobs complete)
# Use pexpect pattern from memory (reference_horeka.md) with:
#   rsync fr_ff1042@horeka:/hkfs/work/workspace/scratch/fr_ff1042-autoresearch/results/exp2_benchmark/<method>/seed_2/
#     → /work/dlclarge1/ferreira-autoresearch-automl/results/exp2_benchmark/<method>/seed_2/

# Sync code to HoreKa
# rsync --exclude=.venv,.git,__pycache__,smac3_output,assets via SSH proxy
```
