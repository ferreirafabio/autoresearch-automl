---
name: 24h hard cap and data integrity
description: All results must be hard-capped at 24h, repo copies refreshed before plotting, interpolation forward-filled
type: feedback
---

## Hard 24h cap
All trials.jsonl files in the repo results/ directory must be hard-cropped at 24h cumulative training time. No trial beyond the 24h budget should be included. This applies to:
- Released JSON files in results/
- Plot generation (cap_trials_at_budget in plot_convergence.py)
- Diversity analysis (analyze_diversity.py)
- Any per-seed best val_bpb computation

**Why:** Methods that ran >24h (e.g., Centaur [27B] seed 0 at 28.6h, TPE seeds at 127%) had unfair advantage.

## Repo copy freshness
Always refresh repo copies from original results dir BEFORE regenerating plots. The original results dir is at `/work/dlclarge1/ferreira-autoresearch-automl/results/exp2_benchmark/`. Name mapping needed:
- `llm_greedy` -> `karpathy_agent_hps`
- `llm_greedy_Qwen3_5_27B_nothink` -> `karpathy_agent_hps_Qwen3_5_27B`
- `llambo_Qwen3_5_27B_nothink` -> `llambo_Qwen3_5_27B`
- `llambo_original_Qwen3_5_27B_nothink` -> `llambo_original_Qwen3_5_27B`

## Interpolation
Wall-time plots use `right=values[-1]` (forward-fill last incumbent) to avoid "line going up" artifact when seeds end slightly before 24h.

## Budget filter
95% (MIN_BUDGET_FRAC=0.95). Seeds below this are excluded from convergence plots. This includes Karpathy Agent (Code) [27B] seed 1 at 97%.

## Checklist before plotting
1. Refresh all results from original dir
2. Hard crop at 24h
3. Regenerate plots
4. Push to code repo (private + public)
5. Copy plots to paper repo figures/
6. Push paper repo

## Common pitfalls caught
- Stale repo copies showing old data while original has newer runs
- Seeds at 97-98% excluded by 99% filter but included by 95% filter
- 1-seed vs 3-seed comparison (Karpathy Code [27B] with only 1 completed seed looked artificially good)
- >24h data inflating trial-number plots and diversity metrics
