---
name: Results dir naming and de-aliasing history
description: Map of exp2_benchmark dir names to methods, plus the 2026-05-15 de-aliasing that replaced symlinks with renamed physical dirs
type: project
originSessionId: aabedbd7-00d1-4a75-a99c-cb2d93f6ce73
---
## Current dir → display name (post-2026-05-15)

| Dir name | Display name |
|----------|-------------|
| cma_es | CMA-ES |
| optuna | TPE (Optuna) |
| random | Random |
| smac | SMAC |
| llambo | LLAMBO Optuna [0.8B] |
| llambo_original | LLAMBO Paper [0.8B] |
| karpathy_agent_hps | Karpathy Agent (14 HPs) [0.8B] |
| llambo_Qwen3_5_27B_nothink | LLAMBO Optuna [27B] |
| llambo_original_Qwen3_5_27B_nothink | LLAMBO Paper [27B] |
| karpathy_agent_hps_Qwen3_5_27B_nothink | Karpathy Agent (14 HPs) [27B] |
| karpathy_agent_Qwen3_5_0_8B | Karpathy Agent (Code) [0.8B] |
| karpathy_agent_Qwen3_5_27B | Karpathy Agent (Code) [27B] |
| centaur_Qwen3_5_0_8B | Centaur [0.8B] |
| centaur_Qwen3_5_27B | Centaur [27B] |

## De-aliasing on 2026-05-15

Three symlinks were removed and their targets renamed in place:

- `karpathy_agent_hps -> llm_greedy` → `llm_greedy/` renamed to `karpathy_agent_hps/`
- `karpathy_agent_hps_Qwen3_5_27B_nothink -> llm_greedy_Qwen3_5_27B_nothink` → `llm_greedy_Qwen3_5_27B_nothink/` renamed to `karpathy_agent_hps_Qwen3_5_27B_nothink/`
- `optuna -> tpe` → `tpe/` renamed to `optuna/`

**Why:** commit `1b0bc6a` (Apr) had already renamed the *backend* from `llm_greedy` to `karpathy_agent_hps`, but the results directory kept the old name with a symlink alias. Result: seeds 0/1/2 in those dirs have `backend=llm_greedy` in trials.jsonl, seeds 3/4 have `backend=karpathy_agent_hps`. Same algorithm, two backend-name strings. Plotting code (`scripts/plot_convergence.py`, `plot_opus46_vs_opus47.py`) already used the canonical names `karpathy_agent_hps` and `optuna`, so the rename satisfied them without code changes.

For `optuna/`: the underlying Optuna backend always used the TPE sampler (`optuna.samplers.TPESampler`), so `optuna` and `tpe` were always the same algorithm — just two names. Display name is still "TPE" in plots.

**How to apply:** If you see references to `llm_greedy/`, `llm_greedy_Qwen3_5_27B_nothink/`, or `tpe/` in old scripts/notebooks/memories, those paths no longer exist — update to the post-rename names above. Slurm scripts already accept `karpathy_agent_hps` as a backend arg and `optuna` as a classical backend, so resubmits work without changes.
