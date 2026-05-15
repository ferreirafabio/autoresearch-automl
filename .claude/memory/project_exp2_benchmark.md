---
name: Exp2 benchmark setup
description: 14 methods × 3 seeds, baseline ~0.991, 76GB VRAM cap, 24h training budget
type: project
originSessionId: aabedbd7-00d1-4a75-a99c-cb2d93f6ce73
---
## Experiment design

14 methods × 3 seeds (s0,s1 on kislurm, s2 on HoreKa). Each trial trains for 300s with 24h cumulative training budget (86400s). Baseline val_bpb ~0.991 on healthy H200s at ~1750K tok/s.

## Methods (14 total)

**Classical (4):** CMA-ES, TPE (Optuna), Random, SMAC (RF surrogate)
**LLM 0.8B (3):** LLAMBO Optuna, LLAMBO Paper, Karpathy Agent (14 HPs)
**LLM 27B (3):** LLAMBO Optuna, LLAMBO Paper, Karpathy Agent (14 HPs)
**Code agents (2):** Karpathy Agent (Code) [0.8B], Karpathy Agent (Code) [27B]
**Hybrid (2):** Centaur (CMA-ES+LLM) [0.8B], Centaur (CMA-ES+LLM) [27B]

All LLM methods use Qwen3.5 with thinking disabled, max_tokens=2048 — **except karpathy_agent_backend.py which uses max_tokens=8192** (full train.py rewrite needs more headroom; see project_thinking_disabled.md).

## Key config

- CUDA_MEM_FRACTION=0.543 (76GB cap for all methods)
- 27B: vLLM gpu_util=0.45, 0.8B: gpu_util=0.15
- Karpathy starting config as baseline (trial 0, not random init)
- 27B methods need multiple 24h SLURM rounds to reach 24h training budget

## SLURM script arg orders (CRITICAL - verify before submitting!)

- `exp2_classical.sh`: `<backend> <seed>`
- `exp2_llm.sh`: `<backend> <seed> [model_name] [nothink]` (default model: Qwen3.5-0.8B)
- `exp2_karpathy_agent.sh`: `<seed> <model_name>`
- `exp2_centaur.sh`: `<seed> <model_name>`

Examples:
- `sbatch slurm/exp2_classical.sh smac 0`
- `sbatch slurm/exp2_llm.sh llambo 0 Qwen3.5-27B nothink`
- `sbatch slurm/exp2_karpathy_agent.sh 0 Qwen3.5-27B`
- `sbatch slurm/exp2_centaur.sh 0 Qwen3.5-0.8B`
