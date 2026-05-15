---
name: autoresearch-automl project overview
description: HPO integration into Karpathy's autoresearch - key constraints, dataset, architecture, and roadmap phases
type: project
originSessionId: aabedbd7-00d1-4a75-a99c-cb2d93f6ce73
---
Project integrates AutoML HPO into Karpathy's autoresearch framework.

**Key constraints:**
- Can ONLY modify train.py — prepare.py is read-only
- Cannot install new packages into autoresearch's pyproject.toml
- 5-min fixed time budget per experiment, single GPU
- evaluate_bpb in prepare.py is ground truth metric
- Dataset: karpathy/climbmix-400b-shuffle (NOT FineWeb-Edu)

**Architecture:**
- HPO backends (current): random, optuna (TPE), smac, cma_es (classical); llambo, llambo_original, karpathy_agent_hps (LLM-fixed-space, ex-`llm_greedy`); karpathy_agent (LLM code-editing); centaur (hybrid CMA+LLM); plus `*_claude_code` variants of the LLM ones for Anthropic SDK
- Note: `llm_greedy` was renamed to `karpathy_agent_hps` in commit 1b0bc6a (Apr); results dirs de-aliased on 2026-05-15 — see project_dir_naming.md
- LLM serving: vLLM with Qwen3.5-{0.8B, 27B} on H200; Gemini & Claude variants via API/SDK; Kimi K2.6 self-hosted (see project_kimi_k26_setup.md)
- 14 HPs extracted from train.py: 10 numerical + 4 categorical
- Categorical HPs: DEVICE_BATCH_SIZE, TOTAL_BATCH_SIZE, HEAD_DIM, WINDOW_PATTERN

**Experiment setup:**
- Exp2 benchmark: TPE vs LLAMBO, 3 seeds, 24h runs, 300s budget per trial
- Results at /work/dlclarge1/ferreira-autoresearch-automl/results/exp2_benchmark/
- Plotting script: scripts/plot_convergence.py

**Why:** Research paper "Beyond LLM Hill-Climbing: Integrating AutoML into Autonomous ML Research"

**How to apply:** All pipeline work must respect the train.py-only constraint. Baseline val_bpb ~0.998 on H200.
