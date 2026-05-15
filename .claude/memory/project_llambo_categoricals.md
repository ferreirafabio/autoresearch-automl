---
name: LLAMBO categorical HP limitation
description: LLAMBO delegates categoricals to random sampling - causes high OOM failure rate with batch size and model width HPs
type: project
---

OptunaHub's LLAMBO sampler splits HPs into numerical (LLM-optimized) and categorical (randomly sampled). Our 4 categorical HPs (DEVICE_BATCH_SIZE, TOTAL_BATCH_SIZE, HEAD_DIM, WINDOW_PATTERN) are randomly sampled, never seen by the LLM.

**Impact:** 62% failure rate for LLAMBO vs 26% for TPE. Failed configs have high DEPTH + large DEVICE_BATCH_SIZE → OOM. The LLM can't learn to avoid OOM because it never sees batch size in the prompt.

**Root cause analysis:**
- Failed configs: mean DEPTH=15.7, mean DEVICE_BATCH_SIZE=175
- Successful configs: mean DEPTH=10.4, mean DEVICE_BATCH_SIZE=65
- LLM's pretrained prior biases toward deeper/wider models (correct in general, wrong with GPU memory constraint)
- Penalty value 100.0 tells LLM "bad" but not "OOM" — no memory constraint info in prompts

**Potential fixes (not yet implemented):**
1. Convert categoricals to IntDistribution so LLM sees them (LLAMBO has hidden "ordinal" type support in core but not exposed via Optuna interface)
2. Enrich task description with OOM context: "configs with DEVICE_BATCH_SIZE=256 and DEPTH>12 often OOM"
3. Modify sampler_base.py `_split_search_space()` to route categoricals through LLAMBO

**Why:** This is the main bottleneck for LLAMBO performance in our experiments.

**How to apply:** When improving LLAMBO results, address the categorical handling before tuning other aspects.
