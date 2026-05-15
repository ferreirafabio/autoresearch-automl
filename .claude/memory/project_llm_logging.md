---
name: LLM call logging
description: Per-trial LLM logging captures prompts/responses; thinking DISABLED for all models (0.8B and 27B)
type: project
---

LLM call logging captures every prompt/response for LLAMBO, LLM Greedy, Centaur, and Karpathy Agent backends.

**File structure per trial:**
- `{backend}_seed{N}_trial{T}.jsonl` — response + metadata

**Thinking mode: DISABLED for all models (since 2026-03-16)**
- 0.8B: enters infinite `<think>` loop, never produces answer
- 27B: thinking disabled too (max_tokens=2048)
- vLLM flag: `--default-chat-template-kwargs '{"enable_thinking": false}'`

**vLLM config:**
- `--max-model-len 32768` (prompts grow with trials)
- 27B: `--enforce-eager --enable-prefix-caching`, gpu_util=0.45
- 0.8B: gpu_util=0.15

**Resume safety:**
- LLM Greedy: `replay()` sets `_replaying=True` to skip log writes
- LLAMBO: `replay()` syncs `_trial_id = len(history)`
- SLURM: `--requeue` + HPO `--resume` for preemptive scheduling
