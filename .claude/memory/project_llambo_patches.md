---
name: LLAMBO cached file patches
description: Patches applied to cached LLAMBO files for Qwen3.5 compatibility and pandas fixes - must be reapplied if cache is cleared
type: project
---

Three patches applied to cached LLAMBO files at `~/.cache/optunahub/github.com/optuna/optunahub-registry/main/package/samplers/llambo/llambo/`:

1. **`llm.py` — Disable thinking mode for Qwen3.5:**
   - Added `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` to `chat.completions.create()` call
   - Qwen3.5 is a thinking model; without this, `message.content` is None (output goes to `reasoning_content`)
   - Also wrapped cost calculator in try/except for unknown model names (local vLLM paths)

2. **`acquisition_function.py`, `discriminative_sm.py`, `generative_sm.py` — pandas 2.0 fix:**
   - Changed `row[i]` → `row.iloc[i]` in all three files
   - pandas 2.0+ requires `.iloc` for positional access on string-indexed Series

**Why:** These are upstream bugs/incompatibilities in the OptunaHub LLAMBO sampler. Without them, all LLM calls return None and LLAMBO falls back to random sampling.

**How to apply:** If `~/.cache/optunahub` is cleared or optunahub updates, these patches must be reapplied. Consider upstreaming to optunahub-registry.
