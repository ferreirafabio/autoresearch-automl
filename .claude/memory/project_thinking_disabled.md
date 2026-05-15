---
name: Thinking disabled, max_tokens=2048, LLAMBO params aligned
description: All LLM runs use nothink mode with max_tokens=2048. llambo_original n_gens=10 and alpha=0.1 aligned with OptunaHub LLAMBO defaults.
type: project
originSessionId: aabedbd7-00d1-4a75-a99c-cb2d93f6ce73
---
## Decision (2026-03-16): Disable thinking, reduce max_tokens, align LLAMBO params

### Thinking disabled for all models
Qwen3.5-27B with thinking enabled generated **213K thinking tokens per trial** (855KB traces) to produce tiny answers (279 chars for ACQ, 14 chars for SM). This made llambo_original 27B take ~60 min/trial (24 trials/24h) vs ~6 min/trial without thinking.

**Why:** Thinking traces are wasteful for structured HP output. The model spends 15 minutes "thinking" to produce `## 0.98 ##`. No quality benefit observed.

### max_tokens reduced from 16384 to 2048 (most backends)
Without thinking, actual output is:
- ACQ (HP config): ~70 tokens (279 chars)
- SM (score): ~4 tokens (14 chars)
- karpathy_agent_hps (fixed HP space): ~200-500 tokens

2048 tokens is generous headroom. Applied to llambo_original_backend.py, llambo_backend.py (OptunaHub monkey-patch), and karpathy_agent_hps_backend.py.

**Exception — karpathy_agent_backend.py uses max_tokens=8192** (commit afda70d, 2026-05-15). It rewrites the full train.py per trial, so it needs more headroom. Was 32768/16384 historically, but vLLM 0.17 rejected those values with "max input length 0 chars", silently falling back to last-best source (which corrupted karpathy_agent_Qwen3_5_27B s3/s4 on 2026-05-09 — those dirs were renamed `.INVALID_silent_fallback_2026-05-09` and later deleted). 8192 is the working middle ground.

### LLAMBO parameters aligned
Previously mismatched between OptunaHub and llambo_original:

| Parameter | OptunaHub LLAMBO | llambo_original (old) | llambo_original (new) |
|---|---|---|---|
| n_candidates | 10 | 10 | 10 |
| n_gens | 10 | **5** | **10** |
| alpha | 0.1 | **-0.2** | **0.1** |
| n_templates | 1 | 1 | 1 |
| API batching | n=1 sequential | n=10 batched | n=10 batched |

API batching (n=1 vs n=10) is an implementation detail — same output, different parallelism. Not aligned because it's inherent to each implementation.

### Data reset
All LLM experiment data deleted and 18 jobs resubmitted:
- `{llm_greedy, llambo, llambo_original} × {0.8B, 27B} × {seed 0,1,2}`
- All nothink, max_tokens=2048
- TPE data preserved (unaffected)

**How to apply:** Never enable thinking for LLM-based HPO. If comparing LLAMBO variants, ensure algorithmic params (n_gens, alpha, n_candidates) match.
