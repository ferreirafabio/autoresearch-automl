---
name: Use optimizer/optimizee terminology in paper
description: Distinguish LLM optimizer (Qwen3.5 doing HPO) from LLM optimizee (50M model being trained) to avoid ambiguity
type: feedback
---

Use "optimizer" and "optimizee" to distinguish the two LLMs in the paper:
- **Optimizer**: Qwen3.5 (0.8B or 27B) that suggests HP configurations or code edits
- **Optimizee**: The ~50M parameter language model being trained/tuned

This is orthogonal to the fixed/unconstrained search space distinction. Both are needed — "optimizer/optimizee" clarifies *which* model, "fixed/unconstrained" clarifies *how* the optimizer operates.

**Why:** "LLM" is ambiguous when both the optimizer and the model being optimized are language models.

**How to apply:** Introduce the terms early (abstract or intro), then use consistently. Especially important wherever "LLM" could refer to either model.
