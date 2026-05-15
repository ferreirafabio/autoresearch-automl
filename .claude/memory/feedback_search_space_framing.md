---
name: Search space framing
description: How to frame the search space section in README - emphasize fairness motivation and human prior elimination
type: feedback
---

The Search Space section should be framed as:
1. Classical methods need search spaces
2. Search space quality greatly affects results
3. To make comparison with Karpathy's autoresearch fair, we must eliminate human priors from the search space
4. So we auto-extract HPs from train.py via AST (not hand-picked)
5. Then show the 14 extracted HPs

**Why:** The motivation is fairness of comparison, not just a technical detail. The reader needs to understand WHY we use AST extraction before seeing WHAT we extracted.

**How to apply:** When writing/editing the Search Space section, lead with the fairness argument, not the technical mechanism.
