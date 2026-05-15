---
name: No domain-specific hacks in LLAMBO fixes
description: Solutions must be general-purpose, not hardcoded to our specific task - the paper must show LLAMBO works as a general optimizer
type: feedback
---

Don't use domain-specific hacks like hardcoded penalty values (3.0) or OOM threshold formulas (DEPTH*DBS > 2304). The research paper needs to show LLAMBO is a general-purpose "best of both worlds" (LLM + TPE/random) optimizer.

**Why:** We can't assume a-priori knowledge of the performance range or failure modes. A penalty of 3.0 only works because we know val_bpb is ~1.0. For arbitrary tasks, we don't know this.

**How to apply:** Fixes should be general: map failures to infinity and handle that properly, or leverage the LLM's unique strength (reading task descriptions) rather than engineering around its weaknesses with hardcoded rules.
