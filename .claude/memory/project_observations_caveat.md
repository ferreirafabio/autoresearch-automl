---
name: README observations need verification after runs complete
description: Key observations in README Results section are preliminary — re-verify rankings and update specific numbers when all runs finish
type: project
---

The "Key Observations" section in README.md contains specific claims and numbers based on partial data (runs still in progress as of 2026-03-21). Before publishing:

1. **Re-run the ranking analysis** — method ordering may change as runs reach 100% budget
2. **Update specific val_bpb numbers** — current values are snapshots, not final
3. **Verify "worse than random" claim** — LLAMBO Optuna 27B and Karpathy Agent (14 HPs) may improve with more training time
4. **Check Centaur [0.8B] vs [27B]** — gap may narrow or reverse as both approach 24h
5. **Regenerate plots** before any publication push

**Why:** Experiments are still running. Rankings at 40-80% budget may not hold at 100%.

**How to apply:** After all runs reach ~100%, re-run the ranking script, update numbers in README Key Observations, regenerate plots, and push to both repos.
