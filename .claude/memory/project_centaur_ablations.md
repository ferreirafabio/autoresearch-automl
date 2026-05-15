---
name: Centaur ratio ablation runs
description: Centaur LLM ratio ablations (0.1, 0.2, 0.5) for 0.8B and 27B, submitted 2026-03-23
type: project
---

## Centaur ratio ablation experiment

Testing LLM ratios r ∈ {0.1, 0.2, 0.3, 0.5} × {0.8B, 27B}. Default r=0.3 already has 3 seeds complete.

**Submitted 2026-03-23 (seed 0 only):**

| Job ID | Model | Ratio | Results dir |
|--------|-------|-------|-------------|
| 27851390 | 27B | 0.1 | `centaur_Qwen3_5_27B_ratio_0.1/seed_0` |
| 27851391 | 27B | 0.2 | `centaur_Qwen3_5_27B_ratio_0.2/seed_0` |
| 27851392 | 27B | 0.5 | `centaur_Qwen3_5_27B_ratio_0.5/seed_0` |
| 27851393 | 0.8B | 0.1 | `centaur_Qwen3_5_0_8B_ratio_0.1/seed_0` |
| 27851394 | 0.8B | 0.2 | `centaur_Qwen3_5_0_8B_ratio_0.2/seed_0` |
| 27851395 | 0.8B | 0.5 | `centaur_Qwen3_5_0_8B_ratio_0.5/seed_0` |

**Why:** Reviewer robustness question — show Centaur performance isn't overly sensitive to the 30% ratio choice.

**How to apply:** Once complete, add ablation table to paper appendix. Correct sbatch invocation: `sbatch slurm/exp2_centaur.sh <seed> <model> ratio_<r> <r>` (arg 3 = suffix for dir name, arg 4 = actual ratio value).

**Previous broken attempt:** 3 jobs all wrote to `centaur_Qwen3_5_0_8B_ratio_/seed_0` (missing ratio in dir name, all clobbering same dir). Cancelled and deleted.
