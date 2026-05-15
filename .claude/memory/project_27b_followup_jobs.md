---
name: 27B follow-up jobs for full 24h training
description: 27B methods need multiple SLURM rounds to reach 24h training budget due to LLM inference overhead
type: project
---

## Problem

27B LLM methods can't reach 24h training in a single 24h SLURM job because LLM inference overhead eats real time:
- LLAMBO 27B: ~90-210s avg training per trial (high OOM rate = early terminations)
- Karpathy Agent (Code) 27B: ~200-285s avg per trial
- Karpathy Agent (14 HPs) 27B: ~300s per trial (fastest, minimal LLM overhead)

After one 24h SLURM round, 27B methods reach only ~10-18% of the 24h training budget.

## Solution

Submit follow-up jobs with `--dependency=afterany:<current_jobs>` and `--resume`. Each round adds another ~10-18% of training budget. Need ~4-6 rounds total for 27B methods to reach 24h training.

## Jobs submitted 2026-03-20

Follow-up round (27796670–27796679): 10 jobs for all 27B methods × seeds 0,1, dependent on current batch finishing.

**Why:** wall_time_seconds in trials.jsonl measures only training time (fair), but SLURM wall-clock includes LLM inference overhead.

**How to apply:** When checking status, 27B methods at <100% need more SLURM rounds. Submit with `sbatch --dependency=afterany:<deps> slurm/exp2_*.sh <seed> Qwen3.5-27B [method]`.
