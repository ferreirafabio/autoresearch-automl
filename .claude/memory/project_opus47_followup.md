---
name: Opus 4.7 followup state (rolling)
description: Opus 4.7 5-seed campaign is back on the cluster; state is now tracked by the Live Benchmark tab and the cron monitor, not by static memory
type: project
originSessionId: aabedbd7-00d1-4a75-a99c-cb2d93f6ce73
---
## Status as of 2026-05-15

Opus 4.7 went from "stalled after April" back to active. 5-seed campaign in progress.

| Method | Completed at 100% | Submitted today (jobids 29007710–29007719) |
|---|---|---|
| Centaur [Opus 4.7] | s0, s1 (n=2) | s2 resume, s3, s4 |
| KA Code [Opus 4.7] | s0 (n=1) | s1, s2, s3, s4 |
| KA HPs [Opus 4.7]  | s0, s1 (n=2) | s2 resume, s3, s4 |

10 jobs submitted (1 round of 24h each). Will chain more rounds as needed.

Results dir: `/work/dlclarge1/ferreira-autoresearch-automl/results/opus47_benchmark/<method>_claude_opus_4_7/seed_<s>/`. Per-trial JSONs all carry `"model": "claude-opus-4-7"` (audited 2026-05-15, zero contamination with the 4.6 dir).

## Live state is on the demo

Don't try to track per-seed numbers in this memory file — they move daily. Look at:
- The Live Benchmark tab: https://ferreirafabio.github.io/autoresearch-automl/#tab=tracker
- Sections A (convergence), B (slopegraph 4.6→4.7), C (Wilcoxon Δ vs TPE), D (per-generation cards)
- Or rerun `PYTHONPATH=. python3 scripts/build_tracker_hero.py` locally.

## Cron monitor handles pausing

`scripts/claude_usage_monitor.sh` (cron every 15 min, THRESHOLD=60) auto-pauses + reschedules these jobs across the Claude Code 7d reset. So submit liberally; the monitor manages quota. See project_claude_usage_monitor.md.

## Slurm scripts

`slurm/exp2_{centaur,karpathy_agent,karpathy_agent_hps}_claude_code_opus47.sh <seed>` (no second arg). They consume the user's Claude Code Max subscription, not the paid API (see feedback_claude_code_vs_api.md).

## Comparison baseline (Opus 4.6, finalized)

- Centaur [Opus 4.6]: 5/5 + s5/6/7 done. Mean ~0.9738.
- KA Code [Opus 4.6]: 4/5 done; s4 chained (jobs 29008253, 29008254 submitted 2026-05-15).
- KA HPs [Opus 4.6]: not run for 4.6.

## Adding a new Opus generation

See `project_live_benchmark_tab.md` — the GENERATIONS list in `scripts/build_tracker_hero.py` is the single source of truth for what shows up on the tracker.
