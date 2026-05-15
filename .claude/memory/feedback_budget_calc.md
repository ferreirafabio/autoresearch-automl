---
name: Budget % includes all trials
description: Budget percentage must sum wall_time_seconds of ALL trials (success + failure), not just successful ones
type: feedback
---

Budget % = sum(wall_time_seconds for ALL trials) / 86400 * 100, capped at 100%.

**Why:** Failed trials (OOM, errors) still occupy the GPU. A method with 56% OOM rate still ran for 24h wall-clock — the failed trials just had short wall times (a few seconds each). The budget % should reflect total GPU time used, not just "productive" time.

**How to apply:** When computing budget % for status tables, always sum wall_time_seconds across ALL trials regardless of success/failure status. Also applies when checking if a run is complete (>= 90% budget means complete).
