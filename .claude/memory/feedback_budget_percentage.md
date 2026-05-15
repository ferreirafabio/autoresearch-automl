---
name: Budget percentage calculation
description: Runs that exceeded 24h are cropped to 24h and should show 100%, not 99.7%
type: feedback
---

When computing budget percentage for status tables, runs that exceeded the 24h budget (86400s) were cropped to exactly 24h. These are 100% complete, not 99.7%.

**Why:** The raw cumulative training time for most completed runs exceeds 86400s (e.g., 86700s). We hard-cap at 24h, so the last trial fitting within the cap may end a few minutes before 24h. Reporting 99.7% is misleading — the run used its full budget and was cropped.

**How to apply:** When calculating percentage, use `min(total_time / 86400 * 100, 100.0)` where `total_time` is the uncapped sum. If total_time >= 86400, report 100%. Only report <100% for runs that genuinely didn't finish (like karpathy_agent_27B/seed_2 at 91%).
