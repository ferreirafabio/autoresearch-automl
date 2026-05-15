---
name: Status update format
description: When user asks for status, show % of 24h budget and best val_bpb per run/seed in table format
type: feedback
---

When user asks for "status", show a table with % of 24h wall-clock training time budget used and best val_bpb per method/seed.

**Why:** User wants a quick at-a-glance view of progress and performance across all runs.

**How to apply:** Run the python snippet that computes `cum_time / 86400 * 100` and `min(val_bpb)` from trials.jsonl, then format as a markdown table:

| Method | s0 | s1 | s2 |
|--------|----|----|-----|
| **method_name** | X.X% best=0.XXXX | X.X% best=0.XXXX | — or pending |

Include pending/empty centaur rows. Add notable observations at the bottom (e.g. which method is leading).
