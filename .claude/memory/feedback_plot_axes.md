---
name: Plot axes convention
description: Primary plots use cumulative training wall-time (x-axis), secondary/appendix use trial number
type: feedback
---

Primary convergence plots: x-axis = cumulative training wall-time (seconds). This is the scientifically fair comparison — same compute budget for all methods. LLM overhead (vLLM inference) is visible and honest.

Secondary plots (appendix): x-axis = trial number. Shows sample efficiency — how well each method uses each evaluation.

**Why:** Fixed wall-clock budget is standard in HPO benchmarking (BOHB, SMAC3 papers). Trial-number plots hide LLM inference overhead, making LLM methods look cheaper than they are.

**How to apply:** When generating convergence plots, always produce both views. `trials.jsonl` has `wall_time_seconds` and timestamps per trial for computing cumulative training time.
