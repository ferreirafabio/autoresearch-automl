---
name: Gemini 3.1 Pro Preview result paths
description: Gemini 3.1 Pro Preview results are SPLIT across fabio's and zelaa's result dirs — check both before assuming seeds are missing
type: project
---

Gemini 3.1 Pro Preview results live in `gemini31pro_benchmark/` but are SPLIT across TWO result dirs. When searching, ALWAYS check both.

**Centaur [Gemini 3.1 Pro Preview]:**
`/work/dlclarge1/ferreira-autoresearch-automl/results/gemini31pro_benchmark/centaur_gemini_3_1_pro_preview/seed_{0,1,2}/trials.jsonl`
All 3 seeds present (359, 340, 336 trials as of 2026-04-11). This is in fabio's dir, NOT zelaa's.

**Karpathy Agent (Code) [Gemini 3.1 Pro Preview]:**
`/home/zelaa/autoresearch-automl-private/results/gemini31pro_benchmark/karpathy_agent_gemini_3_1_pro_preview/seed_{0,1,2}/trials.jsonl`
All 3 seeds (324, 302, 300 trials).

**LLAMBO (Paper) [Gemini 3.1 Pro Preview]:**
`/home/zelaa/autoresearch-automl-private/results/gemini31pro_benchmark/llambo_original_gemini_3_1_pro_preview/seed_{0,1,2}/trials.jsonl`
All 3 seeds (532, 597, 591 trials).

**How to apply:** Any analysis or plot touching Gemini 3.1 Pro Preview must merge results from BOTH dirs. Do not assume one dir has everything. The earlier version of this memory (from 9 days ago) said all results lived in zelaa's dir — that was correct at the time for KA Code and LLAMBO but Centaur Gemini Pro was only at seed_0; the seeds 1+2 were later run in fabio's dir and never moved into zelaa's.

**NOT in gemini31_benchmark** — that directory only has Flash-Lite Preview results.

**Read `trials.jsonl` (not per-trial files)** for `wall_time_seconds` and `peak_memory_gb`.
