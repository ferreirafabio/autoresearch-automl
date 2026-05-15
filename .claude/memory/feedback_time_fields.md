---
name: Time fields in result files
description: elapsed_s vs wall_time_seconds — which file to read for training time, and where to find peak memory
type: feedback
---

There are TWO different time fields in TWO different files per trial. Do NOT confuse them.

**`trials.jsonl`** (in the seed directory) — written by `ResultsDB` via `runner.py`. This is the CORRECT source for:
- `wall_time_seconds`: actual training time of the optimizee (parsed from train.py's `training_seconds` output). This is what the 24h budget tracks.
- `peak_memory_gb`: peak GPU memory during training. Use this to verify VRAM cap compliance.
- `success`, `val_bpb`, `config`, etc.

**`<backend>_seed<N>_trial<M>.jsonl`** (per-trial files) — written by the backend's `_flush_llm_log`. Contains:
- `elapsed_s`: time for the LLM API call only (NOT training time). This is a RED HERRING for budget calculations.
- LLM prompt/response, thinking traces, etc.

**Why:** Previously wasted significant time summing `elapsed_s` from per-trial files and concluding seed 0 only had 13.2h. The actual training budget was 24.0h as shown in `trials.jsonl`.

**How to apply:** Always read `trials.jsonl` for budget, performance, and VRAM verification. Only use per-trial JSONL files for LLM conversation analysis. When checking run status, do: `python3 -c "import json; trials=[json.loads(l) for l in open('trials.jsonl')]; print(sum(t.get('wall_time_seconds',0) for t in trials)/3600)"`.
