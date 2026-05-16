# CLAUDE.md

Project-level instructions for any Claude Code (or other LLM) assistant
working in this repo. Distilled from rolling feedback over the autoresearch-automl
project.

## Writing style (HARD rules)

- **No em dashes (—) anywhere.** Not in code comments, not in commit messages,
  not in paper drafts, not in README copy, not on the demo website, not in
  Slack/email/LinkedIn drafts. Use colons, parentheses, or rephrase.
- **No en dashes (–) in casual web copy either.** Use plain hyphens.
- Prefer terse sentences. State results and decisions directly; no
  meta-commentary.

## Git / repo split

- Three remotes:
  - `origin` = `autoresearch-automl-private` (private, full data + cluster paths)
  - `public` = `autoresearch-automl` (public, source of the GitHub Pages demo)
  - paper repo lives separately (Overleaf-synced)
- **The GitHub Pages demo deploys ONLY from `public/main`.** A push to
  `origin` will not update https://ferreirafabio.github.io/autoresearch-automl/.
  Always `git push origin main && git push public main` for non-cluster
  changes.
- **Never push cluster-specific strings to `public`:** any `/work/dlclarge*/`,
  `kislurm`, `horeka`, `hkfs`, `scc.kit.edu`, slurm account IDs, internal
  proxy hostnames. The `slurm/` directory is `.gitignore`d for this reason;
  any new helper scripts under `slurm/` live on disk only.
- Allowed to `git push` without asking for either repo, except force-push to
  main which should be flagged first.
- Never add a "Co-Authored-By: Claude" trailer to commits.

## Memory + Claude Code subscription

- Project memory lives at `~/.claude/projects/.../memory/`. Mirror it into
  `.claude/memory/` via `make memory-sync` then commit. Memory is **private
  repo only**.
- `*_claude_code` backends consume the Claude Code Max subscription
  (rolling 7-day cap), NOT the paid Anthropic API. Don't conflate the two
  when reporting on usage or rate-limiting.
- `scripts/claude_usage_monitor.sh` polls `https://api.anthropic.com/api/oauth/usage`
  every 15 min via cron. When `seven_day.utilization >= THRESHOLD` (env,
  currently 80), it scancels and resubmits all `*claude_code*` jobs with
  `--begin=<resets_at>+30min`. **Cron requires explicit `http_proxy` /
  `https_proxy` env vars** (the cluster blocks outbound HTTPS otherwise);
  the cron entry includes them. Without proxy the curl times out silently.
- The Anthropic OAuth response also exposes `five_hour.utilization` (5-hour
  rolling cap), `seven_day_sonnet.utilization`, etc. The monitor currently
  only acts on 7d. If you want it to also pause on 5h, edit the script.

## Cluster + jobs

- Results live at `/work/dlclarge1/ferreira-autoresearch-automl/results/`,
  with subtrees `exp2_benchmark/`, `opus46_benchmark/`, `opus47_benchmark/`,
  `sonnet46_benchmark/`, etc. Each method has `seed_0/...seed_N/trials.jsonl`.
- **Result-dir naming:** `llm_greedy` was renamed to `karpathy_agent_hps`
  in commit `1b0bc6a`; `tpe` is the canonical name for the Optuna TPE
  backend. The historic symlinks `llm_greedy -> karpathy_agent_hps`,
  `tpe -> optuna` were resolved into real renames on 2026-05-15. Don't
  reintroduce symlinks.
- **24h training-budget cap (`wall_time_seconds` in trials.jsonl).** Budget %
  must sum `wall_time_seconds` of ALL trials (success + failure), not just
  successful. Crop anything past 24h to show 100%, not 99.7%.
- For 27B LLM methods, ~6-10% of the 24h budget completes per 24h slurm
  wall round due to LLM inference overhead. Chain 3-5 rounds via
  `--dependency=afterany:<prev_jid>` to reach 100%.
- Don't add backwards-compat shims or silent fallbacks in HPO backends. On
  any LLM transport failure (connect, timeout, 4xx/5xx) we fail HARD;
  silent fallback to last-best/random would log fake trials.
- `vLLM` startup wait in slurm scripts is 1800s (bumped from 600s after
  multiple jobs hit shard-load timeouts under cluster IO contention).

## Result-dir display names

See `.claude/memory/project_dir_naming.md` for the canonical method-name
table. Use:

- "Karpathy Agent (Code)" and "Karpathy Agent (14 HPs)" everywhere
  user-facing (paper, demo, status reports). **Never abbreviate to
  "KA Code" / "KA HPs"**, not even on mobile.
- TPE (Optuna) for the classical TPE baseline.
- Centaur for the hybrid CMA-ES+LLM method.

## Demo (`docs/index.html`)

- Two top-level tabs: "Interactive Demo" (paper figures, stays canonical)
  and "Live Benchmark" (rolling tracker for each new Claude release).
- Live Benchmark sections use **1/2/3/4** numbering to match the
  Interactive Demo's 1-8 scheme. No A/B/C/D.
- Live Benchmark filter buttons mirror the Fig 1 SVG-line-icon style.
  Filter row uses the existing `initGroupFilters` / `applyGroupFilter`
  pipeline; new generations parse from the bracket in the trace name
  (`\[opus|sonnet|haiku (\d+)\.(\d+)\]`).
- Color convention on Live Benchmark: **color = method family, dash =
  Claude generation.** Centaur=#C62828, KA Code=#1565C0, KA HPs=#FFC107,
  TPE=#2196F3.
- TL;DR is now an **Abstract** block in a `<details>` element (collapsible).
- Tab state persists via `#tab=<paper|tracker>` URL hash.
- After any new seed completes, run `make tracker` to refresh sections
  A-D + the leader banner + the overview table. Then
  `git push origin main && git push public main`.
- **Show the user the regenerated paper figures (Interactive Demo tab)
  before pushing online.** They want a review pass on Fig-1/2/3 numbers
  whenever the underlying 5-seed campaign closes.

## Stats / paper conventions

- 5-seed minimum for any claim. Below n=3, skip the row in the Live
  Benchmark leader/overview/forest.
- Paired Wilcoxon signed-rank (one-sided, lower val_bpb wins) vs TPE as
  classical baseline. Significance markers: `*` p<0.10, `**` p<0.05,
  `***` p<0.01.
- Primary plots use cumulative wall-time x-axis; secondary/appendix use
  trial number. Forward-fill incumbents past seed end.
- Always disambiguate the optimizer (Qwen/Opus/Sonnet/Gemini etc.) from
  the optimizee (the 50M model being trained on Climbmix-400B).

## Tone in status updates

- Per-seed columns with `% budget + best val_bpb` is the expected status
  format.
- When a run wallclocks out, report it; don't dress it up. If quota
  paused jobs across reset, say so plainly.
- Be honest about mistakes. The 2026-05-11 incident (rm -rf'd a directory
  whose parent was a symlink, lost ~26h of compute) was the right move
  to surface immediately rather than hide.

## Misc

- Working directory `/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/`.
- `make tracker` rebuilds the Live Benchmark. `make memory-sync` and
  `make memory-commit` for memory. See `Makefile`.
- Overleaf-synced paper repo: pull first, then push. Always compile
  LaTeX locally before pushing; include main.bbl for reference resolution.
