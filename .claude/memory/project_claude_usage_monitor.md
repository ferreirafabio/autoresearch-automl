---
name: Claude Code 7d usage monitor (cron-based auto-pause)
description: scripts/claude_usage_monitor.sh polls Claude Code OAuth usage and scancel + reschedules *claude_code* jobs when 7d util hits THRESHOLD; runs every 15 min via cron
type: project
originSessionId: aabedbd7-00d1-4a75-a99c-cb2d93f6ce73
---
## What it does

`scripts/claude_usage_monitor.sh` polls the Anthropic OAuth usage endpoint (`https://api.anthropic.com/api/oauth/usage`) using the token from `~/.claude/.credentials.json`. When `seven_day.utilization >= THRESHOLD` (env, currently 60), it:
1. Finds RUNNING/PENDING slurm jobs whose script name contains `claude_code`.
2. `scancel`s them.
3. Re-submits each with `--begin=<seven_day.resets_at>+30min` so they auto-resume after the weekly reset.
4. Skips jobs already deferred via BeginTime (idempotent).

## Current crontab (every 15 min)

```
*/15 * * * * CLAUDE_USAGE_THRESHOLD=60 http_proxy=http://tfproxy.informatik.intra.uni-freiburg.de:8080/ https_proxy=http://tfproxy.informatik.intra.uni-freiburg.de:8080/ no_proxy=localhost,127.0.0.1 /work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/scripts/claude_usage_monitor.sh >> /work/dlclarge1/ferreira-autoresearch-automl/logs/claude_usage_monitor.cron.log 2>&1
```

## Critical: proxy env required

The cluster needs `http_proxy` / `https_proxy` for outbound HTTPS. Cron has no proxy by default. Without it, curl times out silently and the script (pre-fix) wrote an empty cache then logged useless `OK 7d=% < 80%` lines for days. The script now fails loudly with `ERROR: empty curl response (proxy/network issue?)` if proxy isn't set.

## State + logs

- Cache file: `/tmp/claude-usage-cache-<uid>.json`, 60s TTL.
- Live log: `/work/dlclarge1/ferreira-autoresearch-automl/logs/claude_usage_monitor.log`
- Cron-mode log: `…/claude_usage_monitor.cron.log` (same content, different path because of redirected stderr).

**Why:** background of this is the Claude Code Max 20x subscription's rolling 7-day cap. The monitor lets us run any number of `*claude_code*` jobs and they'll pause + auto-reschedule across the weekly reset without manual babysitting.

**How to apply:** if a `*claude_code*` job goes from RUNNING to PENDING with reason `(BeginTime)`, that's the monitor. Don't `scancel` it — just wait for the reset. If you want to bump the threshold, edit the crontab `CLAUDE_USAGE_THRESHOLD=...` value. Don't conflate with Anthropic API spend — see feedback_claude_code_vs_api.md.
