#!/usr/bin/env bash
# Claude Max usage monitor: pause *claude_code* SLURM jobs when over threshold.
#
# Polls https://api.anthropic.com/api/oauth/usage with the OAuth token from
# ~/.claude/.credentials.json. Triggers on EITHER:
#   - 7d utilization >= CLAUDE_USAGE_THRESHOLD       (default 80)
#   - 5h utilization >= CLAUDE_USAGE_THRESHOLD_5H    (default 90)
#
# When the 7d cap triggers, scancel and resubmit with
#   --begin=<seven_day.resets_at>+30min   (next weekly reset)
# When only the 5h cap triggers, resubmit with
#   --begin=<five_hour.resets_at>+5min    (next 5h rolling boundary)
# When both trigger, the more conservative 7d-resume time wins.
#
# Idempotent: jobs already deferred (StartTime in the future / reason
# (BeginTime)) are skipped.
#
# Run via cron:
#   */15 * * * * /work/.../autoresearch-automl/scripts/claude_usage_monitor.sh
#
# Or one-shot:  bash claude_usage_monitor.sh           (real run)
# Or dry-run:   bash claude_usage_monitor.sh --dry-run (just log the plan)

set -euo pipefail

THRESHOLD="${CLAUDE_USAGE_THRESHOLD:-80}"            # 7d cap threshold
THRESHOLD_5H="${CLAUDE_USAGE_THRESHOLD_5H:-90}"       # 5h cap threshold
PROJECT_DIR="/work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl"
TRACKER="${CLAUDE_USAGE_TRACKER:-/tmp/seed_runs_jobs.tsv}"
LOG_FILE="/work/dlclarge1/ferreira-autoresearch-automl/logs/claude_usage_monitor.log"
CACHE_FILE="/tmp/claude-usage-cache-$(id -u).json"
CACHE_TTL=60
CREDS="$HOME/.claude/.credentials.json"
DRY_RUN=0

[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

mkdir -p "$(dirname "$LOG_FILE")"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*" | tee -a "$LOG_FILE" >&2; }

# --- Refresh cache if stale ---
need_refresh=true
if [ -f "$CACHE_FILE" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$CACHE_FILE") ))
    [ "$age" -lt "$CACHE_TTL" ] && need_refresh=false
fi
if $need_refresh; then
    if [ ! -f "$CREDS" ]; then
        log "ERROR: $CREDS not found"; exit 1
    fi
    token=$(jq -r '.claudeAiOauth.accessToken // empty' "$CREDS" 2>/dev/null)
    if [ -z "$token" ]; then
        log "ERROR: no oauth token in $CREDS"; exit 1
    fi
    resp=$(curl -s --max-time 10 \
        -H "Authorization: Bearer $token" \
        -H "anthropic-beta: oauth-2025-04-20" \
        "https://api.anthropic.com/api/oauth/usage" || true)
    if [ -z "$resp" ]; then
        log "ERROR: empty curl response (proxy/network issue?)"; exit 1
    fi
    if echo "$resp" | jq -e '.seven_day.utilization' >/dev/null 2>&1; then
        echo "$resp" > "$CACHE_FILE"
    else
        log "ERROR: bad usage response: ${resp:0:200}"; exit 1
    fi
fi

usage_7d=$(jq -r '.seven_day.utilization' "$CACHE_FILE")
resets_7d=$(jq -r '.seven_day.resets_at' "$CACHE_FILE")
usage_5h=$(jq -r '.five_hour.utilization' "$CACHE_FILE")
resets_5h=$(jq -r '.five_hour.resets_at' "$CACHE_FILE")
usage_son=$(jq -r '.seven_day_sonnet.utilization // 0' "$CACHE_FILE")
resets_son=$(jq -r '.seven_day_sonnet.resets_at  // empty' "$CACHE_FILE")
[ -z "$resets_son" ] && resets_son="$resets_7d"

# resets_at may come back as 'null' / empty when utilization is 0 (no active
# rolling window). Guard the date calls so the script doesn't die.
_safe_begin() {
    local resets="$1" offset="$2"
    [ -z "$resets" ] || [ "$resets" = "null" ] && { echo ""; return; }
    date -d "$resets $offset" '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || echo ""
}
begin_7d=$(_safe_begin "$resets_7d"  "+ 30 min")
begin_5h=$(_safe_begin "$resets_5h"  "+ 5 min")
begin_son=$(_safe_begin "$resets_son" "+ 30 min")
# If a cap is currently at zero, its reset_at is null. That cap can't be
# triggered anyway (usage < threshold), so an empty begin string is fine.

usage_7d_int=$(printf '%.0f'  "$usage_7d")
usage_5h_int=$(printf '%.0f'  "$usage_5h")
usage_son_int=$(printf '%.0f' "$usage_son")

trigger_5h=0; trigger_opus=0; trigger_son=0
[ "$usage_5h_int"  -ge "$THRESHOLD_5H" ] && trigger_5h=1
# seven_day in the API is the Opus-equivalent rolling 7d cap; sonnet has
# its own seven_day_sonnet bucket. We pause Opus-scoped jobs on either
# 5h or 7d-Opus; we pause Sonnet-scoped jobs on either 5h or 7d-Sonnet.
[ "$usage_7d_int"  -ge "$THRESHOLD" ]    && trigger_opus=1
[ "$usage_son_int" -ge "$THRESHOLD" ]    && trigger_son=1

if [ "$trigger_5h" = 0 ] && [ "$trigger_opus" = 0 ] && [ "$trigger_son" = 0 ]; then
    log "OK  7d=${usage_7d}% < ${THRESHOLD}%, 7d-sonnet=${usage_son}% < ${THRESHOLD}%, 5h=${usage_5h}% < ${THRESHOLD_5H}%"
    exit 0
fi

# Decide a resume time per script-kind in the pause loop below.
# Default for logging:
if [ "$trigger_opus" = 1 ] || [ "$trigger_son" = 1 ]; then
    log "TRIGGER 5h=${usage_5h}% / 7d-opus=${usage_7d}% / 7d-sonnet=${usage_son}% (thresholds ${THRESHOLD_5H}/${THRESHOLD}/${THRESHOLD})"
else
    log "TRIGGER 5h=${usage_5h}% >= ${THRESHOLD_5H}% (7d caps OK)"
fi

# --- Discover claude_code jobs directly from squeue (no tracker file) ---
# A job qualifies if its `scontrol show job` Command path contains
# `_claude_code` (covers _claude_code.sh, _claude_code_opus47.sh,
# _claude_code_sonnet46.sh, future variants). RUNNING qualifies; PENDING
# qualifies only if not already deferred (reason != BeginTime AND
# StartTime not >1h in the future).
now_epoch=$(date +%s)
to_pause=()
while read -r jid state reason start; do
    [ -z "$jid" ] && continue
    if [ "$state" = "PENDING" ]; then
        [ "$reason" = "(BeginTime)" ] && continue
        if [[ "$start" =~ ^[0-9]{4}- ]]; then
            ts=$(date -d "$start" +%s 2>/dev/null || echo "")
            if [ -n "$ts" ] && [ "$ts" -gt "$((now_epoch + 3600))" ]; then
                continue
            fi
        fi
    elif [ "$state" != "RUNNING" ]; then
        continue
    fi
    cmd=$(scontrol show job "$jid" 2>/dev/null | awk -F= '/Command=/ {print $2; exit}')
    case "$cmd" in
        *_claude_code*.sh) to_pause+=("$jid|$cmd") ;;
    esac
done < <(squeue -h -u "$(id -un)" -o '%i %T %r %S' 2>/dev/null)

if [ ${#to_pause[@]} -eq 0 ]; then
    log "above threshold but no active claude_code jobs to pause"; exit 0
fi

# --- Pause each job: extract seed from its log, scancel, resubmit with --begin ---
LOG_DIR="/work/dlclarge1/ferreira-autoresearch-automl/logs"
declare -A SEEN
for entry in "${to_pause[@]}"; do
    jid="${entry%%|*}"
    script="${entry#*|}"
    # Try to recover the seed from the latest log line "Seed:" or "(seed N"
    # find won't error under set -e when no file matches; ls would
    log_glob=$(find "$LOG_DIR" -maxdepth 1 -name "exp2_*_${jid}.log" 2>/dev/null | head -1)
    seed=""
    if [ -n "$log_glob" ] && [ -f "$log_glob" ]; then
        seed=$(grep -oE "Seed:\s+[0-9]+" "$log_glob" 2>/dev/null | head -1 | grep -oE "[0-9]+" || true)
        if [ -z "$seed" ]; then
            seed=$(grep -oE "\(seed\s+[0-9]+" "$log_glob" 2>/dev/null | head -1 | grep -oE "[0-9]+" || true)
        fi
    fi
    if [ -z "$seed" ]; then
        log "  $jid: could not extract seed from log; skipping"
        continue
    fi
    key="$script:$seed"
    if [ -n "${SEEN[$key]:-}" ]; then continue; fi
    SEEN[$key]=1

    # Choose the model-specific begin time. Sonnet scripts pause only on
    # 5h or 7d-Sonnet; Opus scripts pause only on 5h or 7d-Opus.
    case "$(basename "$script")" in
        *sonnet*)
            if [ "$trigger_son" = 1 ]; then begin_local="$begin_son"
            elif [ "$trigger_5h" = 1 ]; then begin_local="$begin_5h"
            else continue
            fi
            ;;
        *)
            if [ "$trigger_opus" = 1 ]; then begin_local="$begin_7d"
            elif [ "$trigger_5h" = 1 ]; then begin_local="$begin_5h"
            else continue
            fi
            ;;
    esac

    if [ "$DRY_RUN" = "1" ]; then
        log "  DRY would scancel $jid and: sbatch --begin=$begin_local $script $seed"
    else
        scancel "$jid" 2>&1 | sed "s|^|  scancel $jid: |" | tee -a "$LOG_FILE" >&2 || true
        newjid=$(cd "$PROJECT_DIR" && sbatch --parsable --begin="$begin_local" "$script" "$seed" 2>&1 || echo "ERROR")
        log "  paused $(basename "$script") seed=$seed (was $jid) -> $newjid (begin $begin_local)"
    fi
done

log "done."
