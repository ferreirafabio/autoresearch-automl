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
begin_7d=$(date -d "$resets_7d + 30 min" '+%Y-%m-%dT%H:%M:%S')
begin_5h=$(date -d "$resets_5h + 5 min"  '+%Y-%m-%dT%H:%M:%S')

usage_7d_int=$(printf '%.0f' "$usage_7d")
usage_5h_int=$(printf '%.0f' "$usage_5h")

trigger_7d=0; trigger_5h=0
[ "$usage_7d_int" -ge "$THRESHOLD" ]    && trigger_7d=1
[ "$usage_5h_int" -ge "$THRESHOLD_5H" ] && trigger_5h=1

if [ "$trigger_7d" = 0 ] && [ "$trigger_5h" = 0 ]; then
    log "OK  7d=${usage_7d}% < ${THRESHOLD}%, 5h=${usage_5h}% < ${THRESHOLD_5H}% (7d resets ${resets_7d})"
    exit 0
fi

# If 7d triggered, must wait for the 7d window: the 5h reset alone won't help.
# If only 5h triggered, the next 5h boundary is enough.
if [ "$trigger_7d" = 1 ]; then
    begin_local="$begin_7d"
    log "TRIGGER 7d=${usage_7d}% >= ${THRESHOLD}% (5h=${usage_5h}%): pausing, resume at ${begin_local}"
else
    begin_local="$begin_5h"
    log "TRIGGER 5h=${usage_5h}% >= ${THRESHOLD_5H}% (7d=${usage_7d}%): pausing, resume at ${begin_local}"
fi

# --- Find tracked jobids using *claude_code* slurm scripts ---
if [ ! -f "$TRACKER" ]; then
    log "ERROR: tracker $TRACKER not found"; exit 1
fi
mapfile -t opus_jids < <(awk -F'\t' 'NR>1 && $5 ~ /claude_code/ {print $1}' "$TRACKER" | sort -u)

if [ ${#opus_jids[@]} -eq 0 ]; then
    log "no claude_code jobs in tracker"; exit 0
fi

# --- Filter to RUNNING or PENDING-not-already-deferred ---
# A job counts as "already deferred" if reason=(BeginTime) OR StartTime is
# more than 1h in the future (covers `scontrol update StartTime=...` which
# clears the BeginTime reason but still defers the job).
ids_csv=$(IFS=,; echo "${opus_jids[*]}")
now_epoch=$(date +%s)
mapfile -t to_pause < <(squeue -h -j "$ids_csv" -o '%i %T %r %S' 2>/dev/null \
    | awk -v now="$now_epoch" '
        {
            jid=$1; state=$2; reason=$3; start=$4
            if (state=="RUNNING") { print jid; next }
            if (state=="PENDING") {
                if (reason=="(BeginTime)") next
                if (start ~ /^[0-9]{4}-/) {
                    cmd = "date -d \"" start "\" +%s 2>/dev/null"
                    cmd | getline ts; close(cmd)
                    if (ts != "" && ts+0 > now+3600) next
                }
                print jid
            }
        }')

if [ ${#to_pause[@]} -eq 0 ]; then
    log "above threshold but no active claude_code jobs to pause"; exit 0
fi

declare -A SEEN
for jid in "${to_pause[@]}"; do
    row=$(awk -F'\t' -v j="$jid" '$1==j {print; exit}' "$TRACKER")
    if [ -z "$row" ]; then
        log "  $jid: not in tracker; skipping"; continue
    fi
    IFS=$'\t' read -r _ batch method seed script args _ <<< "$row"
    # Strip any --begin=... tokens from prior deferrals so we don't pass them
    # as positional args to the slurm wrapper (it would treat them as <smoke>
    # or similar and either ignore or fail).
    clean_args=$(echo "$args" | sed -E 's/--begin=[^ ]+//g; s/  +/ /g; s/^ +//; s/ +$//')
    key="$method:$seed"
    if [ -n "${SEEN[$key]:-}" ]; then continue; fi
    SEEN[$key]=1

    if [ "$DRY_RUN" = "1" ]; then
        log "  DRY would scancel $jid ($method seed=$seed) and: sbatch --begin=$begin_local $script $clean_args"
    else
        scancel "$jid" 2>&1 | sed "s|^|  scancel $jid: |" | tee -a "$LOG_FILE" >&2 || true
        newjid=$(cd "$PROJECT_DIR" && sbatch --parsable --begin="$begin_local" $script $clean_args 2>&1 || echo "ERROR")
        log "  paused $method seed=$seed (was $jid) -> $newjid (begin $begin_local)"
        printf '%s\t%s\t%s_PAUSED_BY_USAGE\t%s\t%s\t%s\t%s\n' \
            "$newjid" "$batch" "$method" "$seed" "$script" "$clean_args" "$(date -Iseconds)" >> "$TRACKER"
    fi
done

log "done."
