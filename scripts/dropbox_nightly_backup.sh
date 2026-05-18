#!/bin/bash
# Nightly Dropbox backup of /work/.../results with 7-day rolling retention.
#
# Strategy: rclone sync (mirror) into dropbox:autoresearch-automl/results.
# Changed/deleted files on the destination are not lost: they get moved to
# dropbox:autoresearch-automl/.backups/<YYYY-MM-DD>/ via --backup-dir.
# Then we prune dated backup dirs older than 7 days.
#
# Cron entry (added to crontab via `crontab -e`):
#   0 3 * * * http_proxy=http://tfproxy.informatik.intra.uni-freiburg.de:8080/ \
#              https_proxy=http://tfproxy.informatik.intra.uni-freiburg.de:8080/ \
#              no_proxy=localhost,127.0.0.1 \
#              /work/dlclarge1/ferreira-autoresearch-automl/autoresearch-automl/scripts/dropbox_nightly_backup.sh \
#              >> /work/dlclarge1/ferreira-autoresearch-automl/logs/dropbox_backup.cron.log 2>&1
#
# Manual run:
#   bash scripts/dropbox_nightly_backup.sh

set -euo pipefail

SOURCE="/work/dlclarge1/ferreira-autoresearch-automl/results"
DEST_BASE="dropbox:autoresearch-automl"
DEST="${DEST_BASE}/results"
BACKUPS_BASE="${DEST_BASE}/.backups"
TODAY="$(date +%Y-%m-%d)"
BACKUP_DIR="${BACKUPS_BASE}/${TODAY}"
RETENTION_DAYS=7

log() { echo "[$(date -Iseconds)] $*"; }

log "=== Dropbox nightly backup ==="
log "Source:        ${SOURCE}"
log "Destination:   ${DEST}"
log "Backup-dir:    ${BACKUP_DIR}"
log "Retention:     ${RETENTION_DAYS} days"

# Mirror with retention. --backup-dir captures changes so deletions are reversible.
rclone sync "${SOURCE}" "${DEST}" \
    --backup-dir "${BACKUP_DIR}" \
    --transfers=8 --checkers=8 \
    --fast-list \
    --log-level INFO

# Prune backup dirs older than $RETENTION_DAYS.
log "Pruning backup dirs older than ${RETENTION_DAYS} days..."
cutoff_epoch=$(date -d "${RETENTION_DAYS} days ago" +%s)

while IFS= read -r line; do
    # rclone lsd output: "  -1 YYYY-MM-DD HH:MM:SS  -1 dirname"
    name=$(echo "$line" | awk '{print $NF}')
    if [[ "$name" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        dir_epoch=$(date -d "$name" +%s 2>/dev/null || echo 0)
        if [ "$dir_epoch" -gt 0 ] && [ "$dir_epoch" -lt "$cutoff_epoch" ]; then
            log "  pruning ${BACKUPS_BASE}/${name} (older than ${RETENTION_DAYS}d)"
            rclone purge "${BACKUPS_BASE}/${name}" --rmdirs || log "  WARN: purge failed for ${name}"
        fi
    fi
done < <(rclone lsd "${BACKUPS_BASE}" 2>/dev/null || true)

log "=== done ==="
