#!/usr/bin/env bash
# Archive Open WebUI's data (users, chats, settings) to backups/webui-<timestamp>.tgz.
#
#   scripts/backup.sh
#
# The container is stopped while archiving so the SQLite database is consistent,
# then started again. cache/ (re-downloadable embedding models, ~1 GB) is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

CONTAINER=open-webui        # container_name in docker-compose.yml
DATA_DIR=/app/backend/data  # where the image mounts the volume
HELPER_IMAGE=alpine:3.20

# Print an error and exit.
die() { echo "error: $*" >&2; exit 1; }

# --volumes-from finds the volume through the container, so we never have to guess
# the compose-prefixed volume name (it's <project>_open-webui, not open-webui).
running=$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null) \
  || die "no '$CONTAINER' container - run scripts/bootstrap.sh first"

mkdir -p backups
name="webui-$(date +%Y%m%d-%H%M%S).tgz"

if [ "$running" = true ]; then
  echo "==> Stopping $CONTAINER for a consistent snapshot"
  docker compose stop open-webui >/dev/null
  # Restart even if archiving fails, so a failed backup never leaves the UI down.
  trap 'echo "==> Restarting $CONTAINER"; docker compose up -d --wait open-webui >/dev/null' EXIT
fi

echo "==> Writing backups/$name"
# Write to .partial and rename on success, so an interrupted run never leaves a
# truncated archive that looks like a real backup.
docker run --rm --volumes-from "$CONTAINER" -v "$PWD/backups:/backup" "$HELPER_IMAGE" \
  sh -c "tar czf /backup/$name.partial -C $DATA_DIR --exclude=./cache . && mv /backup/$name.partial /backup/$name"

echo "$(du -h "backups/$name" | cut -f1)  backups/$name"
