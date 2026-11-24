#!/usr/bin/env bash
# Replace Open WebUI's data with an archive made by scripts/backup.sh.
#
#   scripts/restore.sh backups/webui-20261124-120000.tgz [--yes]
#
# Destructive, so it first takes a safety backup of the current data, and asks
# for confirmation unless --yes is given. cache/ is left in place.
set -euo pipefail

CONTAINER=open-webui
DATA_DIR=/app/backend/data
HELPER_IMAGE=alpine:3.20

# Print an error and exit.
die() { echo "error: $*" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: $0 <backup.tgz> [--yes]"
archive=$1
[ -f "$archive" ] || die "no such file: $archive"
# Resolve before cd-ing to the repo root, so relative paths work from anywhere.
archive_dir=$(cd "$(dirname "$archive")" && pwd)
archive_name=$(basename "$archive")

cd "$(dirname "$0")/.."

docker inspect "$CONTAINER" >/dev/null 2>&1 \
  || die "no '$CONTAINER' container - run scripts/bootstrap.sh first"

# List the whole archive before touching anything, so a truncated or corrupt file fails here.
# The listing goes to a file, not into `grep -q`: grep exiting early would kill tar with SIGPIPE.
docker run --rm -v "$archive_dir:/backup:ro" "$HELPER_IMAGE" \
  sh -c "tar tzf '/backup/$archive_name' > /tmp/list && grep -qx './webui.db' /tmp/list" \
  || die "$archive is unreadable or isn't an Open WebUI backup (see tar's error above)"

if [ "${2:-}" != "--yes" ]; then
  read -r -p "Replace all current Open WebUI data with $archive_name? Type 'restore': " answer
  [ "$answer" = restore ] || die "aborted"
fi

echo "==> Safety backup of the current data first"
scripts/backup.sh

echo "==> Stopping $CONTAINER"
docker compose stop open-webui >/dev/null
trap 'echo "==> Starting $CONTAINER"; docker compose up -d --wait open-webui >/dev/null' EXIT

echo "==> Restoring $archive_name"
docker run --rm --volumes-from "$CONTAINER" -v "$archive_dir:/backup:ro" "$HELPER_IMAGE" \
  sh -c "find $DATA_DIR -mindepth 1 -maxdepth 1 ! -name cache -exec rm -rf {} + \
         && tar xzf '/backup/$archive_name' -C $DATA_DIR"
