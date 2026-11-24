#!/usr/bin/env bash
# Bring up the whole stack from a fresh clone: .env, Python venv, Open WebUI, and
# every model models.yaml needs. Safe to re-run; it only does what's missing.
#
#   scripts/bootstrap.sh
#
# Ollama runs natively (see docker-compose.yml), so it must already be running.
set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA_URL="http://localhost:11434"
VENV=.venv

# Print an error and exit.
die() { echo "error: $*" >&2; exit 1; }

# Print a step heading.
step() { echo "==> $*"; }

# Fail early, with the fix, if a required command is missing.
require() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found - $2"; }

require docker  "install Docker Desktop: https://www.docker.com/products/docker-desktop/"
require ollama  "install Ollama: https://ollama.com/download"
require python3 "install Python 3.10+"
require curl    "install curl"
require openssl "install openssl"

step "Checking Docker and Ollama"
docker info >/dev/null 2>&1 || die "Docker isn't running - start Docker Desktop and retry"
curl -sf -m 5 "$OLLAMA_URL/api/version" >/dev/null \
  || die "Ollama isn't answering at $OLLAMA_URL - start the Ollama app or run 'ollama serve'"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || die "Python 3.10+ required (found $(python3 --version 2>&1))"

if [ -f .env ]; then
  step "Keeping existing .env"
else
  step "Creating .env with a fresh WEBUI_SECRET_KEY"
  key=$(openssl rand -hex 32)
  # umask in a subshell: .env holds secrets, so create it owner-only from the start.
  (umask 077 && sed "s/^WEBUI_SECRET_KEY=\$/WEBUI_SECRET_KEY=$key/" .env.example > .env)
fi

step "Installing Python dependencies into $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --disable-pip-version-check -r requirements.txt

step "Starting Open WebUI and waiting until it's healthy (first run can take minutes)"
docker compose up -d --wait

step "Pulling models listed in models.yaml"
# A bad models.yaml exits 1 with a one-line error (on stderr, not captured), and set -e stops here.
models=$("$VENV/bin/python" -c '
import sys
from client.chat import RegistryError, load_registry, required_models
try:
    print("\n".join(required_models(load_registry())))
except RegistryError as e:
    sys.exit(f"error: {e}")')
for model in $models; do
  # Skip models already present: re-pulling can silently swap in a newer build
  # behind the same tag, which is exactly the drift this script should not cause.
  if ollama show "$model" >/dev/null 2>&1; then
    echo "    $model: already present"
  else
    ollama pull "$model"
  fi
done

port=$(grep -E '^OPEN_WEBUI_PORT=' .env | cut -d= -f2 || true)
cat <<EOF

Ready: http://localhost:${port:-3000}
  - First visit: create the admin account (sign-up then closes automatically).
  - CLI:  $VENV/bin/python client/chat.py chat
  - To mirror models.yaml into Open WebUI, add OPEN_WEBUI_API_KEY to .env
    (see README) and run: $VENV/bin/python sync_webui.py
EOF
