"""Mirror the task registry (models.yaml) into Open WebUI as custom models.

Each task becomes a Workspace model with id `task-<name>`, based on the task's
Ollama model, with the task's system prompt and temperature. The registry is the
source of truth: re-running creates missing models, updates changed ones, and
overwrites edits made to these models in the UI.

    python3 sync_webui.py            # apply
    python3 sync_webui.py --dry-run  # show what would change

Needs OPEN_WEBUI_API_KEY (Open WebUI -> Settings -> Account -> API Keys), read
from the environment or .env. OPEN_WEBUI_URL defaults to http://localhost:<OPEN_WEBUI_PORT or 3000>.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from client.chat import RegistryError, load_registry

ENV_PATH = Path(__file__).resolve().parent / ".env"
MODEL_ID_PREFIX = "task-"
TIMEOUT_SECONDS = 10


class SyncError(Exception):
    """An operational failure talking to Open WebUI, with a message fit for the user."""


# Return one .env value as docker compose reads it: quotes removed, inline comment dropped.
def parse_env_value(raw, line_number):
    # Covers the forms compose users write in practice. Compose's rarer ones (multi-line
    # values, \n escapes, ${VAR} expansion) aren't needed for a URL, a port and an API key.
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        if end == -1:
            raise SyncError(f".env line {line_number}: unclosed quote")
        return raw[1:end]
    return re.split(r"\s#", raw, maxsplit=1)[0].strip()


# Fill unset environment variables from KEY=VALUE lines in a .env file, if it exists.
def load_dotenv(path=ENV_PATH, environ=os.environ):
    # docker compose reads .env on its own; doing the same here means one place for
    # the key. Real environment variables win, matching compose's precedence.
    if not path.exists():
        return
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        environ.setdefault(key, parse_env_value(value.strip(), number))


# Build the Open WebUI model payload for one registry task.
def to_webui_model(task, cfg):
    params = {"temperature": cfg["temperature"]}
    if cfg.get("system"):
        params["system"] = cfg["system"]
    return {
        "id": MODEL_ID_PREFIX + task,
        "base_model_id": cfg["model"],
        "name": f"{task.capitalize()} ({cfg['model']})",
        "meta": {"description": cfg.get("description")},
        "params": params,
    }


# Return True if an existing Open WebUI model differs from the payload in any field we own.
def needs_update(existing, desired):
    return (existing.get("name") != desired["name"]
            or existing.get("base_model_id") != desired["base_model_id"]
            or (existing.get("meta") or {}).get("description") != desired["meta"]["description"]
            or (existing.get("params") or {}) != desired["params"])


# Create or update every registry task in Open WebUI; return (model id, action) pairs.
def sync(registry, api, dry_run=False):
    results = []
    for task, cfg in registry["tasks"].items():
        desired = to_webui_model(task, cfg)
        existing = api.get_model(desired["id"])
        if existing is None:
            action = "created"
            if not dry_run:
                api.create_model(desired)
        elif needs_update(existing, desired):
            action = "updated"
            if not dry_run:
                # Skipping no-op updates matters: each update adds a version to the
                # model's history in Open WebUI.
                api.update_model(desired)
        else:
            action = "unchanged"
        results.append((desired["id"], action))
    return results


class WebUIClient:
    """Minimal client for Open WebUI's workspace-models API (v0.13.x)."""

    # Store the base URL and API key used for every request.
    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    # Send one JSON request and return the decoded body (None on 404 if allowed); raise SyncError otherwise.
    def _request(self, method, path, body=None, allow_404=False):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as e:
            if e.code == 404 and allow_404:
                return None
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (401, 403):
                raise SyncError(f"Open WebUI rejected the API key ({e.code} on {method} {path}): "
                                f"{detail}. Check OPEN_WEBUI_API_KEY.") from e
            raise SyncError(f"Open WebUI returned {e.code} on {method} {path}: {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            reason = getattr(e, "reason", e)
            raise SyncError(f"can't reach Open WebUI at {self.base_url} ({reason}) - "
                            "is it running? (`docker compose up -d`)") from e

    # Return the model with this id, or None if Open WebUI doesn't have it.
    def get_model(self, model_id):
        return self._request("GET", "/api/v1/models/model?id=" + urllib.parse.quote(model_id),
                             allow_404=True)

    # Create a new workspace model.
    def create_model(self, payload):
        self._request("POST", "/api/v1/models/create", payload)

    # Replace an existing workspace model's name, base model, meta, and params.
    def update_model(self, payload):
        self._request("POST", "/api/v1/models/model/update", payload)


# Parse args, sync the registry, and print one line per model.
def main(argv=None):
    parser = argparse.ArgumentParser(description="Mirror models.yaml tasks into Open WebUI.")
    parser.add_argument("--dry-run", action="store_true", help="show changes without applying them")
    args = parser.parse_args(argv)

    try:
        load_dotenv()
    except SyncError as e:
        sys.exit(f"error: {e}")
    api_key = os.getenv("OPEN_WEBUI_API_KEY")
    if not api_key:
        sys.exit("error: set OPEN_WEBUI_API_KEY in .env "
                 "(create one in Open WebUI under Settings -> Account -> API Keys)")
    base_url = os.getenv("OPEN_WEBUI_URL",
                         f"http://localhost:{os.getenv('OPEN_WEBUI_PORT', '3000')}")

    try:
        results = sync(load_registry(), WebUIClient(base_url, api_key), dry_run=args.dry_run)
    except (SyncError, RegistryError) as e:
        sys.exit(f"error: {e}")
    prefix = "[dry run] " if args.dry_run else ""
    for model_id, action in results:
        print(f"{prefix}{model_id}: {action}")


if __name__ == "__main__":
    main()
