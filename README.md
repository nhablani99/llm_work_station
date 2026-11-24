# llm_work_station

Run open-source LLMs privately on your Mac. Each kind of work (chat, code,
quick answers) is a *task* in `models.yaml`, mapped to a model that
[Ollama](https://ollama.com) runs locally. You can use the tasks in two places:

- **Terminal:** `python3 client/chat.py code "..."` for a single question, or
  `python3 client/chat.py chat` for a conversation that's saved to local history.
- **Browser:** [Open WebUI](https://openwebui.com) at http://localhost:3000,
  where `sync_webui.py` sets up each task as a ready-to-use model.

The models run on this machine and chats are stored locally.
`scripts/bootstrap.sh` sets everything up in one command.

## Layout

| Path | Purpose |
|---|---|
| `models.yaml` | Default model and per-task model, description, and temperature |
| `client/` | Python client code: `chat.py` (CLI/REPL), `history.py` (SQLite history) |
| `data/history.db` | Interactive chat history, created on first use (gitignored; override with `CHAT_HISTORY_DB`) |
| `sync_webui.py` | Mirrors `models.yaml` tasks into Open WebUI as custom models |
| `scripts/` | `bootstrap.sh` (one-command setup), `backup.sh` / `restore.sh` (Open WebUI data) |
| `docker-compose.yml` | Open WebUI container (Ollama runs natively, outside Docker) |
| `tests/` | Tests |
| `.env.example` | Template for local environment variables — copy to `.env` (gitignored) |
| `LEARNING.md` | Notes |

## Quick start

Prerequisites: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
and the [Ollama app](https://ollama.com/download), both running, plus Python 3.10+.
Ollama runs natively, not in Docker, because Docker on macOS can't use the Apple GPU.

```sh
git clone https://github.com/nhablani99/llm_work_station.git
cd llm_work_station
scripts/bootstrap.sh
```

The script creates `.env` with a generated secret, installs the Python
dependencies into `.venv`, starts Open WebUI and waits until it's healthy,
then pulls every model in `models.yaml` that's missing. It's safe to re-run.

Then open http://localhost:3000 and create your account. The first account
becomes admin, and sign-up closes automatically after that. Use Admin Panel →
Users to add anyone else.

Open WebUI only listens on `127.0.0.1`. To reach it from other devices, remove
the `127.0.0.1:` prefix from `ports` in `docker-compose.yml`. Ollama's API has
no authentication, so keep it bound to localhost; that's its default.

## Chat history

`python3 client/chat.py chat` (interactive mode) saves each completed turn to
`data/history.db`. `/reset` starts a new session. One-shot questions aren't saved.
Open WebUI keeps its own chats in its Docker volume.

```sh
sqlite3 -header -column data/history.db \
  "SELECT session_id, role, substr(content, 1, 60), created_at FROM messages ORDER BY id"
```

## Open WebUI models

Each task in `models.yaml` gets its own custom model in Open WebUI
(Workspace → Models). Each one is named after its task, e.g. `Code (qwen2.5-coder:7b)`, and has
the task's base model, system prompt and temperature. The CLI uses the same
prompts, so a task behaves the same in both places.

1. Start Open WebUI: `docker compose up -d`, then open http://localhost:3000.
2. API keys are off by default. As admin, turn on Admin Panel → Settings →
   General → Enable API Keys. Then create a key under Settings → Account → API
   Keys and put it in `.env` as `OPEN_WEBUI_API_KEY`.
3. Sync, re-running whenever you edit `models.yaml`:

   ```sh
   python3 sync_webui.py --dry-run   # preview
   python3 sync_webui.py
   ```

`models.yaml` is the source of truth. Edits made in the UI to the `task-*`
models are overwritten on the next sync, so change prompts in the yaml file. If
you remove a task from `models.yaml`, its Open WebUI model stays; delete it in
the UI.

## Backups

Open WebUI's users, chats and settings live in a Docker volume. To back them up:

```sh
scripts/backup.sh                                   # -> backups/webui-<timestamp>.tgz
scripts/restore.sh backups/webui-<timestamp>.tgz    # asks first; add --yes to skip
```

Open WebUI is stopped for a few seconds while the backup runs, so the database
copy is consistent. Downloaded embedding models in `cache/` are skipped because
Open WebUI downloads them again. Before replacing anything, `restore.sh` checks
the archive and backs up the current data. `backups/` is gitignored because the
archives contain chats and password hashes. Copy them somewhere off this machine.

The CLI's own history in `data/history.db` is a plain file, so copy it directly.

## License

[MIT](LICENSE)
