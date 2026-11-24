# llm_work_station

A local LLM workstation: route tasks (chat, code, fast) to locally run models
served by [Ollama](https://ollama.com), with the task-to-model mapping kept in
`models.yaml`.

> **Status:** early work in progress — most files are scaffolding.

## Layout

| Path | Purpose |
|---|---|
| `models.yaml` | Default model and per-task model, description, and temperature |
| `client/` | Python client code: `chat.py` (CLI/REPL), `history.py` (SQLite history) |
| `data/history.db` | Interactive chat history, created on first use (gitignored; override with `CHAT_HISTORY_DB`) |
| `sync_webui.py` | Mirrors `models.yaml` tasks into Open WebUI as custom models |
| `scripts.py` | Helper scripts |
| `docker-compose.yml` | Container setup |
| `tests/` | Tests |
| `.env.example` | Template for local environment variables — copy to `.env` (gitignored) |
| `LEARNING.md` | Notes |

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- The models listed in `models.yaml` pulled, e.g. `ollama pull llama3.2:3b`

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

## License

[MIT](LICENSE)
