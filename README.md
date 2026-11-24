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

## License

[MIT](LICENSE)
