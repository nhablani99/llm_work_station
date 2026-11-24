# llm_work_station

A local LLM workstation: route tasks (chat, code, fast) to locally run models
served by [Ollama](https://ollama.com), with the task-to-model mapping kept in
`models.yaml`.

> **Status:** early work in progress — most files are scaffolding.

## Layout

| Path | Purpose |
|---|---|
| `models.yaml` | Default model and per-task model, description, and temperature |
| `client/` | Python client code (e.g. `chat.py`) |
| `scripts.py` | Helper scripts |
| `docker-compose.yml` | Container setup |
| `tests/` | Tests |
| `.env.example` | Template for local environment variables — copy to `.env` (gitignored) |
| `LEARNING.md` | Notes |

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- The models listed in `models.yaml` pulled, e.g. `ollama pull llama3.2:3b`

## License

[MIT](LICENSE)
