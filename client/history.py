"""Persist chat history to SQLite (standard-library sqlite3; no server, one file).

Open WebUI keeps its own chats in its Docker volume; this store is only for the
terminal client in client/chat.py.
"""
import os
import sqlite3
import uuid
from pathlib import Path

# Resolved from this file, like models.yaml, so the REPL finds the same database
# whichever folder it is started from. data/ is gitignored: chats stay local.
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"

# CHECK on role keeps junk out of a table that is meant to be replayed to a model later.
# created_at is UTC ISO-8601 with milliseconds, so it sorts and compares as text.
SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY,
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content    TEXT    NOT NULL,
    model      TEXT,
    created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS messages_by_session ON messages (session_id, id);
"""


# Return the database path: CHAT_HISTORY_DB if set, otherwise data/history.db in the repo.
def db_path():
    override = os.getenv("CHAT_HISTORY_DB")
    return Path(override).expanduser() if override else DEFAULT_DB_PATH


# Open (creating folder, file, and table if needed) the history database and return the connection.
def open_store(path=None):
    path = db_path() if path is None else path
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # timeout: wait briefly instead of failing if another REPL is mid-write to the same file.
    conn = sqlite3.connect(path, timeout=5)
    try:
        conn.executescript(SCHEMA)
    except BaseException:
        conn.close()
        raise
    return conn


# Return a new random session id (32 hex characters).
def new_session_id():
    return uuid.uuid4().hex


# Insert messages for a session in one transaction: all rows are saved or none are.
def save_messages(conn, session_id, messages, model):
    rows = [(session_id, m["role"], m["content"], model) for m in messages]
    with conn:  # commits on success, rolls back on any error
        conn.executemany(
            "INSERT INTO messages (session_id, role, content, model) VALUES (?, ?, ?, ?)", rows)


# Return a session's messages in the order they were saved, in the chat API's {role, content} shape.
def load_messages(conn, session_id):
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id", (session_id,))
    return [{"role": role, "content": content} for role, content in rows]
