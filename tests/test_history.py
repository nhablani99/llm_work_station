"""Unit tests for client/history.py — temporary SQLite files only, never the real history DB.

Run from the repo root:  python3 -m unittest discover tests
"""
import os
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Make `client` importable no matter which folder the tests are run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import history  # noqa: E402  (must follow the sys.path tweak)

REPO_ROOT = Path(__file__).resolve().parent.parent


class DbPathTests(unittest.TestCase):
    # With no override, the database lives in data/ at the repo root, whatever the working directory.
    def test_default_is_repo_data_folder(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("CHAT_HISTORY_DB", None)
            self.assertEqual(history.db_path(), REPO_ROOT / "data" / "history.db")

    # CHAT_HISTORY_DB points the store somewhere else (with ~ expanded).
    def test_env_var_overrides_default(self):
        with mock.patch.dict(os.environ, {"CHAT_HISTORY_DB": "~/chats/h.db"}):
            self.assertEqual(history.db_path(), Path.home() / "chats" / "h.db")


class StoreTests(unittest.TestCase):
    # Open a fresh store in a not-yet-existing subfolder of a temp dir removed after the test.
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "nested" / "history.db"
        self.conn = history.open_store(self.path)
        self.addCleanup(self.conn.close)

    # Opening creates the missing folder, the file, and the messages table with the agreed columns.
    def test_creates_folder_file_and_table(self):
        self.assertTrue(self.path.is_file())
        columns = [row[1] for row in self.conn.execute("PRAGMA table_info(messages)")]
        self.assertEqual(columns, ["id", "session_id", "role", "content", "model", "created_at"])

    # Messages come back in insertion order, and each session only returns its own.
    def test_round_trip_keeps_order_and_separates_sessions(self):
        first = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"},
                 {"role": "assistant", "content": "hello"}]
        later = [{"role": "user", "content": "again"}, {"role": "assistant", "content": "yes"}]
        other = [{"role": "user", "content": "elsewhere"}]
        history.save_messages(self.conn, "s1", first, "m")
        history.save_messages(self.conn, "s2", other, "m")
        history.save_messages(self.conn, "s1", later, "m")
        self.assertEqual(history.load_messages(self.conn, "s1"), first + later)
        self.assertEqual(history.load_messages(self.conn, "s2"), other)
        self.assertEqual(history.load_messages(self.conn, "missing"), [])

    # Each row records the model it was sent to and a UTC ISO-8601 timestamp.
    def test_records_model_and_utc_timestamp(self):
        history.save_messages(self.conn, "s", [{"role": "user", "content": "x"}], "llama3.2:3b")
        model, created_at = self.conn.execute("SELECT model, created_at FROM messages").fetchone()
        self.assertEqual(model, "llama3.2:3b")
        self.assertRegex(created_at, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")

    # Rows survive closing and reopening, and reopening doesn't recreate or wipe the table.
    def test_reopening_keeps_existing_rows(self):
        history.save_messages(self.conn, "s", [{"role": "user", "content": "kept"}], "m")
        self.conn.close()
        self.conn = history.open_store(self.path)
        self.assertEqual(history.load_messages(self.conn, "s"), [{"role": "user", "content": "kept"}])

    # A batch with an invalid row is rejected as a whole, so a turn is never half-saved.
    def test_bad_row_rolls_back_whole_batch(self):
        batch = [{"role": "user", "content": "ok"}, {"role": "tool", "content": "not allowed"}]
        with self.assertRaises(sqlite3.IntegrityError):
            history.save_messages(self.conn, "s", batch, "m")
        self.assertEqual(history.load_messages(self.conn, "s"), [])


class SessionIdTests(unittest.TestCase):
    # Session ids are unique 32-character hex strings (uuid4).
    def test_ids_are_unique_hex(self):
        a, b = history.new_session_id(), history.new_session_id()
        self.assertNotEqual(a, b)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{32}", a))


if __name__ == "__main__":
    unittest.main()
