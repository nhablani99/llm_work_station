"""Unit tests for sync_webui.py — no network or Open WebUI needed.

Run from the repo root:  python3 -m unittest discover tests
"""
import io
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sync_webui  # noqa: E402  (must follow the sys.path tweak)

CODE_CFG = {"model": "qwen2.5-coder:7b", "description": "Writes code.",
            "temperature": 0.2, "system": "Code first."}
REGISTRY = {"default": "d", "tasks": {"code": CODE_CFG}}


class FakeWebUI:
    """Stands in for WebUIClient: holds models by id and records writes."""

    # Start with the given existing models, keyed by id.
    def __init__(self, models=None):
        self.models = dict(models or {})
        self.created, self.updated = [], []

    # Return a stored model, or None like a 404.
    def get_model(self, model_id):
        return self.models.get(model_id)

    # Record and store a created model.
    def create_model(self, payload):
        self.created.append(payload)
        self.models[payload["id"]] = payload

    # Record and store an updated model.
    def update_model(self, payload):
        self.updated.append(payload)
        self.models[payload["id"]] = payload


class ToWebUIModelTests(unittest.TestCase):
    # A task maps to a stable id, its base model, and its prompt and temperature as params.
    def test_maps_task_fields(self):
        self.assertEqual(sync_webui.to_webui_model("code", CODE_CFG), {
            "id": "task-code",
            "base_model_id": "qwen2.5-coder:7b",
            "name": "Code (qwen2.5-coder:7b)",
            "meta": {"description": "Writes code."},
            "params": {"temperature": 0.2, "system": "Code first."},
        })

    # A task with no system prompt sends no system param, leaving the base model's own.
    def test_omits_missing_system_prompt(self):
        cfg = {"model": "m", "temperature": 0.5}
        self.assertEqual(sync_webui.to_webui_model("fast", cfg)["params"], {"temperature": 0.5})


class SyncTests(unittest.TestCase):
    # Models missing from Open WebUI are created.
    def test_creates_missing_model(self):
        api = FakeWebUI()
        self.assertEqual(sync_webui.sync(REGISTRY, api), [("task-code", "created")])
        self.assertEqual(api.created, [sync_webui.to_webui_model("code", CODE_CFG)])

    # A model that already matches the registry is left alone (no new history entry).
    def test_skips_unchanged_model(self):
        desired = sync_webui.to_webui_model("code", CODE_CFG)
        # Real responses carry extra fields (user_id, timestamps, null meta keys); ignore them.
        existing = {**desired, "user_id": "u1", "updated_at": 1,
                    "meta": {**desired["meta"], "profile_image_url": None}}
        api = FakeWebUI({"task-code": existing})
        self.assertEqual(sync_webui.sync(REGISTRY, api), [("task-code", "unchanged")])
        self.assertEqual(api.updated, [])

    # A changed prompt, or a param added by hand in the UI, is overwritten by the registry.
    def test_updates_drifted_model(self):
        desired = sync_webui.to_webui_model("code", CODE_CFG)
        for drift in ({"system": "old prompt"}, {"num_ctx": 8192}):
            with self.subTest(drift=drift):
                existing = {**desired, "params": {**desired["params"], **drift}}
                api = FakeWebUI({"task-code": existing})
                self.assertEqual(sync_webui.sync(REGISTRY, api), [("task-code", "updated")])
                self.assertEqual(api.updated, [desired])

    # Dry run reports what would happen but writes nothing.
    def test_dry_run_writes_nothing(self):
        api = FakeWebUI()
        self.assertEqual(sync_webui.sync(REGISTRY, api, dry_run=True), [("task-code", "created")])
        self.assertEqual((api.created, api.updated), ([], []))


class LoadDotenvTests(unittest.TestCase):
    # Parse a .env body and return the resulting environment, starting from `existing`.
    def load(self, text, existing=None):
        environ = dict(existing or {})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".env"
            path.write_text(text)
            sync_webui.load_dotenv(path, environ)
        return environ

    # KEY=VALUE lines are read; comments, blanks and quotes are handled.
    def test_parses_values(self):
        env = self.load("# comment\n\nA=1\nB='two'\nC=\"x=y\"\n")
        self.assertEqual(env, {"A": "1", "B": "two", "C": "x=y"})

    # Variables already set in the environment win over .env.
    def test_environment_wins(self):
        self.assertEqual(self.load("A=file", {"A": "env"}), {"A": "env"})

    # A missing .env is not an error.
    def test_missing_file_is_ignored(self):
        environ = {}
        sync_webui.load_dotenv(Path("/nonexistent/.env"), environ)
        self.assertEqual(environ, {})


class WebUIClientTests(unittest.TestCase):
    # Return an HTTPError with the given status code and body.
    def http_error(self, code, body=b'{"detail":"nope"}'):
        return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body))

    # A 404 on lookup means "not there yet", so the model gets created.
    def test_get_missing_model_returns_none(self):
        api = sync_webui.WebUIClient("http://x", "k")
        with mock.patch("urllib.request.urlopen", side_effect=self.http_error(404)):
            self.assertIsNone(api.get_model("task-code"))

    # A bad key gives a message that names the variable to fix.
    def test_auth_failure_mentions_key(self):
        api = sync_webui.WebUIClient("http://x", "k")
        with mock.patch("urllib.request.urlopen", side_effect=self.http_error(401)):
            with self.assertRaisesRegex(sync_webui.SyncError, "OPEN_WEBUI_API_KEY"):
                api.create_model({})

    # A 404 on a write is an error, not silently ignored.
    def test_404_on_write_is_an_error(self):
        api = sync_webui.WebUIClient("http://x", "k")
        with mock.patch("urllib.request.urlopen", side_effect=self.http_error(404)):
            with self.assertRaisesRegex(sync_webui.SyncError, "404"):
                api.update_model({})

    # Open WebUI being down gives a hint to start it.
    def test_unreachable_server_hints_docker(self):
        api = sync_webui.WebUIClient("http://x", "k")
        error = urllib.error.URLError(ConnectionRefusedError("refused"))
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(sync_webui.SyncError, "docker compose up"):
                api.get_model("task-code")


class MainTests(unittest.TestCase):
    # Without an API key, main exits with instructions instead of calling Open WebUI.
    def test_missing_key_exits_with_hint(self):
        with mock.patch.object(sync_webui, "load_dotenv"), \
                mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SystemExit) as cm:
                sync_webui.main([])
        self.assertIn("OPEN_WEBUI_API_KEY", str(cm.exception.code))


if __name__ == "__main__":
    unittest.main()
