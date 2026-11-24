"""Unit tests for client/chat.py — pure logic only, no network or Ollama needed.

Run from the repo root:  python3 -m unittest discover tests
"""
import contextlib
import copy
import io
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import openai

# Make `client` importable no matter which folder the tests are run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import chat, history  # noqa: E402  (must follow the sys.path tweak)

REGISTRY = {
    "default": "default-model",
    "tasks": {"code": {"model": "code-model", "temperature": 0.2}},
}


class OllamaBaseUrlTests(unittest.TestCase):
    # Return the base URL computed with OLLAMA_BASE_URL set to value (None = unset).
    def url_for(self, value):
        env = {k: v for k, v in os.environ.items() if k != "OLLAMA_BASE_URL"}
        if value is not None:
            env["OLLAMA_BASE_URL"] = value
        with mock.patch.dict(os.environ, env, clear=True):
            return chat.ollama_base_url()

    # Unset env var falls back to local Ollama's OpenAI-compatible endpoint.
    def test_default_is_local_ollama_v1(self):
        self.assertEqual(self.url_for(None), "http://localhost:11434/v1")

    # Ollama-style URL without /v1 (as Open WebUI uses) gets /v1 appended.
    def test_appends_v1_when_missing(self):
        self.assertEqual(self.url_for("http://localhost:11434"), "http://localhost:11434/v1")

    # URLs already ending in /v1, with or without a trailing slash, are kept as-is.
    def test_keeps_existing_v1(self):
        self.assertEqual(self.url_for("http://host:11434/v1/"), "http://host:11434/v1")
        self.assertEqual(self.url_for("http://host:11434/v1"), "http://host:11434/v1")


class PickTests(unittest.TestCase):
    # A task listed in the registry returns its own config.
    def test_known_task(self):
        self.assertEqual(chat.pick("code", REGISTRY), {"model": "code-model", "temperature": 0.2})

    # An unknown task falls back to the default model.
    def test_unknown_task_falls_back_to_default(self):
        self.assertEqual(chat.pick("nope", REGISTRY)["model"], "default-model")


class SystemPromptTests(unittest.TestCase):
    # A task's own system prompt is used when the registry gives one.
    def test_uses_task_prompt(self):
        self.assertEqual(chat.system_prompt({"system": "be terse"}), "be terse")

    # Tasks with no (or an empty) system prompt use the built-in one.
    def test_falls_back_to_builtin_prompt(self):
        self.assertEqual(chat.system_prompt({}), chat.SYSTEM_PROMPT)
        self.assertEqual(chat.system_prompt({"system": ""}), chat.SYSTEM_PROMPT)

    # One-shot mode sends the task's prompt as the system message.
    def test_one_shot_sends_task_prompt(self):
        registry = {"default": "d", "tasks": {"code": {"model": "c", "temperature": 0,
                                                       "system": "code prompt"}}}
        api = FakeApi("ok")
        with mock.patch.object(chat, "load_registry", return_value=registry), \
                mock.patch.object(chat, "client", api), \
                contextlib.redirect_stdout(io.StringIO()):
            chat.chat("hi", task="code")
        self.assertEqual(api.requests[0]["messages"][0],
                         {"role": "system", "content": "code prompt"})


class RegistryTests(unittest.TestCase):
    # models.yaml is found even when the working directory is elsewhere.
    def test_loads_from_any_working_directory(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                registry = chat.load_registry()
            finally:
                os.chdir(original)
        self.assertIn("tasks", registry)
        self.assertIn("default", registry)


class RequiredModelsTests(unittest.TestCase):
    # Default first, then each task's model in file order, each listed once.
    def test_default_then_tasks_deduplicated(self):
        registry = {"default": "a", "tasks": {"x": {"model": "b"}, "y": {"model": "a"},
                                              "z": {"model": "b"}, "w": {"model": "c"}}}
        self.assertEqual(chat.required_models(registry), ["a", "b", "c"])

    # A registry with no tasks still needs its default model.
    def test_no_tasks_needs_only_default(self):
        self.assertEqual(chat.required_models({"default": "a"}), ["a"])
        self.assertEqual(chat.required_models({"default": "a", "tasks": None}), ["a"])

    # The real models.yaml lists the models bootstrap.sh will pull.
    def test_real_registry(self):
        models = chat.required_models(chat.load_registry())
        self.assertIn(chat.load_registry()["default"], models)
        self.assertEqual(len(models), len(set(models)))


class CliArgumentTests(unittest.TestCase):
    # Run main() with argv and return its exit code, silencing argparse's usage output.
    def exit_code(self, argv):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            chat.main(argv)
        return ctx.exception.code

    # No arguments is a usage error, not a crash.
    def test_no_arguments_is_usage_error(self):
        self.assertEqual(self.exit_code([]), 2)

    # A task without a prompt starts the REPL with the history store, and closes it afterwards.
    def test_task_without_prompt_starts_repl(self):
        with mock.patch.object(chat, "repl") as repl, mock.patch.object(chat, "chat") as one_shot, \
                mock.patch.object(history, "open_store") as open_store:
            chat.main(["fast"])
        repl.assert_called_once_with("fast", store=open_store.return_value)
        open_store.return_value.close.assert_called_once()
        one_shot.assert_not_called()

    # If the history database can't be opened, the REPL still runs, just without saving.
    def test_repl_runs_without_history_when_db_unavailable(self):
        err = io.StringIO()
        with mock.patch.object(chat, "repl") as repl, contextlib.redirect_stderr(err), \
                mock.patch.object(history, "open_store",
                                  side_effect=sqlite3.OperationalError("unable to open")):
            chat.main(["fast"])
        repl.assert_called_once_with("fast", store=None)
        self.assertIn("history won't be saved", err.getvalue())

    # One-shot questions are not persisted: the history database is never opened.
    def test_one_shot_does_not_open_history(self):
        with mock.patch.object(chat, "chat"), mock.patch.object(history, "open_store") as open_store:
            chat.main(["fast", "hello"])
        open_store.assert_not_called()

    # Any Ollama API error in one-shot mode (here a 500) becomes a one-line error, not a traceback.
    def test_one_shot_server_error_is_one_line(self):
        response = httpx.Response(500, request=httpx.Request("POST", "http://ollama"))
        server_error = openai.InternalServerError("model failed to load", response=response, body=None)
        with mock.patch.object(chat, "chat", side_effect=server_error), \
                self.assertRaises(SystemExit) as ctx:
            chat.main(["fast", "hello"])
        self.assertEqual(ctx.exception.code, "error: model failed to load")


# Build a fake streaming chunk: text content, or a final usage-only chunk.
def fake_chunk(content=None, usage=None):
    choices = [] if content is None else [SimpleNamespace(delta=SimpleNamespace(content=content))]
    return SimpleNamespace(choices=choices, usage=usage)


class FakeApi:
    """Stands in for the OpenAI client: replays scripted replies and records each request."""

    # Store scripted replies (strings, or exceptions to raise) and an empty call log.
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    # Record a snapshot of the request and stream back the next scripted reply.
    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=3,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=90))
        return iter([fake_chunk(word) for word in reply.split(" ")] + [fake_chunk(usage=usage)])


CFG = {"model": "m", "temperature": 0}


# Return a connection error like the OpenAI client raises when Ollama is down.
def connection_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "http://ollama"))


class StreamReplyTests(unittest.TestCase):
    # The streamed pieces are echoed and joined, and usage (incl. cached tokens) is read from the last chunk.
    def test_joins_text_and_reads_usage(self):
        out = io.StringIO()
        text, stats = chat.stream_reply(chat.new_history(), CFG, api=FakeApi("a b"), out=out)
        self.assertEqual(text, "ab")
        self.assertEqual(out.getvalue(), "ab\n")
        self.assertEqual((stats.prompt_tokens, stats.cached_tokens, stats.completion_tokens),
                         (100, 90, 3))
        self.assertIsNotNone(stats.ttft)

    # Usage is requested from the server so the stats line has real token counts.
    def test_requests_usage_in_stream(self):
        api = FakeApi("x")
        chat.stream_reply(chat.new_history(), CFG, api=api, out=io.StringIO())
        self.assertEqual(api.requests[0]["stream_options"], {"include_usage": True})


class TakeTurnTests(unittest.TestCase):
    # Each turn sends the whole conversation so far, then stores the assistant reply.
    def test_history_accumulates_across_turns(self):
        api, messages = FakeApi("hello there", "fine"), chat.new_history()
        chat.take_turn(messages, "hi", CFG, api=api, out=io.StringIO())
        chat.take_turn(messages, "how are you", CFG, api=api, out=io.StringIO())
        self.assertEqual([m["role"] for m in api.requests[1]["messages"]],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1], {"role": "assistant", "content": "fine"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "hellothere"})

    # A failed request rolls the user message back so history never ends unanswered.
    def test_failed_turn_is_rolled_back(self):
        messages = chat.new_history()
        with self.assertRaises(openai.APIConnectionError):
            chat.take_turn(messages, "hi", CFG, api=FakeApi(connection_error()), out=io.StringIO())
        self.assertEqual(messages, chat.new_history())


# A fixed registry for REPL tests, so editing models.yaml's prompts can't break them.
REPL_REGISTRY = {
    "default": "default-model",
    "tasks": {"chat": {"model": "chat-model", "temperature": 0.7, "system": "test prompt"}},
}
REPL_START = chat.new_history("test prompt")


class ReplTests(unittest.TestCase):
    # Serve REPL_REGISTRY instead of reading models.yaml for every test in this class.
    def setUp(self):
        patcher = mock.patch.object(chat, "load_registry", return_value=REPL_REGISTRY)
        patcher.start()
        self.addCleanup(patcher.stop)

    # Drive the REPL with scripted input lines (ending in Ctrl-D) and return (api, stderr text).
    def run_repl(self, lines, *replies, store=None):
        api, err = FakeApi(*replies), io.StringIO()
        inputs = iter(lines)

        # Return the next scripted line, or raise EOFError like input() does on Ctrl-D.
        def read(_prompt):
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError from None

        chat.repl("chat", read=read, api=api, out=io.StringIO(), err=err, store=store)
        return api, err.getvalue()

    # Open an in-memory history store that is closed when the test ends.
    def memory_store(self):
        store = history.open_store(":memory:")
        self.addCleanup(store.close)
        return store

    # Return every saved session as a list of messages, in the order the sessions started.
    def saved_sessions(self, store):
        ids = [row[0] for row in store.execute(
            "SELECT session_id FROM messages GROUP BY session_id ORDER BY MIN(id)")]
        return [history.load_messages(store, session_id) for session_id in ids]

    # Each completed turn is saved; the system prompt is stored once, at the start of the session.
    def test_saves_completed_turns(self):
        store = self.memory_store()
        self.run_repl(["one", "two"], "r1", "r2", store=store)
        self.assertEqual(self.saved_sessions(store), [REPL_START + [
            {"role": "user", "content": "one"}, {"role": "assistant", "content": "r1"},
            {"role": "user", "content": "two"}, {"role": "assistant", "content": "r2"}]])
        expected_model = "chat-model"
        models = {row[0] for row in store.execute("SELECT model FROM messages")}
        self.assertEqual(models, {expected_model})

    # Failed and Ctrl-C'd turns leave nothing in the database, matching the in-memory rollback.
    def test_failed_and_interrupted_turns_are_not_saved(self):
        store = self.memory_store()
        self.run_repl(["one", "two", "three"], connection_error(), KeyboardInterrupt(), "r3",
                      store=store)
        self.assertEqual(self.saved_sessions(store), [REPL_START + [
            {"role": "user", "content": "three"}, {"role": "assistant", "content": "r3"}]])

    # /reset starts a new session id, each with its own system prompt.
    def test_reset_starts_new_session(self):
        store = self.memory_store()
        self.run_repl(["one", "/reset", "two"], "r1", "r2", store=store)
        self.assertEqual(self.saved_sessions(store), [
            REPL_START + [{"role": "user", "content": "one"},
                                  {"role": "assistant", "content": "r1"}],
            REPL_START + [{"role": "user", "content": "two"},
                                  {"role": "assistant", "content": "r2"}]])

    # A database failure is reported as a warning and the conversation carries on.
    def test_save_failure_warns_and_chat_continues(self):
        store = history.open_store(":memory:")
        store.close()  # any write now raises sqlite3.ProgrammingError
        api, err = self.run_repl(["one", "two"], "r1", "r2", store=store)
        self.assertIn("couldn't save history", err)
        self.assertEqual(len(api.requests), 2)
        self.assertEqual(len(api.requests[1]["messages"]), 4)  # history still kept in memory

    # /reset drops earlier turns; blank lines are ignored and send nothing.
    def test_reset_clears_history(self):
        api, _ = self.run_repl(["one", "", "/reset", "two"], "r1", "r2")
        self.assertEqual(len(api.requests), 2)
        self.assertEqual(api.requests[1]["messages"],
                         REPL_START + [{"role": "user", "content": "two"}])

    # An API error is reported and the session continues with clean history.
    def test_error_keeps_session_alive(self):
        api, err = self.run_repl(["one", "two"], connection_error(), "r2")
        self.assertIn("can't reach Ollama", err)
        self.assertEqual([m["content"] for m in api.requests[1]["messages"][1:]], ["two"])

    # /exit stops reading input even when more lines would follow.
    def test_exit_command(self):
        api, _ = self.run_repl(["/exit", "never sent"])
        self.assertEqual(api.requests, [])


class FormatStatsTests(unittest.TestCase):
    # Build stats with the given prompt size and fixed, plausible timings.
    def stats(self, prompt_tokens):
        return chat.TurnStats(ttft=0.5, total=1.5, prompt_tokens=prompt_tokens,
                              cached_tokens=0, completion_tokens=11)

    # The line shows prompt size, time to first token, and decode speed.
    def test_shows_key_numbers(self):
        line = chat.format_stats(3, self.stats(1234))
        self.assertIn("turn 3", line)
        self.assertIn("prompt 1,234 tok", line)
        self.assertIn("ttft 0.50s", line)
        self.assertIn("10.0 tok/s", line)  # (11 - 1) tokens over 1.0s of decoding
        self.assertNotIn("warning", line)

    # Nearing the context window adds a warning about silent truncation.
    def test_warns_near_context_limit(self):
        self.assertIn("warning", chat.format_stats(9, self.stats(chat.CONTEXT_TOKENS)))


if __name__ == "__main__":
    unittest.main()
