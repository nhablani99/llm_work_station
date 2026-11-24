import argparse
import difflib
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import openai
import yaml
from openai import OpenAI

# Run as `python3 client/chat.py`, only client/ is on sys.path; add the repo root so
# `client.history` imports the same way it does under the tests and `python3 -m client.chat`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import history  # noqa: E402  (must follow the sys.path tweak)

try:
    import readline  # noqa: F401  (gives input() arrow-key editing and history)
except ImportError:  # not available on every platform; the REPL works without it
    pass

# models.yaml sits at the repo root, one level above client/. Resolving it from
# this file (not the working directory) lets the script run from any folder.
REGISTRY_PATH = Path(__file__).resolve().parent.parent / "models.yaml"

# The only settings models.yaml may contain; anything else is reported as a likely typo.
TOP_LEVEL_SETTINGS = ("default", "tasks")
TASK_SETTINGS = ("model", "temperature", "system", "description")
MAX_TEMPERATURE = 2  # the OpenAI-compatible API's range is 0 to 2

SYSTEM_PROMPT = "You are a concise assistant."

# Ollama's default num_ctx. The OpenAI-compatible endpoint can't change it, and
# past it Ollama silently drops the oldest tokens, so warn before getting there.
CONTEXT_TOKENS = 4096
CONTEXT_WARN_FRACTION = 0.85


# Return Ollama's OpenAI-compatible URL, accepting OLLAMA_BASE_URL with or without /v1.
def ollama_base_url():
    # Open WebUI and most Ollama docs set this variable *without* /v1, but the
    # OpenAI client needs it, so normalize rather than 404 on a common config.
    url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


client = OpenAI(base_url=ollama_base_url(), api_key="ollama")  # key required, ignored


class RegistryError(Exception):
    """models.yaml is missing or unusable, with a message fit for the user."""


# Load the task-to-model registry from models.yaml; raise RegistryError if it can't be used.
def load_registry(path=REGISTRY_PATH):
    try:
        with open(path) as f:
            registry = yaml.safe_load(f)
    except OSError as e:
        raise RegistryError(f"can't read {path} ({e.strerror})") from e
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" line {mark.line + 1}" if mark else ""
        raise RegistryError(f"{path}{where}: invalid YAML ({getattr(e, 'problem', None) or e})") from e
    # No default means no model to fall back to; guessing one would hide a broken file.
    if registry is None:
        raise RegistryError(f"{path} is empty - it needs at least `default: <model>`")
    if not isinstance(registry, dict):
        raise RegistryError(f"{path} must hold settings like `default: <model>`, "
                            f"but its top level is a {type(registry).__name__}")
    problems = registry_problems(registry)
    if len(problems) == 1:
        raise RegistryError(f"{path}: {problems[0]}")
    if problems:
        raise RegistryError(f"{path} has {len(problems)} problems:\n  " + "\n  ".join(problems))
    # YAML reads an empty `tasks:` section as None; normalizing here means callers can
    # use registry["tasks"] directly instead of each guarding against it.
    registry["tasks"] = registry.get("tasks") or {}
    return registry


# Return " (did you mean `x`?)" for the allowed name closest to a misspelled key, or "".
def did_you_mean(key, allowed):
    close = difflib.get_close_matches(str(key), allowed, n=1)
    return f" (did you mean `{close[0]}`?)" if close else ""


# Return True if value is non-blank text, as model names must be.
def is_name(value):
    return isinstance(value, str) and bool(value.strip())


# List every problem with a parsed registry's settings, as readable messages (empty when valid).
def registry_problems(registry):
    # Unknown keys are errors, not ignored: a typo like `task:` or `sytem:` would
    # otherwise silently drop the tasks or the prompt it was meant to set.
    problems = [f"unknown top-level setting `{key}`{did_you_mean(key, TOP_LEVEL_SETTINGS)}"
                for key in registry if key not in TOP_LEVEL_SETTINGS]
    default = registry.get("default")
    if not default:
        problems.append("no `default:` model")
    elif not is_name(default):
        problems.append(f"`default` must be a model name like llama3.2:3b, got {default!r}")
    tasks = registry.get("tasks") or {}
    if not isinstance(tasks, dict):
        problems.append(f"`tasks` must hold one entry per task, like `chat:`, "
                        f"not a {type(tasks).__name__}")
        return problems
    for name, cfg in tasks.items():
        problems += task_problems(name, cfg)
    return problems


# List every problem with one task's name and settings, each prefixed with where it is.
def task_problems(name, cfg):
    where, problems = f"tasks.{name}", []
    # Task names become Open WebUI model ids, which can't contain whitespace.
    if not isinstance(name, str) or not name or any(c.isspace() for c in name):
        hint = (" (YAML reads yes/no/on/off as true/false; quote the name, e.g. 'yes':)"
                if isinstance(name, bool) else "")
        problems.append(f"{where}: task names must be text with no spaces, got {name!r}{hint}")
    if not isinstance(cfg, dict):
        return problems + [f"{where}: must hold settings like `model:` and `temperature:` "
                           f"on their own indented lines, not {cfg!r}"]
    if "model" not in cfg:
        problems.append(f"{where}: missing `model`")
    elif not is_name(cfg["model"]):
        problems.append(f"{where}: `model` must be a model name like llama3.2:3b, got {cfg['model']!r}")
    temperature = cfg.get("temperature")
    if "temperature" not in cfg:
        problems.append(f"{where}: missing `temperature` (a number from 0 to {MAX_TEMPERATURE})")
    # bool is a subclass of int in Python, so `temperature: true` must be excluded explicitly.
    elif (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
          or not 0 <= temperature <= MAX_TEMPERATURE):
        problems.append(f"{where}: `temperature` must be a number from 0 to {MAX_TEMPERATURE}, "
                        f"got {temperature!r}")
    for key in ("system", "description"):  # optional; an empty `system:` reads as None
        if cfg.get(key) is not None and not isinstance(cfg[key], str):
            problems.append(f"{where}: `{key}` must be text, got {cfg[key]!r}")
    problems += [f"{where}: unknown setting `{key}`{did_you_mean(key, TASK_SETTINGS)}"
                 for key in cfg if key not in TASK_SETTINGS]
    return problems


# List every model the registry needs pulled: the default, then each task's, without repeats.
def required_models(registry):
    models = [registry["default"]]
    models += [cfg["model"] for cfg in (registry.get("tasks") or {}).values()]
    return list(dict.fromkeys(models))  # dedupe, keeping first-seen order


# Route a task name to its model config; fall back to the default model.
def pick(task, registry):
    return registry["tasks"].get(task, {"model": registry["default"], "temperature": 0.7})


# Return the task's own system prompt from the registry, or the built-in one if it has none.
def system_prompt(cfg):
    return cfg.get("system") or SYSTEM_PROMPT


@dataclass(frozen=True)
class TurnStats:
    """Timing and token counts for one streamed reply (None = server didn't report it)."""
    ttft: float | None           # seconds until the first text token
    total: float                 # seconds for the whole request
    prompt_tokens: int | None    # full prompt size, cached or not
    cached_tokens: int | None    # part of the prompt Ollama reused from its KV cache
    completion_tokens: int | None

    # Return decode speed in tokens/s, leaving out the first token because it also pays for prefill.
    @property
    def tokens_per_second(self):
        if self.ttft is None or not self.completion_tokens or self.completion_tokens < 2:
            return None
        decode_seconds = self.total - self.ttft
        return (self.completion_tokens - 1) / decode_seconds if decode_seconds > 0 else None


# Stream a reply for the full message list to out and return (text, TurnStats).
def stream_reply(messages, cfg, api=None, out=None, max_tokens=None):
    api = api or client
    out = out or sys.stdout
    request = dict(model=cfg["model"], temperature=cfg["temperature"], messages=messages,
                   stream=True, stream_options={"include_usage": True})
    if max_tokens is not None:
        request["max_tokens"] = max_tokens

    start = time.perf_counter()
    first_token_at, usage, parts = None, None, []
    for chunk in api.chat.completions.create(**request):
        if chunk.choices:  # the final usage chunk (and some trailing ones) have no choices
            piece = chunk.choices[0].delta.content or ""
            if piece and first_token_at is None:
                first_token_at = time.perf_counter()
            parts.append(piece)
            out.write(piece)
            out.flush()
        if getattr(chunk, "usage", None):
            usage = chunk.usage
    end = time.perf_counter()
    out.write("\n")
    out.flush()  # the stats line goes to stderr; flush so it can't land mid-reply

    details = getattr(usage, "prompt_tokens_details", None)
    stats = TurnStats(
        ttft=None if first_token_at is None else first_token_at - start,
        total=end - start,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        cached_tokens=getattr(details, "cached_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
    )
    return "".join(parts), stats


# Stream a single question with no history (the one-shot CLI mode).
def chat(prompt, task="chat", system=None):
    cfg = pick(task, load_registry())
    system = system or system_prompt(cfg)
    stream_reply([{"role": "system", "content": system},
                  {"role": "user", "content": prompt}], cfg)


# Return a fresh conversation containing only the system prompt.
def new_history(system=SYSTEM_PROMPT):
    return [{"role": "system", "content": system}]


# Send one user turn with the whole history; append the reply, or undo the turn if it fails.
def take_turn(messages, text, cfg, api=None, out=None):
    messages.append({"role": "user", "content": text})
    try:
        reply, stats = stream_reply(messages, cfg, api=api, out=out)
    except BaseException:  # includes Ctrl-C: never leave an unanswered user message behind
        messages.pop()
        raise
    messages.append({"role": "assistant", "content": reply})
    return stats


# Format the per-turn stats line, plus a warning when the context window is nearly full.
def format_stats(turn, stats):
    parts = [f"turn {turn}"]
    if stats.prompt_tokens is not None:
        cached = f" ({stats.cached_tokens:,} cached)" if stats.cached_tokens is not None else ""
        parts.append(f"prompt {stats.prompt_tokens:,} tok{cached}")
    if stats.ttft is not None:
        parts.append(f"ttft {stats.ttft:.2f}s")
    if stats.tokens_per_second is not None:
        parts.append(f"{stats.tokens_per_second:.1f} tok/s")
    line = "[" + " · ".join(parts) + "]"
    if stats.prompt_tokens and stats.prompt_tokens >= CONTEXT_WARN_FRACTION * CONTEXT_TOKENS:
        line += (f"\n[warning: prompt is {stats.prompt_tokens / CONTEXT_TOKENS:.0%} of the "
                 f"{CONTEXT_TOKENS}-token context; past it Ollama silently drops the oldest "
                 "turns. /reset to start fresh]")
    return line


# Turn an OpenAI-client error from Ollama into a one-line, actionable message.
def error_message(exc, model):
    if isinstance(exc, openai.APIConnectionError):
        return (f"can't reach Ollama at {client.base_url} - is it running? "
                "(start the Ollama app or run `ollama serve`)")
    if isinstance(exc, openai.NotFoundError):
        return (f"Ollama returned 404 for model '{model}' ({exc.message}). "
                f"Pull it with `ollama pull {model}`.")
    return str(exc)


# Save the messages not yet stored for this session and return the new saved count.
def persist(store, session_id, messages, saved, model, err):
    # Skip when no completed turn is waiting: saving just the system prompt would
    # leave an empty session behind every chat that was quit without a turn.
    if store is None or not any(m["role"] == "user" for m in messages[saved:]):
        return saved
    try:
        history.save_messages(store, session_id, messages[saved:], model)
    except sqlite3.Error as e:
        # The chat matters more than the log. Leaving `saved` unchanged means the next
        # turn, /reset, or the end of the chat retries these messages.
        print(f"[warning: couldn't save history: {e}]", file=err)
        return saved
    return len(messages)


# Run an interactive chat that resends the full history every turn; /reset clears it.
# With a store, each completed turn is saved to SQLite under a per-conversation session id.
def repl(task, system=None, read=input, api=None, out=None, err=None, store=None):
    out, err = out or sys.stdout, err or sys.stderr
    cfg = pick(task, load_registry())
    system = system or system_prompt(cfg)
    messages = new_history(system)
    session_id, saved = history.new_session_id(), 0
    print(f"chatting with {cfg['model']} - /reset clears history, /exit or Ctrl-D quits",
          file=err)
    try:
        while True:
            try:
                text = read("> ").strip()
            except EOFError:
                print(file=out)
                return
            except KeyboardInterrupt:  # like Python's REPL: Ctrl-C abandons the line, not the chat
                print(file=out)
                continue
            if not text:
                continue
            if text in ("/exit", "/quit"):
                return
            if text == "/reset":
                persist(store, session_id, messages, saved, cfg["model"], err)  # last retry
                messages = new_history(system)
                session_id, saved = history.new_session_id(), 0
                print("[history cleared]", file=err)
                continue
            try:
                stats = take_turn(messages, text, cfg, api=api, out=out)
            except KeyboardInterrupt:
                print("\n[interrupted - turn discarded]", file=err)
                continue
            except openai.APIError as e:  # keep the session alive; the failed turn was rolled back
                print("error: " + error_message(e, cfg["model"]), file=err)
                continue
            saved = persist(store, session_id, messages, saved, cfg["model"], err)
            turn = sum(m["role"] == "user" for m in messages)
            print(format_stats(turn, stats), file=err)
    finally:
        # Last retry for turns whose save failed, however the chat ends (/exit, Ctrl-D, a crash).
        persist(store, session_id, messages, saved, cfg["model"], err)


# Open the history database, or warn and return None so the chat still runs without saving.
def open_history():
    try:
        return history.open_store()
    except (sqlite3.Error, OSError) as e:
        print(f"[warning: history won't be saved: {e}]", file=sys.stderr)
        return None


# Parse CLI args, run one-shot or REPL mode, and turn common failures into one-line errors.
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Chat with a local Ollama model chosen by task (see models.yaml). "
                    "With no prompt, starts an interactive chat that remembers the conversation.")
    parser.add_argument("task", help="task from models.yaml, e.g. chat, code, fast; "
                                     "unknown names use the default model")
    parser.add_argument("prompt", nargs="*", help="the message to send; omit to start a chat")
    args = parser.parse_args(argv)

    if not args.prompt:
        store = open_history()
        try:
            repl(args.task, store=store)
        except RegistryError as e:
            sys.exit(f"error: {e}")
        finally:
            if store is not None:
                store.close()
        return
    try:
        chat(" ".join(args.prompt), task=args.task)
    except RegistryError as e:
        sys.exit(f"error: {e}")
    except openai.APIError as e:
        sys.exit("error: " + error_message(e, pick(args.task, load_registry())["model"]))
    except KeyboardInterrupt:
        print()
        sys.exit(130)


if __name__ == "__main__":
    main()
