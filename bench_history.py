"""Measure how chat speed changes as conversation history grows.

Drives a scripted multi-turn conversation against local Ollama (fixed prompts,
temperature 0, capped reply length) and records per turn:
  - prompt size and how much of it Ollama served from its KV cache
  - time to first token (TTFT) and decode speed, warm (normal chat)
  - whether the model still recalls a code word from turn 1
Then replays each history prefix with a cache-busting system prompt to get the
cold-cache TTFT for the same prompt size.

Run from anywhere:  python3 bench_history.py --turns 20 --csv out.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from client.chat import (SYSTEM_PROMPT, CONTEXT_TOKENS, RegistryError,  # noqa: E402
                         load_registry, new_history, pick, stream_reply)

CODE_WORD = "MARMALADE"
RECALL_QUESTION = ("Quick check: what is the code word I gave you in my very first message? "
                   "Answer with just the word.")
COLORS = ["red", "green", "amber", "violet", "silver", "teal", "ochre"]
ANIMALS = ["otter", "heron", "badger", "lynx", "tortoise", "falcon", "gecko"]
CITIES = ["Lisbon", "Osaka", "Quito", "Tallinn", "Nairobi", "Hobart", "Calgary"]
FACTS_PER_TURN = 6


# Round a duration to milliseconds, keeping None when a reply had no text to time.
def seconds(value):
    return None if value is None else round(value, 3)


# Format a duration for the progress lines, or "n/a" when it wasn't measured.
def show(value):
    return "n/a" if value is None else f"{value:.2f}s"


# Build the deterministic user message for turn i (1-based): a block of made-up facts plus a question.
def turn_prompt(i):
    facts = [f"Note {i}.{j}: the {COLORS[(i + j) % 7]} {ANIMALS[(i * 3 + j) % 7]} from "
             f"{CITIES[(i * 5 + j) % 7]} weighs {(i * 37 + j * 11) % 90 + 10} kg and sleeps "
             f"{(i + j) % 9 + 3} hours a day."
             for j in range(1, FACTS_PER_TURN + 1)]
    preamble = f"Remember this: the code word is {CODE_WORD}.\n" if i == 1 else ""
    return (preamble + "\n".join(facts) +
            "\nIn about 60 words, which of these notes is most unusual and why?")


# Ask the recall question on top of the history (not stored) and report whether the code word comes back.
def recalls_code_word(messages, cfg, sink):
    probe = messages + [{"role": "user", "content": RECALL_QUESTION}]
    reply, _ = stream_reply(probe, cfg, out=sink, max_tokens=10)
    return CODE_WORD in reply.upper()


# Time the first token for a prompt with a unique system prompt, so no cached prefix can be reused.
def cold_ttft(messages, cfg, tag, sink):
    busted = [{"role": "system", "content": f"[cold run {tag}] {SYSTEM_PROMPT}"}] + messages[1:]
    _, stats = stream_reply(busted, cfg, out=sink, max_tokens=1)
    return stats.ttft


# Run the scripted conversation and return one result row per turn.
def run(cfg, turns, max_tokens, cold, sink):
    messages, rows, prefixes = new_history(), [], []
    for i in range(1, turns + 1):
        messages.append({"role": "user", "content": turn_prompt(i)})
        prefixes.append(list(messages))
        reply, stats = stream_reply(messages, cfg, out=sink, max_tokens=max_tokens)
        messages.append({"role": "assistant", "content": reply})
        rows.append({
            "turn": i,
            "sent_chars": sum(len(m["content"]) for m in prefixes[-1]),
            "prompt_tokens": stats.prompt_tokens,
            "cached_tokens": stats.cached_tokens,
            "ttft_warm_s": seconds(stats.ttft),
            "decode_tok_s": round(stats.tokens_per_second, 1) if stats.tokens_per_second else None,
            "completion_tokens": stats.completion_tokens,
            "recalls_turn1": recalls_code_word(messages, cfg, sink),
        })
        print(f"turn {i:>2}: prompt {stats.prompt_tokens} tok, ttft {show(stats.ttft)}",
              file=sys.stderr)
    # Cold probes run after the warm pass: each one evicts the slot's cache, which
    # would otherwise distort the next warm turn.
    for row, prefix in zip(rows, prefixes):
        row["ttft_cold_s"] = seconds(cold_ttft(prefix, cfg, row["turn"], sink)) if cold else None
        if cold:
            print(f"cold {row['turn']:>2}: ttft {show(row['ttft_cold_s'])}", file=sys.stderr)
    return rows


# Print the result rows as an aligned table.
def print_table(rows):
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.rjust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).rjust(widths[c]) for c in cols))


# Parse a command-line count that must be at least 1, for argparse.
def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


# Parse args, run the benchmark, print the table, and optionally write a CSV.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="chat", help="task from models.yaml (default: chat)")
    parser.add_argument("--turns", type=positive_int, default=20)
    parser.add_argument("--max-tokens", type=int, default=100, help="reply length cap per turn")
    parser.add_argument("--no-cold", action="store_true", help="skip the cold-cache replays")
    parser.add_argument("--csv", type=Path, help="also write results to this CSV file")
    args = parser.parse_args(argv)

    try:
        cfg = {**pick(args.task, load_registry()), "temperature": 0}
    except RegistryError as e:
        sys.exit(f"error: {e}")
    print(f"model {cfg['model']}, {args.turns} turns, context {CONTEXT_TOKENS} tokens",
          file=sys.stderr)
    with open(os.devnull, "w") as sink:
        rows = run(cfg, args.turns, args.max_tokens, not args.no_cold, sink)
    print_table(rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
