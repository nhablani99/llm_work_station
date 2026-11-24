"""Unit tests for bench_history.py — Ollama is faked, so no network is needed.

Run from the repo root:  python3 -m unittest discover tests
"""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

# Make the repo root importable no matter which folder the tests are run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bench_history  # noqa: E402  (must follow the sys.path tweak)
from client.chat import TurnStats  # noqa: E402

CFG = {"model": "fake-model", "temperature": 0}


# Build a fake stream_reply that always returns this reply text and first-token time.
def fake_stream_reply(reply, ttft):
    stats = TurnStats(ttft=ttft, total=0.5, prompt_tokens=40, cached_tokens=0,
                      completion_tokens=5 if reply else 0)
    return mock.Mock(return_value=(reply, stats))


class RunTest(unittest.TestCase):
    # Run the benchmark against a fake model with stderr captured, returning the result rows.
    def run_bench(self, reply, ttft, turns=2):
        with mock.patch.object(bench_history, "stream_reply", fake_stream_reply(reply, ttft)), \
                contextlib.redirect_stderr(io.StringIO()):
            return bench_history.run(CFG, turns=turns, max_tokens=5, cold=True, sink=io.StringIO())

    # A reply with no text has no TTFT; the run must record None instead of crashing.
    def test_empty_reply_records_missing_ttft(self):
        rows = self.run_bench(reply="", ttft=None)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIsNone(row["ttft_warm_s"])
            self.assertIsNone(row["ttft_cold_s"])

    # Measured times are rounded to milliseconds for the table and CSV.
    def test_measured_ttft_is_rounded(self):
        rows = self.run_bench(reply="hello", ttft=0.123456)
        self.assertEqual(rows[0]["ttft_warm_s"], 0.123)
        self.assertEqual(rows[0]["ttft_cold_s"], 0.123)


if __name__ == "__main__":
    unittest.main()
