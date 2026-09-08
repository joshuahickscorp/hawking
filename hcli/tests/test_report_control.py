"""The control is the whole test.

A second identical request being fast proves nothing on its own -- a warm
process, a warm allocator and a filled page cache all look exactly like a reused
prefix. `warm_turns` is only allowed to say "prefix reuse" when a DIFFERENT
prompt of the same size stayed slow. These drive that decision with a fake
surface, because the three outcomes it has to separate cannot be produced on
demand from a real one.
"""
from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hcli.report import warm_turns


def _surface(latency_for):
    """A fake OpenAI surface whose latency is a function of the prompt text."""
    import time as _time

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            text = "".join(m.get("content", "") for m in body.get("messages", []))
            _time.sleep(latency_for(text))
            payload = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": max(1, len(text) // 4),
                          "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


class TestWarmTurnsNeedsItsControl(unittest.TestCase):
    def test_a_real_prefix_hit_is_reported(self):
        # Slow the first time a body of text is seen, fast for anything that
        # EXTENDS it -- and slow again for the unrelated control.
        seen = []

        def latency(text):
            if "unrelated" in text.lower():
                return 0.30
            hit = any(text.startswith(p) for p in seen)
            seen.append(text)
            return 0.01 if hit else 0.30

        httpd, base = _surface(latency)
        self.addCleanup(httpd.shutdown)
        got = warm_turns(base, size=40, turns=3, timeout=30)
        self.assertTrue(got["prefix_reuse"], got)
        self.assertIn("does NOT re-pay", got["verdict"])
        self.assertGreaterEqual(got["speedup"], 2.0)

    def test_no_reuse_is_reported_as_no_reuse(self):
        httpd, base = _surface(lambda text: 0.20)
        self.addCleanup(httpd.shutdown)
        got = warm_turns(base, size=40, turns=3, timeout=30)
        self.assertFalse(got["prefix_reuse"], got)
        self.assertIn("re-pays its history", got["verdict"])

    def test_a_fast_control_makes_the_claim_UNDECIDED(self):
        # THE DEFECT THIS EXISTS FOR: everything got faster after the first
        # call (a warm process), so turn 2 looks like a prefix hit. It is not
        # attributable, and the report must say so rather than claim reuse.
        calls = []

        def latency(text):
            calls.append(text)
            return 0.30 if len(calls) == 1 else 0.01

        httpd, base = _surface(latency)
        self.addCleanup(httpd.shutdown)
        got = warm_turns(base, size=40, turns=3, timeout=30)
        self.assertFalse(got["prefix_reuse"],
                         "a warm process was reported as prefix reuse")
        self.assertIn("UNDECIDED", got["verdict"])

    def test_the_control_row_is_always_recorded(self):
        httpd, base = _surface(lambda text: 0.05)
        self.addCleanup(httpd.shutdown)
        got = warm_turns(base, size=40, turns=2, timeout=30)
        kinds = [r["kind"] for r in got["turns"]]
        self.assertTrue(any("control" in k for k in kinds), kinds)
        self.assertIsNotNone(got["control_s"])


if __name__ == "__main__":
    unittest.main()
