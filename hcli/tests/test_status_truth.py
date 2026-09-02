from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]

from hcli.commands import format_status
from hcli.controller import Controller, _http_json
from hcli.resources import MutationLock


class TestFormatStatus(unittest.TestCase):
    def test_one_screen_contains_required_fields(self):
        text = format_status(
            {
                "mission_id": "m-1",
                "phase": "running",
                "goal": "ship truthful status",
                "units_by_status": {
                    "ready": 1,
                    "running": 2,
                    "completed": 3,
                    "failed": 0,
                    "pending": 4,
                },
                "blocked_units": 4,
                "qwen": {
                    "health": "ok",
                    "resident": 4,
                    "active_decode": 1,
                    "queued": 0,
                    "n_ctx": 262144,
                    "prompt_tokens": 100,
                    "tps": None,
                },
                "grok": {
                    "admitted": 2,
                    "active": 2,
                    "queued": 0,
                    "done": 10,
                    "failed": 1,
                    "latency_s": 12.2,
                },
                "occupancy": {"GPU_DECODE": 1, "COMPILE": 0, "TEST": 0, "TOOL_WAIT": 0},
                "mutation": {
                    "held": False,
                    "pid": 99,
                    "owner": "wu-9",
                    "owner_display": "stale",
                    "waiters": 0,
                },
                "verifier_backlog": 3,
                "accepted_units_per_hour": 12.0,
                "checkpoint_age_s": 12,
                "watchdog": "GOAL_NOT_MET",
            }
        )
        self.assertNotIn("failed_restarting", text)
        self.assertNotIn("(none)", text)
        lines = text.splitlines()
        self.assertLessEqual(len(lines), 10)
        self.assertIn("mission m-1", text)
        self.assertIn("phase=running", text)
        self.assertIn("Goal: ship truthful status", text)
        self.assertIn("WU ready=1 running=2 blocked=4 completed=3", text)
        self.assertIn("Qwen health=ok resident=4 active=1", text)
        self.assertIn("n_ctx=262144", text)
        self.assertIn("Grok admitted=2 active=2 queued=0 done=10 failed=1", text)
        self.assertIn("CPU", text)
        self.assertIn("Mutation held=false pid=99 owner=stale", text)
        self.assertIn("Verifier backlog=3", text)
        # A precomputed rate with NO WINDOW behind it must print `unknown`.
        # This assertion used to demand "accepted/h=12.0", which enshrined the
        # exact defect that annualised a 12.4-second window into 1164 units per
        # hour. A rate whose window cannot be checked is not a rate.
        self.assertIn("accepted/h=unknown", text)
        self.assertNotIn("accepted/h=12.0", text)
        self.assertIn("ckpt=12s", text)
        self.assertIn("watchdog=GOAL_NOT_MET", text)
        self.assertNotIn("1787453169", text)

    def test_qwen_down_does_not_echo_intended_count(self):
        text = format_status(
            {
                "qwen": {
                    "health": "down",
                    "resident": 0,
                    "active_decode": 0,
                    "queued": 0,
                    "n_ctx": None,
                    "prompt_tokens": None,
                    "tps": None,
                },
                "grok": {"admitted": 1, "active": 0, "queued": 0, "done": 0, "failed": 0},
                "mutation": {"held": False, "waiters": 0},
            }
        )
        self.assertIn("Qwen health=down resident=0", text)
        self.assertNotIn("failed_restarting", text)


class TestQwenProbe(unittest.TestCase):
    def test_unreachable_endpoint_is_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = Controller(workspace=tmp, runtime_count=1)
            controller._qwen_endpoint = lambda: "http://127.0.0.1:1"
            snap = controller._qwen_pool_status()
            self.assertEqual(snap["health"], "down")
            self.assertEqual(snap["resident"], 0)
            self.assertIsNone(snap["n_ctx"])
            self.assertNotIn("failed_restarting", snap)
            controller.shutdown()

    def test_slots_and_health_are_used_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = Controller(workspace=tmp, runtime_count=1)

            def fake_http(url, timeout=0.4):
                if url.endswith("/health"):
                    return {"status": "ok"}
                if url.endswith("/slots"):
                    return [
                        {
                            "id": 0,
                            "n_ctx": 4096,
                            "is_processing": True,
                            "n_prompt_tokens": 12,
                        },
                        {
                            "id": 1,
                            "n_ctx": 4096,
                            "is_processing": False,
                            "n_prompt_tokens": 0,
                        },
                    ]
                return None

            with patch("hcli.controller._http_json", side_effect=fake_http):
                snap = controller._qwen_pool_status()
            self.assertEqual(snap["health"], "ok")
            self.assertEqual(snap["resident"], 2)
            self.assertEqual(snap["active_decode"], 1)
            self.assertEqual(snap["n_ctx"], 4096)
            self.assertEqual(snap["prompt_tokens"], 12)
            controller.shutdown()


class TestMutationStaleOwner(unittest.TestCase):
    def test_owner_is_stale_when_lock_record_exists_but_not_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = MutationLock(tmp)
            lock.write(
                {
                    "pid": 99999999,
                    "start_time": "ghost",
                    "acquired_at": 0,
                    "unit_id": "wu-9",
                }
            )
            controller = Controller(workspace=tmp, runtime_count=1)
            snap = controller._mutation_status()
            self.assertFalse(snap["held"])
            self.assertEqual(snap["owner"], "wu-9")
            self.assertEqual(snap["owner_display"], "stale")
            text = format_status({"mutation": snap, "qwen": {"health": "down"}, "grok": {}})
            self.assertIn("owner=stale", text)
            self.assertNotIn("owner=wu-9", text)
            controller.shutdown()


class TestHttpJson(unittest.TestCase):
    def test_http_json_returns_none_on_error(self):
        self.assertIsNone(_http_json("http://127.0.0.1:1/health", timeout=0.2))

    def test_live_http_server_is_probed_not_assumed(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path == "/health":
                    body = b'{"status":"ok"}'
                elif self.path == "/slots":
                    body = json.dumps(
                        [
                            {
                                "id": 0,
                                "n_ctx": 8192,
                                "is_processing": True,
                                "n_prompt_tokens": 42,
                            }
                        ]
                    ).encode("utf-8")
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            with tempfile.TemporaryDirectory() as tmp:
                controller = Controller(workspace=tmp, runtime_count=1)
                controller._qwen_endpoint = lambda: f"http://{host}:{port}"
                snap = controller._qwen_pool_status()
                self.assertEqual(snap["health"], "ok")
                self.assertEqual(snap["resident"], 1)
                self.assertEqual(snap["active_decode"], 1)
                self.assertEqual(snap["n_ctx"], 8192)
                self.assertEqual(snap["prompt_tokens"], 42)
                controller.shutdown()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
