"""Conformance for the shared local HTTP child lifecycle."""
from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from hawking.backends import LlamaServerBackend, MlxServerBackend, MlxVlmServerBackend, _post_json
from hawking.resources import process_identity_matches, process_identity_status
from hawking.runtime import Runtime, RuntimePool
from hawking.serve import _stop_owned_backend_children


class _Response:
    def __init__(self, status: int = 200, body: bytes = b"{}") -> None:
        self.status = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body


class _Process:
    def __init__(self, *, pid: int = 4312, exited=None, wait_timeouts: int = 0) -> None:
        self.pid = pid
        self.exited = exited
        self.wait_timeouts = wait_timeouts
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return self.exited

    def terminate(self) -> None:
        self.terminate_calls += 1

    def wait(self, timeout: float):
        if self.wait_timeouts:
            self.wait_timeouts -= 1
            raise subprocess.TimeoutExpired("fake-child", timeout)
        if self.exited is None:
            self.exited = 0
        return self.exited

    def kill(self) -> None:
        self.kill_calls += 1
        self.exited = -9


class _LogHandle:
    def __init__(self) -> None:
        self.flushes = 0
        self.closed = False

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True


def _backends():
    return (
        LlamaServerBackend(
            "/no-model.gguf", port=8123, ctx_size=128, binary="llama-server"
        ),
        MlxServerBackend("/no-model", port=8123, binary="mlx_lm.server"),
    )


class TestHttpChildLifecycle(unittest.TestCase):
    def test_process_identity_requires_the_same_recorded_incarnation(self):
        cases = (
            ("matches", "recorded", "recorded", True),
            ("no-recorded-token", None, "recorded", False),
            ("unreadable-live-token", "recorded", None, False),
            ("reused-pid", "recorded", "new-process", False),
        )
        for label, recorded, live, expected_match in cases:
            with self.subTest(case=label), patch(
                "hawking.resources.process_start_token", return_value=live
            ):
                self.assertEqual(
                    process_identity_matches(9331, recorded), expected_match
                )
                expected_status = "matches" if expected_match else {
                    "no-recorded-token": "no_recorded_start_time",
                    "unreadable-live-token": "live_start_time_unreadable",
                    "reused-pid": "pid_reused_start_time_mismatch",
                }[label]
                self.assertEqual(process_identity_status(9331, recorded), expected_status)

    def test_concrete_backends_use_one_lifecycle_implementation(self):
        for method in ("endpoint", "log_tail", "ready", "stop"):
            with self.subTest(method=method):
                self.assertIs(
                    getattr(LlamaServerBackend, method),
                    getattr(MlxServerBackend, method),
                )
                self.assertIs(
                    getattr(MlxServerBackend, method),
                    getattr(MlxVlmServerBackend, method),
                )

    def test_endpoint_and_log_tail_handle_present_and_missing_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "child.log"
            log.write_text("prefix-tail", encoding="utf-8")
            for backend in _backends():
                with self.subTest(backend=type(backend).__name__):
                    backend.port = None
                    with self.assertRaisesRegex(RuntimeError, "no port"):
                        backend.endpoint()
                    self.assertFalse(backend.ready(0.0))

                    backend.port = 8123
                    handle = _LogHandle()
                    backend._log_path = str(log)
                    backend._log_handle = handle
                    self.assertEqual(backend.endpoint(), "http://127.0.0.1:8123")
                    self.assertEqual(backend.log_tail(4), "tail")
                    self.assertEqual(handle.flushes, 1)

                    backend._log_path = str(log.with_name("missing.log"))
                    self.assertEqual(backend.log_tail(), "")

    def test_ready_accepts_success_and_refuses_dead_child(self):
        for backend in _backends():
            with self.subTest(backend=type(backend).__name__, state="healthy"):
                backend.process = _Process()
                with patch(
                    "hawking.backends.urllib.request.urlopen", return_value=_Response()
                ) as open_url:
                    self.assertTrue(backend.ready(0.5))
                open_url.assert_called_once_with(
                    "http://127.0.0.1:8123/health", timeout=2
                )

            with self.subTest(backend=type(backend).__name__, state="early-exit"):
                backend.process = _Process(exited=1)
                with patch(
                    "hawking.backends.urllib.request.urlopen",
                    side_effect=AssertionError("dead child must not receive health traffic"),
                ):
                    self.assertFalse(backend.ready(0.5))

    def test_ready_handles_503_404_and_deadline(self):
        for status in (503, 404):
            for backend in _backends():
                with self.subTest(status=status, backend=type(backend).__name__):
                    backend.process = _Process()
                    failure = urllib.error.HTTPError(
                        "http://127.0.0.1:8123/health",
                        status,
                        "not ready",
                        None,
                        None,
                    )
                    with patch(
                        "hawking.backends.time.monotonic", side_effect=(0.0, 0.0, 1.0)
                    ), patch("hawking.backends.time.sleep") as sleep, patch(
                        "hawking.backends.urllib.request.urlopen", side_effect=failure
                    ) as open_url:
                        self.assertFalse(backend.ready(0.5))
                    open_url.assert_called_once()
                    sleep.assert_called_once_with(0.4)
                    failure.close()

    def test_post_json_proves_success_and_propagates_non_success(self):
        with patch(
            "hawking.backends.urllib.request.urlopen", return_value=_Response(body=b'{"ok":true}')
        ):
            self.assertEqual(
                _post_json("http://127.0.0.1:8123/request", {"x": 1}, 1.0, "fixture"),
                {"ok": True},
            )

        failure = urllib.error.HTTPError(
            "http://127.0.0.1:8123/request",
            418,
            "failure",
            None,
            io.BytesIO(b"teapot"),
        )
        with patch("hawking.backends.urllib.request.urlopen", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "fixture HTTP 418: teapot"):
                _post_json("http://127.0.0.1:8123/request", {}, 1.0, "fixture")
        failure.close()

    def test_spawn_preserves_an_already_running_child(self):
        for backend in _backends():
            with self.subTest(backend=type(backend).__name__):
                child = _Process()
                backend.process = child
                with patch.object(backend, "command") as command, patch(
                    "hawking.backends.subprocess.Popen"
                ) as popen:
                    backend.spawn(port=8222)
                self.assertIs(backend.process, child)
                command.assert_not_called()
                popen.assert_not_called()

    def test_stop_cleans_up_and_is_idempotent(self):
        for backend in _backends():
            with self.subTest(backend=type(backend).__name__):
                child = _Process()
                handle = _LogHandle()
                backend.process = child
                backend.pid = child.pid
                backend._log_handle = handle
                with patch("hawking.backends.pid_is_alive", return_value=False):
                    report = backend.stop()
                    repeated = backend.stop()
                self.assertEqual(report["gone"], True)
                self.assertEqual(report["unreaped"], [])
                self.assertEqual(child.terminate_calls, 1)
                self.assertEqual(backend.pid, None)
                self.assertIsNone(backend.process)
                self.assertTrue(handle.closed)
                self.assertEqual(repeated["gone"], True)

    def test_stop_kills_after_wait_timeout_and_keeps_unreaped_detached_pid(self):
        for backend in _backends():
            with self.subTest(backend=type(backend).__name__, state="timeout"):
                child = _Process(wait_timeouts=1)
                backend.process = child
                backend.pid = child.pid
                with patch("hawking.backends.pid_is_alive", return_value=False):
                    report = backend.stop()
                self.assertTrue(report["gone"])
                self.assertEqual(child.terminate_calls, 1)
                self.assertEqual(child.kill_calls, 1)

            with self.subTest(backend=type(backend).__name__, state="detached-unreaped"):
                backend.process = None
                backend.pid = 9331
                backend.start_time = "recorded-start"
                backend._stopped = True
                with patch("hawking.backends.pid_is_alive", return_value=True), patch(
                    "hawking.backends.process_identity_status", return_value="matches"
                ), patch(
                    "hawking.backends.terminate_pid",
                    return_value={"pid": 9331, "gone": False, "unreaped": []},
                ):
                    report = backend.stop()
                self.assertFalse(report["gone"])
                self.assertEqual(report["unreaped"], [9331])
                self.assertEqual(backend.pid, 9331)

    def test_stop_refuses_a_reused_or_unverified_pid(self):
        for backend in _backends():
            with self.subTest(backend=type(backend).__name__):
                backend.process = None
                backend.pid = 9331
                backend.start_time = "old-process"
                backend._stopped = True
                with patch("hawking.backends.pid_is_alive", return_value=True), patch(
                    "hawking.backends.process_identity_status",
                    return_value="pid_reused_start_time_mismatch",
                ), patch(
                    "hawking.backends.terminate_pid",
                    side_effect=AssertionError("must not kill a reused PID"),
                ):
                    report = backend.stop()
                self.assertTrue(report["gone"])
                self.assertEqual(report["unreaped"], [])
                self.assertEqual(
                    report["skipped"],
                    [
                        {
                            "pid": 9331,
                            "action": "skipped",
                            "identity": "pid_reused_start_time_mismatch",
                        }
                    ],
                )

    def test_runtime_pool_refuses_to_repeat_an_unverified_backend_kill(self):
        class StaleBackend:
            def stop(self):
                return {"pid": 9331, "gone": False, "unreaped": [9331]}

        with tempfile.TemporaryDirectory() as tmp:
            pool = RuntimePool(
                "/missing-model", workspace=tmp, repo_root=tmp, topology="slot"
            )
            pool.runtimes = [
                Runtime(
                    index=0,
                    pid=9331,
                    backend=StaleBackend(),
                    start_time="old-process",
                )
            ]
            with patch("hawking.runtime.pid_is_alive", return_value=True), patch(
                "hawking.runtime.process_identity_status",
                return_value="pid_reused_start_time_mismatch",
            ), patch(
                "hawking.runtime.terminate_pid",
                side_effect=AssertionError("pool must not kill a reused PID"),
            ):
                report = pool.stop()
        self.assertEqual(report["reaped"], [])
        self.assertEqual(report["unreaped"], [])
        self.assertEqual(
            report["skipped"],
            [{"pid": 9331, "reason": "pid_reused_start_time_mismatch"}],
        )

    def test_runtime_pool_keeps_the_original_identity_for_an_unreaped_child(self):
        class OwnedBackend:
            def stop(self):
                return {"pid": 9332, "gone": False, "unreaped": [9332]}

        with tempfile.TemporaryDirectory() as tmp:
            pool = RuntimePool(
                "/missing-model", workspace=tmp, repo_root=tmp, topology="slot"
            )
            pool.runtimes = [
                Runtime(
                    index=0,
                    pid=9332,
                    backend=OwnedBackend(),
                    start_time="recorded-start",
                )
            ]
            with patch("hawking.runtime.pid_is_alive", return_value=True), patch(
                "hawking.runtime.process_identity_status", return_value="matches"
            ), patch(
                "hawking.runtime.terminate_pid",
                return_value={"pid": 9332, "gone": False},
            ), patch(
                "hawking.runtime.process_start_token",
                side_effect=AssertionError("must not replace the recorded start token"),
            ):
                report = pool.stop()
            persisted = json.loads(pool._ownership_path().read_text(encoding="utf-8"))
        self.assertEqual(report["unreaped"], [9332])
        self.assertEqual(persisted["children"][0]["start_time"], "recorded-start")

    def test_runtime_pool_defers_an_unreadable_but_recorded_child(self):
        class UnreadableBackend:
            def stop(self):
                return {"pid": 9334, "gone": False, "unreaped": []}

        with tempfile.TemporaryDirectory() as tmp:
            pool = RuntimePool(
                "/missing-model", workspace=tmp, repo_root=tmp, topology="slot"
            )
            pool.runtimes = [
                Runtime(
                    index=0,
                    pid=9334,
                    backend=UnreadableBackend(),
                    start_time="recorded-start",
                )
            ]
            with patch("hawking.runtime.pid_is_alive", return_value=True), patch(
                "hawking.runtime.process_identity_status",
                return_value="live_start_time_unreadable",
            ), patch(
                "hawking.runtime.terminate_pid",
                side_effect=AssertionError("must not kill without a live identity token"),
            ):
                report = pool.stop()
            persisted = json.loads(pool._ownership_path().read_text(encoding="utf-8"))
        self.assertEqual(report["deferred"], [9334])
        self.assertEqual(persisted["children"][0]["start_time"], "recorded-start")
        self.assertEqual(
            persisted["children"][0]["cleanup_status"], "identity_unreadable"
        )

    def test_runtime_pool_does_not_persist_a_race_dead_secondary_child(self):
        class BackendWithDeadSecondaryChild:
            def stop(self):
                return {"pid": 9335, "gone": False, "unreaped": [9336]}

        with tempfile.TemporaryDirectory() as tmp:
            pool = RuntimePool(
                "/missing-model", workspace=tmp, repo_root=tmp, topology="slot"
            )
            pool.runtimes = [
                Runtime(
                    index=0,
                    pid=9335,
                    backend=BackendWithDeadSecondaryChild(),
                    start_time="recorded-start",
                )
            ]
            with patch("hawking.runtime.pid_is_alive", side_effect=lambda pid: pid == 9335), patch(
                "hawking.runtime.process_identity_status", return_value="matches"
            ), patch(
                "hawking.runtime.terminate_pid", return_value={"pid": 9335, "gone": False}
            ):
                report = pool.stop()
        self.assertEqual(report["reaped"], [9336])
        self.assertEqual(report["unreaped"], [9335])
        self.assertEqual(report["deferred"], [])

    def test_serving_fallback_reports_a_reused_pid_without_killing_it(self):
        class Child:
            pid = 9333
            process = None
            start_time = "old-process"

        class Root:
            def __init__(self) -> None:
                self.backend = Child()
                self.stop_calls = 0

            def stop(self) -> None:
                self.stop_calls += 1

        root = Root()
        with patch("hawking.resources.pid_is_alive", return_value=True), patch(
            "hawking.resources.process_identity_status",
            return_value="pid_reused_start_time_mismatch",
        ), patch(
            "hawking.backends.terminate_pid",
            side_effect=AssertionError("serve must not kill a reused PID"),
        ):
            report = _stop_owned_backend_children(root)
        self.assertEqual(root.stop_calls, 1)
        self.assertEqual(
            report,
            {
                "skipped": [
                    {
                        "pid": 9333,
                        "action": "skipped",
                        "identity": "pid_reused_start_time_mismatch",
                    }
                ]
            },
        )
