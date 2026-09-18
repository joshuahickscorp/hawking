"""The wire contract Open WebUI actually depends on.

Driven against the real handler over a real socket with a fake backend, because
the failure this guards is a SHAPE failure -- a missing `created`, a stream flag
answered with a JSON body instead of SSE -- and none of those are visible to a
test that calls the functions directly.
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from hawking.catalog import Body
from hawking.serve import (
    GREEDY_OK,
    Resident,
    chat_payload,
    make_handler,
    models_payload,
    profile_is_greedy,
    sampler_refusal,
    stream_frames,
    visible_text,
    worker_tool_allowlist,
    worker_request_mode,
)


class _Result:
    text = "batched matmul reuses each loaded weight across many rows."
    finish_reason = "stop"
    prompt_tokens = 11
    completion_tokens = 9
    total_tokens = 20
    degraded: list = []
    raw: dict = {}


class _Backend:
    def __init__(self):
        self.seen = []

    def complete(self, payload, timeout=None):
        self.seen.append(payload)
        return _Result()


class _ToolBackend(_Backend):
    identity = "kimi-test"
    native_tools = False
    tools_verdict = {"qualified": True}

    def complete(self, payload, timeout=None):
        self.seen.append(payload)
        if len(self.seen) == 1:
            return type("R", (), {
                "text": '{"tool":"fs.list","arguments":{"path":"."}}',
                "finish_reason": "stop", "degraded": [], "raw": {},
            })()
        return type("R", (), {
            "text": "The dispatcher returned the directory observation.",
            "finish_reason": "stop", "degraded": [], "raw": {},
        })()


class _WebUIManager:
    def __init__(self):
        self.started = []
        self.stopped = []

    def snapshot(self):
        return [{"pid": 123, "port": 8081, "state": "running"}]

    def start(self, port, endpoint, requester_pid):
        self.started.append((port, endpoint, requester_pid))
        return {"pid": 456, "port": port, "endpoint": endpoint, "state": "running"}

    def stop(self, port, *, requester_pid=None):
        self.stopped.append((port, requester_pid))
        return True


def _serve(greedy=True, webui_manager=None):
    backend = _Backend()
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(backend, "sealed-3.14", greedy=greedy,
                     health={"status": "ok", "resident": "sealed-3.14"},
                     webui_manager=webui_manager,
                     endpoint_base="http://127.0.0.1:8014/v1"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, backend, f"http://127.0.0.1:{httpd.server_address[1]}"


def _post(base, body, path="/v1/chat/completions"):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


class TestOpenAISurface(unittest.TestCase):
    def setUp(self):
        self.httpd, self.backend, self.base = _serve()
        self.addCleanup(self.httpd.shutdown)

    def test_models_carries_the_fields_the_picker_reads(self):
        with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as r:
            data = json.loads(r.read())
        self.assertEqual(data["object"], "list")
        row = data["data"][0]
        # H-Web deliberately exposes the admitted device identity, not an
        # arbitrary test backend label.  The router still owns runtime choice.
        self.assertEqual(row["id"], "KIMI_P0_OPERATIONAL")
        # A missing `created` renders as "Invalid Date" in the Open WebUI picker.
        self.assertIsInstance(row["created"], int)
        self.assertEqual(row["owned_by"], "hawking")

    def test_h_web_goal_ui_uses_the_canonical_hawking_control_surface(self):
        with urllib.request.urlopen(self.base + "/api/goal/ui", timeout=10) as response:
            body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
        self.assertIn("text/html", content_type)
        self.assertIn("Hawking discrete Goal", body)
        self.assertIn("/api/goal/control", body)

    def test_non_stream_answer_is_openai_shaped(self):
        code, ctype, raw = _post(self.base, {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200, raw)
        body = json.loads(raw)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["role"], "assistant")
        self.assertIn("batched matmul", body["choices"][0]["message"]["content"])
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"]["total_tokens"], 20)

    def test_provider_timings_survive_the_hawking_response(self):
        result = _Result()
        result.raw = {"timings": {"predicted_per_second": 88.5}}
        body = chat_payload(result, "sealed-3.14", request_id="timing-test")
        self.assertEqual(body["timings"]["predicted_per_second"], 88.5)
        self.assertEqual(body["hawking"]["timings_scope"], "final_model_call_only")

    def test_stream_true_returns_SSE_not_json(self):
        # The defect this pins: answering a streaming request with a JSON body
        # leaves the browser waiting forever on a stream that never frames.
        code, ctype, raw = _post(
            self.base, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertEqual(code, 200, raw)
        self.assertIn("text/event-stream", ctype)
        text = raw.decode()
        self.assertTrue(text.startswith("data: "), text[:80])
        self.assertIn("[DONE]", text)
        deltas = [json.loads(line[6:])
                  for line in text.splitlines()
                  if line.startswith("data: ") and "[DONE]" not in line]
        self.assertEqual(deltas[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertIn("batched matmul", deltas[1]["choices"][0]["delta"]["content"])
        self.assertEqual(deltas[-1]["choices"][0]["finish_reason"], "stop")
        for d in deltas:
            self.assertEqual(d["object"], "chat.completion.chunk")

    def test_the_stream_flag_never_reaches_the_resident(self):
        _post(self.base, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertNotIn("stream", self.backend.seen[-1])

    def test_streaming_browser_turn_persists_the_durable_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _Backend()
            httpd = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(
                    backend,
                    "sealed-3.14",
                    greedy=False,
                    health={"status": "ok", "resident": "sealed-3.14"},
                    stores={"state_root": tmp},
                ),
            )
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.shutdown)
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            code, _, raw = _post(base, {
                "session_id": "stream-persist",
                "messages": [{"role": "user", "content": "remember this objective"}],
                "stream": True,
            })
            self.assertEqual(code, 200, raw)
            saved = json.loads((
                pathlib.Path(tmp) / ".hawking/chat/stream-persist.json"
            ).read_text(encoding="utf-8"))
            self.assertEqual(saved["turns"], 1)
            self.assertEqual(saved["objective"], "remember this objective")

    def test_a_sampler_request_is_refused_with_the_accepted_value(self):
        code, _, raw = _post(self.base, {
            "messages": [{"role": "user", "content": "hi"}], "temperature": 0.8})
        self.assertEqual(code, 400)
        message = json.loads(raw)["error"]["message"]
        self.assertIn("temperature=0.0", message)
        self.assertIn("Advanced Params", message, "the refusal does not say where to fix it")

    def test_greedy_defaults_are_accepted(self):
        code, _, raw = _post(self.base, {
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0, "top_p": 1, "n": 1})
        self.assertEqual(code, 200, raw)

    def test_no_messages_says_the_shape_it_wants(self):
        code, _, raw = _post(self.base, {})
        self.assertEqual(code, 400)
        self.assertIn("'role'", json.loads(raw)["error"]["message"])

    def test_an_unknown_route_names_the_ones_that_exist(self):
        code, _, raw = _post(self.base, {}, path="/v1/completions")
        self.assertEqual(code, 404)
        self.assertIn("/v1/chat/completions", json.loads(raw)["error"]["message"])

    def test_daemon_webui_control_is_local_and_uses_the_shared_endpoint(self):
        manager = _WebUIManager()
        httpd, _, base = _serve(webui_manager=manager)
        self.addCleanup(httpd.shutdown)
        code, _, raw = _post(base, {
            "port": 8082, "requester_pid": 789,
        }, path="/hawkingd/webui")
        self.assertEqual(code, 200, raw)
        self.assertEqual(manager.started, [(8082, "http://127.0.0.1:8014/v1", 789)])
        self.assertEqual(json.loads(raw)["endpoint"], "http://127.0.0.1:8014/v1")

        with urllib.request.urlopen(base + "/hawkingd/webui", timeout=10) as response:
            listing = json.loads(response.read())
        self.assertEqual(listing["owner"], "hawkingd")
        self.assertEqual(listing["client_surfaces"][0]["port"], 8081)

        code, _, raw = _post(base, {
            "port": 8082, "requester_pid": 789,
        }, path="/hawkingd/webui/stop")
        self.assertEqual(code, 200, raw)
        self.assertEqual(manager.stopped, [(8082, 789)])


class TestNonGreedyBackendDoesNotRefuse(unittest.TestCase):
    def test_an_mlx_specimen_may_sample(self):
        httpd, _, base = _serve(greedy=False)
        self.addCleanup(httpd.shutdown)
        code, _, raw = _post(base, {
            "messages": [{"role": "user", "content": "hi"}], "temperature": 0.8})
        self.assertEqual(code, 200, raw)


class TestToolLivenessEvidence(unittest.TestCase):
    def test_read_only_research_allowlist_excludes_local_or_runtime_side_effects(self):
        allowed = worker_tool_allowlist(worker_request_mode("read_only_research"))
        self.assertIsNotNone(allowed)
        self.assertTrue({"fs.read", "receipt.read", "campaign.state"} <= allowed)
        self.assertFalse({
            "shell.readonly", "shell.exec", "forensics.snapshot", "repo.edit",
            "tests.run", "gravity.experiment", "benchmark.run", "physical.emit",
            "odyssey.read", "git.log", "processes.list", "context.recall",
            "architecture.inspect", "tools.catalog", "odyssey.ledger",
            "odyssey.status", "observation.expand",
        } & allowed)

    def test_read_only_research_refuses_runtime_tool_from_a_write_built_registry(self):
        from hawking.tool_registry import (
            READ_ONLY, REVERSIBLE_RUNTIME, ToolContext, ToolRegistry, ToolSpec,
        )

        class _RuntimeSeekingBackend(_Backend):
            identity = "kimi-test"
            native_tools = False
            tools_verdict = {"qualified": True}

            def complete(self, payload, timeout=None):
                self.seen.append(payload)
                if len(self.seen) == 1:
                    return type("R", (), {
                        "text": '{"tool":"shell.exec","arguments":{}}',
                        "finish_reason": "stop", "degraded": [], "raw": {},
                    })()
                return type("R", (), {
                    "text": "No runtime tool was available to this request.",
                    "finish_reason": "stop", "degraded": [], "raw": {},
                })()

        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(ToolContext(
                workspace=tmp, repo_root=tmp,
                permissions=frozenset({READ_ONLY, REVERSIBLE_RUNTIME}),
            ))
            invoked = []
            registry.register(ToolSpec(
                "shell.exec", "must not be available to read-only research",
                {"type": "object", "additionalProperties": False,
                 "properties": {}},
                mutation=REVERSIBLE_RUNTIME,
                handler=lambda _context, _args: invoked.append(True) or {},
            ))
            backend = _RuntimeSeekingBackend()
            httpd = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(backend, "kimi-test", greedy=False,
                             health={"status": "ok", "resident": "kimi-test"},
                             registry=registry, stores={"engine": object()}))
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.shutdown)
            code, _, raw = _post(
                f"http://127.0.0.1:{httpd.server_address[1]}",
                {"messages": [{"role": "user", "content": "inspect only"}],
                 "hawking_worker_mode": "read_only_research"})

        self.assertEqual(code, 200, raw)
        body = json.loads(raw)
        self.assertEqual(body["hawking"]["tools_used"][0]["error"],
                         "not offered to chat")
        self.assertEqual(invoked, [])

    def test_read_only_research_dispatch_suppresses_ambient_credential_context(self):
        from hawking.tool_registry import (
            READ_ONLY, RESEARCH, ToolContext, ToolRegistry, ToolSpec,
            network_credentials_allowed,
        )

        class _CredentialSeekingBackend(_Backend):
            identity = "kimi-test"
            native_tools = False
            tools_verdict = {"qualified": True}

            def complete(self, payload, timeout=None):
                self.seen.append(payload)
                text = (
                    '{"tool":"github.search","arguments":{"query":"public"}}'
                    if len(self.seen) == 1 else "Public lookup completed."
                )
                return type("R", (), {
                    "text": text, "finish_reason": "stop", "degraded": [], "raw": {},
                })()

        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(ToolContext(
                workspace=tmp, repo_root=tmp,
                permissions=frozenset({READ_ONLY, RESEARCH}),
            ))
            credential_context = []
            registry.register(ToolSpec(
                "github.search", "public lookup only",
                {"type": "object", "required": ["query"], "additionalProperties": False,
                 "properties": {"query": {"type": "string"}}},
                mutation=RESEARCH,
                handler=lambda _context, _args: credential_context.append(
                    network_credentials_allowed()) or {"results": []},
            ))
            backend = _CredentialSeekingBackend()
            httpd = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(backend, "kimi-test", greedy=False,
                             health={"status": "ok", "resident": "kimi-test"},
                             registry=registry),
            )
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.shutdown)
            with mock.patch.dict(os.environ, {"GH_TOKEN": "not-for-worker"}):
                code, _, raw = _post(
                    f"http://127.0.0.1:{httpd.server_address[1]}",
                    {"messages": [{"role": "user", "content": "inspect only"}],
                     "hawking_worker_mode": "read_only_research"},
                )

        self.assertEqual(code, 200, raw)
        self.assertEqual(credential_context, [False])

    def test_restricted_worker_cannot_select_another_resident(self):
        inner = _Backend()
        resident = Resident(
            Body(name="first", path="/tmp/first", kind="mlx", bytes=0,
                 revision="", source="test"),
            inner,
        )
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(resident, "first", greedy=False,
                         health={"status": "ok", "resident": "first"}),
        )
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        for mode in ("reason_only", "read_only_research"):
            code, _, raw = _post(base, {
                "model": "other-resident",
                "messages": [{"role": "user", "content": "inspect only"}],
                "hawking_worker_mode": mode,
            })
            self.assertEqual(code, 409, raw)
            self.assertEqual(
                json.loads(raw)["error"]["type"], "model_switch_denied")

        self.assertEqual(resident.identity, "first")
        self.assertEqual(inner.seen, [])

    def test_restricted_mode_fences_a_direct_native_adapter_too(self):
        from hawking.hawking_native import native_runtime_spawning_allowed

        class _DirectNativeLikeBackend(_Backend):
            identity = "native-test"

            def __init__(self):
                super().__init__()
                self.spawn_allowed = []

            def complete(self, payload, timeout=None):
                self.spawn_allowed.append(native_runtime_spawning_allowed())
                return super().complete(payload, timeout)

        backend = _DirectNativeLikeBackend()
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(backend, "native-test", greedy=False,
                         health={"status": "ok", "resident": "native-test"}),
        )
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        code, _, raw = _post(
            f"http://127.0.0.1:{httpd.server_address[1]}",
            {"messages": [{"role": "user", "content": "reason only"}],
             "hawking_worker_mode": "reason_only"},
        )

        self.assertEqual(code, 200, raw)
        self.assertEqual(backend.spawn_allowed, [False])

    def test_restricted_worker_never_recovers_a_dead_owned_provider(self):
        class _ExitedProcess:
            def poll(self):
                return 17

        class _DeadOwnedBackend:
            def __init__(self):
                self.process = _ExitedProcess()
                self.spawned = False
                self.stopped = False
                self.seen = []

            def spawn(self):
                self.spawned = True

            def ready(self, _timeout):
                return True

            def stop(self):
                self.stopped = True
                return {"gone": True}

            def complete(self, payload, timeout=None):
                self.seen.append(payload)
                raise AssertionError("a dead restricted resident must not be called")

        dead = _DeadOwnedBackend()
        resident = Resident(
            Body(name="first", path="/tmp/first", kind="mlx", bytes=0,
                 revision="", source="test"),
            dead, provider_owned=True,
        )
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(resident, "first", greedy=False,
                         health={"status": "ok", "resident": "first"}),
        )
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        import hawking.runtime_iface as runtime_iface
        with mock.patch.object(
                runtime_iface, "make_backend_for_model",
                side_effect=AssertionError("restricted request started recovery"),
        ) as factory:
            for mode in ("reason_only", "read_only_research"):
                code, _, raw = _post(base, {
                    "messages": [{"role": "user", "content": "inspect only"}],
                    "hawking_worker_mode": mode,
                })
                self.assertEqual(code, 503, raw)
                self.assertEqual(
                    json.loads(raw)["error"]["type"], "resident_unavailable")

        factory.assert_not_called()
        self.assertFalse(dead.spawned)
        self.assertFalse(dead.stopped)
        self.assertEqual(dead.seen, [])

    def test_restricted_worker_never_recovers_a_live_but_refusing_listener(self):
        class _LiveProcess:
            def poll(self):
                return None

        class _RefusingOwnedBackend:
            def __init__(self):
                self.process = _LiveProcess()
                self.spawned = False
                self.stopped = False
                self.seen = []

            def spawn(self):
                self.spawned = True

            def ready(self, _timeout):
                return True

            def stop(self):
                self.stopped = True
                return {"gone": True}

            def complete(self, payload, timeout=None):
                self.seen.append(payload)
                raise urllib.error.URLError("listener refused")

        listener = _RefusingOwnedBackend()
        resident = Resident(
            Body(name="first", path="/tmp/first", kind="mlx", bytes=0,
                 revision="", source="test"),
            listener, provider_owned=True,
        )
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(resident, "first", greedy=False,
                         health={"status": "ok", "resident": "first"}),
        )
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)

        import hawking.runtime_iface as runtime_iface
        with mock.patch.object(
                runtime_iface, "make_backend_for_model",
                side_effect=AssertionError("restricted request started recovery"),
        ) as factory:
            code, _, raw = _post(
                f"http://127.0.0.1:{httpd.server_address[1]}",
                {"messages": [{"role": "user", "content": "inspect only"}],
                 "hawking_worker_mode": "read_only_research"},
            )

        self.assertEqual(code, 502, raw)
        factory.assert_not_called()
        self.assertFalse(listener.spawned)
        self.assertFalse(listener.stopped)
        self.assertEqual(len(listener.seen), 1)

    def test_restricted_worker_does_not_create_session_or_trace_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _Backend()
            httpd = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(
                    backend, "kimi-test", greedy=False,
                    health={"status": "ok", "resident": "kimi-test"},
                    stores={"state_root": tmp},
                ),
            )
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.shutdown)
            code, _, raw = _post(
                f"http://127.0.0.1:{httpd.server_address[1]}",
                {"session_id": "must-not-persist",
                 "messages": [{"role": "user", "content": "inspect only"}],
                 "hawking_worker_mode": "read_only_research"},
            )
            self.assertEqual(code, 200, raw)
            self.assertEqual(list(pathlib.Path(tmp).iterdir()), [])

    def test_restricted_worker_skips_automatic_repo_context_retrieval(self):
        from hawking.repo_context import RepoContext

        with tempfile.TemporaryDirectory() as tmp:
            repo = RepoContext(root=pathlib.Path(tmp), name="fixture", is_git=False)
            backend = _Backend()
            httpd = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(
                    backend, "kimi-test", greedy=False,
                    health={"status": "ok", "resident": "kimi-test"},
                    repo=repo,
                ),
            )
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.shutdown)
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            with mock.patch.object(repo, "volatile_block") as volatile, \
                    mock.patch("hawking.repo_context.native_rank_code_paths") as rank_code, \
                    mock.patch("hawking.repo_context.native_rank_content_paths") as rank_content, \
                    mock.patch("hawking.repo_context.subprocess.run") as process_run:
                for mode in ("reason_only", "read_only_research"):
                    code, _, raw = _post(base, {
                        "messages": [{"role": "user", "content": "inspect hawking/serve.py"}],
                        "hawking_worker_mode": mode,
                    })
                    self.assertEqual(code, 200, raw)

        volatile.assert_not_called()
        rank_code.assert_not_called()
        rank_content.assert_not_called()
        process_run.assert_not_called()
        self.assertEqual(
            [payload["messages"] for payload in backend.seen],
            [[{"role": "user", "content": "inspect hawking/serve.py"}]] * 2,
        )

    def test_restricted_worker_does_not_persist_large_tool_observations(self):
        from hawking.chat_tools import RESULT_CHARS
        from hawking.paste_cache import PasteCache
        from hawking.tool_registry import READ_ONLY, ToolContext, ToolRegistry, ToolSpec

        class _LargeObservationBackend(_Backend):
            identity = "kimi-test"
            native_tools = False
            tools_verdict = {"qualified": True}

            def complete(self, payload, timeout=None):
                self.seen.append(payload)
                if len(self.seen) == 1:
                    return type("R", (), {
                        "text": '{"tool":"fs.list","arguments":{}}',
                        "finish_reason": "stop", "degraded": [], "raw": {},
                    })()
                return type("R", (), {
                    "text": "The large observation stayed ephemeral.",
                    "finish_reason": "stop", "degraded": [], "raw": {},
                })()

        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(ToolContext(
                workspace=tmp, repo_root=tmp, permissions=frozenset({READ_ONLY}),
            ))
            registry.register(ToolSpec(
                "fs.list", "return one deliberately large read-only observation",
                {"type": "object", "additionalProperties": False, "properties": {}},
                handler=lambda _context, _args: {"body": "x" * (RESULT_CHARS + 128)},
            ))
            cache = PasteCache(tmp)
            backend = _LargeObservationBackend()
            httpd = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(backend, "kimi-test", greedy=False,
                             health={"status": "ok", "resident": "kimi-test"},
                             registry=registry, stores={"cache": cache}),
            )
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.shutdown)
            with mock.patch.object(cache, "store", wraps=cache.store) as store:
                code, _, raw = _post(
                    f"http://127.0.0.1:{httpd.server_address[1]}",
                    {"messages": [{"role": "user", "content": "inspect only"}],
                     "hawking_worker_mode": "read_only_research"},
                )

            self.assertEqual(code, 200, raw)
            store.assert_not_called()
            self.assertFalse((pathlib.Path(tmp) / ".hawking").exists())

    def test_read_only_dispatch_suppresses_optional_native_helper_processes(self):
        from hawking.tool_registry import READ_ONLY, RESEARCH, default_tool_registry

        class _ListBackend(_Backend):
            identity = "kimi-test"
            native_tools = False
            tools_verdict = {"qualified": True}

            def complete(self, payload, timeout=None):
                self.seen.append(payload)
                if len(self.seen) == 1:
                    return type("R", (), {
                        "text": '{"tool":"fs.list","arguments":{"path":"."}}',
                        "finish_reason": "stop", "degraded": [], "raw": {},
                    })()
                return type("R", (), {
                    "text": "The list was read through the pure Python path.",
                    "finish_reason": "stop", "degraded": [], "raw": {},
                })()

        with tempfile.TemporaryDirectory() as tmp:
            registry = default_tool_registry(
                tmp, repo_root=tmp, permissions={READ_ONLY, RESEARCH})
            backend = _ListBackend()
            httpd = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(backend, "kimi-test", greedy=False,
                             health={"status": "ok", "resident": "kimi-test"},
                             registry=registry),
            )
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.shutdown)
            with mock.patch("hawking.repo_context.native_tool_dispatch_admit") as admit, \
                    mock.patch("hawking.repo_context.native_filesystem_list") as list_helper:
                code, _, raw = _post(
                    f"http://127.0.0.1:{httpd.server_address[1]}",
                    {"messages": [{"role": "user", "content": "list files"}],
                     "hawking_worker_mode": "read_only_research",
                     "tools": [{"type": "function", "function": {"name": "shell.exec"}}]},
                )

        self.assertEqual(code, 200, raw)
        admit.assert_not_called()
        list_helper.assert_not_called()
        self.assertTrue(all("tools" not in payload for payload in backend.seen))

    def test_native_research_replaces_caller_tool_schemas_with_server_allowlist(self):
        from hawking.tool_registry import READ_ONLY, ToolContext, ToolRegistry, ToolSpec

        class _NativeBackend(_Backend):
            identity = "native-test"
            native_tools = True
            tools_verdict = {"qualified": True}

        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(ToolContext(
                workspace=tmp, repo_root=tmp, permissions=frozenset({READ_ONLY}),
            ))
            registry.register(ToolSpec(
                "fs.list", "list the bounded workspace",
                {"type": "object", "additionalProperties": False,
                 "properties": {"path": {"type": "string"}}},
                handler=lambda _context, _args: {"files": []},
            ))
            backend = _NativeBackend()
            httpd = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(backend, "native-test", greedy=False,
                             health={"status": "ok", "resident": "native-test"},
                             registry=registry),
            )
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.shutdown)
            code, _, raw = _post(
                f"http://127.0.0.1:{httpd.server_address[1]}",
                {"messages": [{"role": "user", "content": "inspect only"}],
                 "hawking_worker_mode": "read_only_research",
                 "tools": [{"type": "function", "function": {"name": "shell.exec"}}],
                 "functions": [{"name": "shell.exec", "parameters": {}}],
                 "function_call": {"name": "shell.exec"},
                 "chat_template_kwargs": {"tools": [{"name": "shell.exec"}] }},
            )

        self.assertEqual(code, 200, raw)
        forwarded = backend.seen[0].get("tools") or []
        names = {row.get("function", {}).get("name") for row in forwarded}
        from hawking.chat_tools import prompt_menu_names
        from hawking.serve import WorkerRequestMode, worker_tool_allowlist
        self.assertEqual(
            names,
            set(prompt_menu_names(
                registry, write=False,
                allowed_names=worker_tool_allowlist(
                    WorkerRequestMode.READ_ONLY_RESEARCH))),
        )
        self.assertIn("fs.list", names)
        self.assertNotIn("shell.exec", names)
        self.assertNotIn("functions", backend.seen[0])
        self.assertNotIn("function_call", backend.seen[0])
        self.assertNotIn("chat_template_kwargs", backend.seen[0])
        self.assertTrue(json.loads(raw)["hawking"]["tool_contract"]
                        ["caller_tool_schemas_withheld"])

    def test_builder_mutation_turn_projects_gated_doors_to_native_provider(self):
        from pathlib import Path
        from hawking.tool_registry import READ_ONLY, ToolContext, ToolRegistry

        class _NativeBuilderBackend(_Backend):
            identity = "native-builder-test"
            native_tools = True
            tools_verdict = {"qualified": True}

        with tempfile.TemporaryDirectory() as tmp:
            registry = ToolRegistry(ToolContext(
                workspace=tmp, repo_root=tmp,
                permissions=frozenset({READ_ONLY}),
            ))
            backend = _NativeBuilderBackend()
            httpd = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(
                    backend, "native-builder-test", greedy=False,
                    health={
                        "status": "ok", "resident": "native-builder-test",
                        "authority": "remote_goal_write",
                    },
                    registry=registry, stores={"engine": object()},
                ),
            )
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.shutdown)
            code, _, raw = _post(
                f"http://127.0.0.1:{httpd.server_address[1]}",
                {
                    "messages": [{
                        "role": "user",
                        "content": "repair the disposable fixture",
                    }],
                    "hawking_web": True,
                    "hawking_web_build_authorized": True,
                    "hawking_worker_mode": "builder_interactive",
                },
            )

        self.assertEqual(code, 200, raw)
        body = json.loads(raw)
        names = {
            row.get("function", {}).get("name")
            for row in (backend.seen[0].get("tools") or [])
        }
        self.assertIn("repo.edit", names)
        self.assertIn("tests.run", names)
        self.assertLess(len(names), 50)
        self.assertIn("git.status", names)
        self.assertIn("git.diff", names)
        contract = body["hawking"]["tool_contract"]
        self.assertEqual(contract["mutation_authority"], "granted")
        self.assertEqual(contract["tool_budget_ceiling"], 24)
        self.assertEqual(contract["tool_budget_scope"], "interactive_build")
        self.assertEqual(contract["tool_calls_used"], 0)

    def test_discrete_goal_mutation_turn_projects_repo_edit_schema(self):
        """A durable Build WorkUnit must receive the same typed mutation door.

        ``repo.edit`` is deliberately outside the generic registry.  This
        covers the real child-WorkUnit path, which previously carried write
        authority but omitted the extra provider schema and therefore made the
        canonical builder door impossible to call.
        """
        from pathlib import Path
        from hawking.tool_registry import READ_ONLY, ToolContext, ToolRegistry

        class _NativeGoalBackend(_Backend):
            identity = "native-goal-test"
            # Exercise the production regression: a generic resident may
            # advertise the old textual adapter while durable Goals must be
            # promoted onto Hawking's typed provider schema path.
            native_tools = False
            tools_verdict = {"qualified": True}

        with tempfile.TemporaryDirectory() as tmp:
            goals = Path(tmp) / ".hawking" / "goals"
            goals.mkdir(parents=True)
            (goals / "GOAL-TEST.contract.json").write_text(json.dumps({
                "goal_mode": "discrete_goal",
                "mutation_bundle": {
                    "schema": "hawking.mutation_bundle.v1",
                    "allowed_files": ["fixture.py", "test_fixture.py"],
                    "focus_files": ["fixture.py", "test_fixture.py"],
                    "focus_tests": ["test_fixture.py"],
                },
            }), encoding="utf-8")
            registry = ToolRegistry(ToolContext(
                workspace=tmp, repo_root=tmp,
                permissions=frozenset({READ_ONLY}),
            ))
            backend = _NativeGoalBackend()
            httpd = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(
                    backend, "native-goal-test", greedy=False,
                    health={
                        "status": "ok", "resident": "native-goal-test",
                        "authority": "remote_goal_write",
                    },
                    registry=registry,
                    stores={"engine": object(), "state_root": tmp},
                ),
            )
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.shutdown)
            code, _, raw = _post(
                f"http://127.0.0.1:{httpd.server_address[1]}",
                {
                    "messages": [{"role": "user", "content": "repair fixture"}],
                    "hawking_goal_id": "GOAL-TEST",
                    "hawking_goal_mode": "discrete_goal",
                },
            )

        self.assertEqual(code, 200, raw)
        names = {
            row.get("function", {}).get("name")
            for row in (backend.seen[0].get("tools") or [])
        }
        self.assertIn("repo.edit", names)
        self.assertIn("tests.run", names)
        self.assertLess(len(names), 50)
        self.assertEqual(
            json.loads(raw)["hawking"]["tool_contract"]["serialized_format"],
            "openai_tools_argument",
        )

    def test_tool_choice_none_bypasses_the_loop_without_disabling_tools(self):
        backend = _Backend()
        backend.identity = "kimi-test"
        backend.native_tools = False
        backend.tools_verdict = {"qualified": True}
        registry = type("Registry", (), {
            "get": lambda self, name: None,
            "invoke": lambda self, name, args: (_ for _ in ()).throw(
                AssertionError("tool_choice=none must not dispatch a tool")),
        })()
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(backend, "kimi-test", greedy=False,
                         health={"status": "ok", "resident": "kimi-test"},
                         registry=registry))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        code, _, raw = _post(base, {
            "messages": [{"role": "user", "content": "reason from this prompt"}],
            "tool_choice": "none",
        })

        self.assertEqual(code, 200, raw)
        body = json.loads(raw)
        self.assertEqual(body["hawking"]["tool_contract"]["serialized_format"],
                         "disabled_by_request")
        self.assertFalse(body["hawking"]["tool_contract"]["tool_loop_entered"])
        self.assertEqual(body["hawking"]["tools_used"], [])
        self.assertNotIn("tool_choice", backend.seen[0])

    def test_response_distinguishes_offered_tools_from_dispatched_tools(self):
        backend = _ToolBackend()
        registry = type("Registry", (), {
            "get": lambda self, name: None,
            "invoke": lambda self, name, args: type(
                "Result", (), {"ok": True, "value": {"files": ["proof.txt"]},
                                "error": None, "provenance": {}})(),
        })()
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(backend, "kimi-test", greedy=False,
                         health={"status": "ok", "resident": "kimi-test"},
                         registry=registry))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        code, _, raw = _post(
            f"http://127.0.0.1:{httpd.server_address[1]}",
            {"messages": [{"role": "user", "content": "inspect it"}]})
        self.assertEqual(code, 200, raw)
        body = json.loads(raw)
        self.assertEqual(body["choices"][0]["message"]["content"],
                         "The dispatcher returned the directory observation.")
        self.assertEqual(body["hawking"]["tool_contract"]["serialized_format"],
                         "textual_json_contract")
        self.assertGreater(body["hawking"]["tool_contract"]["serialized_count"], 0)
        self.assertEqual(body["hawking"]["tool_contract"]["actual_invocations"], 1)
        self.assertTrue(body["hawking"]["tools_used"][0]["dispatched"])
        self.assertEqual(body["hawking"]["runtime"]["source"], "hawkingd:/health")
        self.assertEqual(body["hawking"]["runtime"]["resident"], "kimi-test")
        self.assertTrue(body["hawking"]["runtime"]["identity_match"])

    def test_read_only_research_mode_withholds_builder_authority(self):
        class _EditSeekingBackend(_Backend):
            identity = "kimi-test"
            native_tools = False
            tools_verdict = {"qualified": True}

            def complete(self, payload, timeout=None):
                self.seen.append(payload)
                if len(self.seen) == 1:
                    return type("R", (), {
                        "text": '{"tool":"repo.edit","arguments":{"operations":[]}}',
                        "finish_reason": "stop", "degraded": [], "raw": {},
                    })()
                return type("R", (), {
                    "text": "The requested mutation is not available in this research task.",
                    "finish_reason": "stop", "degraded": [], "raw": {},
                })()

        class _Engine:
            def apply_typed_mutation(self, *_args, **_kwargs):
                raise AssertionError("read-only research must not reach the engine")

        backend = _EditSeekingBackend()
        registry = type("Registry", (), {
            "get": lambda self, name: None,
            "invoke": lambda self, name, args: type(
                "Result", (), {"ok": True, "value": {}, "error": None,
                              "provenance": {}})(),
        })()
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(backend, "kimi-test", greedy=False,
                         health={"status": "ok", "resident": "kimi-test"},
                         registry=registry, stores={"engine": _Engine()}))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        code, _, raw = _post(
            f"http://127.0.0.1:{httpd.server_address[1]}",
            {"messages": [{"role": "user", "content": "inspect only"}],
             "hawking_worker_mode": "read_only_research"})
        self.assertEqual(code, 200, raw)
        body = json.loads(raw)
        contract = body["hawking"]["tool_contract"]
        self.assertEqual(contract["worker_mode"], "read_only_research")
        self.assertEqual(contract["mutation_authority"], "denied")
        self.assertFalse(body["hawking"]["tools_used"][0]["ok"])
        self.assertEqual(body["hawking"]["tools_used"][0]["error"],
                         "not offered to chat")
        self.assertNotIn("hawking_worker_mode", backend.seen[0])

    def test_reason_only_mode_disables_tool_loop_and_mutation_authority(self):
        backend = _Backend()
        backend.identity = "kimi-test"
        backend.native_tools = False
        backend.tools_verdict = {"qualified": True}
        registry = type("Registry", (), {
            "get": lambda self, name: None,
            "invoke": lambda self, name, args: (_ for _ in ()).throw(
                AssertionError("reason_only must not dispatch a tool")),
        })()
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(backend, "kimi-test", greedy=False,
                         health={"status": "ok", "resident": "kimi-test"},
                         registry=registry, stores={"engine": object()}))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        code, _, raw = _post(
            f"http://127.0.0.1:{httpd.server_address[1]}",
            {"messages": [{"role": "user", "content": "reason only"}],
             "hawking_worker_mode": "reason_only",
             "tools": [{"type": "function", "function": {"name": "shell.exec"}}]})
        self.assertEqual(code, 200, raw)
        body = json.loads(raw)
        contract = body["hawking"]["tool_contract"]
        self.assertEqual(contract["worker_mode"], "reason_only")
        self.assertEqual(contract["mutation_authority"], "denied")
        self.assertEqual(contract["serialized_format"],
                         "disabled_by_worker_mode")
        self.assertEqual(body["hawking"]["tools_used"], [])
        self.assertNotIn("hawking_worker_mode", backend.seen[0])
        self.assertNotIn("tools", backend.seen[0])
        self.assertTrue(contract["caller_tool_schemas_withheld"])

    def test_unknown_worker_mode_is_refused_before_provider_call(self):
        backend = _Backend()
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(backend, "kimi-test", greedy=False,
                         health={"status": "ok", "resident": "kimi-test"}))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        code, _, raw = _post(
            f"http://127.0.0.1:{httpd.server_address[1]}",
            {"messages": [{"role": "user", "content": "hi"}],
             "hawking_worker_mode": "write_everything"})
        self.assertEqual(code, 400, raw)
        self.assertIn("unknown hawking_worker_mode", json.loads(raw)["error"]["message"])
        self.assertEqual(backend.seen, [])

    def test_worker_mode_parser_accepts_only_declared_values(self):
        self.assertEqual(worker_request_mode("reason_only").value, "reason_only")
        with self.assertRaises(ValueError):
            worker_request_mode("unbounded")


class TestGreedyDetection(unittest.TestCase):
    def test_do_sample_false_is_greedy(self, ):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "profile.json"
            p.write_text(json.dumps({"generation": {"do_sample": False}}))
            self.assertTrue(profile_is_greedy(str(p)))

    def test_a_sampling_profile_is_not_greedy(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "profile.json"
            p.write_text(json.dumps({"generation": {"do_sample": True, "temperature": 0.7}}))
            self.assertFalse(profile_is_greedy(str(p)))

    def test_the_real_sealed_profile_is_greedy(self):
        import pathlib
        profile = pathlib.Path(__file__).resolve().parents[1] / "hawking-native.sealed-3.14.json"
        self.assertTrue(profile.is_file(), profile)
        self.assertTrue(profile_is_greedy(str(profile)),
                        "the sealed profile decodes argmax and must be detected as greedy")

    def test_true_is_not_accepted_as_the_integer_one(self):
        # `True == 1`, so a bare membership test would wave top_k=True through.
        self.assertIsNotNone(sampler_refusal({"top_k": True}))
        self.assertIsNone(sampler_refusal({"top_k": 1}))

    def test_unset_samplers_are_fine(self):
        self.assertIsNone(sampler_refusal({}))
        self.assertIsNone(sampler_refusal({k: None for k in GREEDY_OK}))


class TestPayloadHelpers(unittest.TestCase):
    def test_single_chunk_streaming_is_labelled_in_the_answer(self):
        body = chat_payload(_Result(), "sealed-3.14", request_id="chatcmpl-x")
        self.assertEqual(body["hawking"]["streaming"], "single-chunk")

    def test_frames_end_with_done(self):
        frames = list(stream_frames(_Result(), "m", request_id="r"))
        self.assertEqual(frames[-1], b"data: [DONE]\n\n")

    def test_models_payload_shape(self):
        self.assertEqual(models_payload("m")["data"][0]["id"], "m")

    def test_known_template_markers_never_reach_visible_answer(self):
        body = chat_payload(
            type("R", (), {"text": "answer<|im_end|>", "finish_reason": "stop",
                            "degraded": [], "raw": {}})(),
            "m", request_id="marker-test")
        self.assertEqual(body["choices"][0]["message"]["content"], "answer")
        self.assertEqual(visible_text("<|im_start|>x<|eot_id|>"), "x")


if __name__ == "__main__":
    unittest.main()
