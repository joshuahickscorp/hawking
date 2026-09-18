import http.client
import json
import tempfile
import threading
import urllib.error
from contextlib import contextmanager
from types import SimpleNamespace
from threading import Event

import pytest

from hawking.providers import GenerationRequest, GenerationResponse
from hawking.remote_cognition import (
    OpenRouterGatewayPolicy,
    OpenRouterProvider,
    RemoteCancelledError,
    RemoteCostPolicy,
    RemoteCognitionError,
    RemoteRateLimitError,
    _semantic_response_problem,
    _surface_semantic_miss_as_response,
)
from hawking.serve import make_handler


class _Backend:
    identity = "hawking-native-test"
    greedy = False
    state = "ready"
    native_tools = False

    def catalog(self):
        return [{
            "id": self.identity,
            "object": "model",
            "created": 1,
            "owned_by": "hawking",
        }]

    def complete(self, payload, timeout=None):
        del timeout
        return GenerationResponse(
            text=str(payload.get("messages", [{}])[-1].get("content") or ""),
            raw={"choices": [{"message": {"content": "native"}, "finish_reason": "stop"}]},
        )


class _FakeOpenRouter:
    requests = []

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.last_receipt = {
            "schema": "hawking.remote.cognition.receipt.v1",
            "status": "completed",
            "provider": "openrouter",
            "model_id": model_id,
            "pid": None,
            "port": None,
        }

    def generate(self, request, *, timeout=None):
        del timeout
        self.__class__.requests.append(request)
        return GenerationResponse(
            text="remote answer",
            raw={
                "id": "or-response-1",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "remote answer",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 3,
                    "total_tokens": 7,
                    "cost": 0.01,
                },
                "remote_cognition": self.last_receipt,
            },
            usage={
                "prompt_tokens": 4,
                "completion_tokens": 3,
                "total_tokens": 7,
                "cost_usd": 0.01,
                "remote_request_id": "or-response-1",
            },
            finish_reason="tool_calls",
            provider="openrouter",
            request_id=request.request_id,
        )

    def generate_stream(self, request, *, timeout=None):
        del timeout
        self.__class__.requests.append(request)
        yield {"type": "delta", "text": "streamed ", "tool_calls": None}
        yield {
            "type": "delta",
            "text": "answer",
            "tool_calls": [{
                "index": 0,
                "id": "call_stream",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }],
        }
        yield {
            "type": "done",
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            "receipt": self.last_receipt,
        }

    def cancel(self, request_id):
        return {"request_id": request_id, "cancel_requested": True, "pid": None, "port": None}


@contextmanager
def _server(monkeypatch, *, registry=None, state_root=None):
    monkeypatch.setenv("HAWKING_REMOTE_MAX_COST_USD", "1")
    monkeypatch.setenv("HAWKING_OPENROUTER_MODELS", "openai/gpt-5.6-luna")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _FakeOpenRouter.requests = []
    import hawking.remote_cognition as remote_cognition

    monkeypatch.setattr(remote_cognition, "OpenRouterProvider", _FakeOpenRouter)
    server = __import__("http.server", fromlist=["ThreadingHTTPServer"]).ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            _Backend(),
            "hawking-native-test",
            greedy=False,
            health={"status": "ok"},
            registry=registry,
            stores=({"state_root": state_root} if state_root else None),
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    payload = json.dumps(body).encode() if body is not None else None
    connection.request(method, path, body=payload, headers={
        "Content-Type": "application/json",
        **(headers or {}),
    })
    response = connection.getresponse()
    raw = response.read()
    status = response.status
    response_headers = dict(response.getheaders())
    connection.close()
    return status, response_headers, raw


def test_structured_request_accepts_named_intermediate_tool_call_without_content():
    request = GenerationRequest.from_mapping({
        "model": "moonshotai/kimi-k3",
        "messages": [{"role": "user", "content": "inspect the source"}],
        "tools": [{
            "type": "function",
            "function": {"name": "repo_read", "parameters": {"type": "object"}},
        }],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
    })
    payload = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "repo_read", "arguments": "{\"path\":\"README.md\"}"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
    }
    assert _semantic_response_problem(request, payload, None, "tool_calls") is None


def test_structured_request_still_rejects_empty_tool_call_without_content():
    request = GenerationRequest.from_mapping({
        "model": "moonshotai/kimi-k3",
        "messages": [{"role": "user", "content": "inspect the source"}],
        "tools": [{
            "type": "function",
            "function": {"name": "repo_read", "parameters": {"type": "object"}},
        }],
        "response_format": {"type": "json_object"},
    })
    payload = {
        "choices": [{
            "message": {"role": "assistant", "content": None, "tool_calls": [{}]},
            "finish_reason": "tool_calls",
        }],
    }
    problem = _semantic_response_problem(request, payload, None, "tool_calls")
    assert problem is not None
    assert problem["reason"] == "structured_content_missing"


def test_typed_workunit_surfaces_invalid_structured_http_200_as_status_packet():
    request = GenerationRequest.from_mapping({
        "model": "deepseek/deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": "return a patch"}],
        "response_format": {"type": "json_object"},
        "transport_options": {"hawking_semantic_miss_as_response": True},
    })
    payload = {
        "choices": [{
            "message": {"role": "assistant", "content": "I will make the edit."},
            "finish_reason": "length",
        }],
    }
    problem = _semantic_response_problem(
        request, payload, "I will make the edit.", "length",
    )
    assert problem is not None
    assert problem["code"] == "REMOTE_SEMANTIC_INVALID"
    assert _surface_semantic_miss_as_response(request, problem) is True


def test_models_exposes_configured_openrouter_without_fetch(monkeypatch):
    with _server(monkeypatch) as server:
        status, _headers, raw = _request(server, "GET", "/v1/models")
    assert status == 200
    data = json.loads(raw)
    # The current minimal H-Web catalog does not expose arbitrary configured
    # cloud IDs until Hawking's explicit model search qualifies them.  With no
    # credential and no search query, only the honest withheld local identity
    # may appear; direct gateway calls remain separately testable below.
    assert {row["id"] for row in data["data"]} == {"KIMI_P0_OPERATIONAL"}
    assert _FakeOpenRouter.requests == []


def test_chat_uses_same_provider_and_forwards_tools_but_not_client_metadata(monkeypatch):
    with _server(monkeypatch) as server:
        status, _headers, raw = _request(
            server,
            "POST",
            "/v1/chat/completions",
            {
                "model": "openai/gpt-5.6-luna",
                "messages": [{"role": "user", "content": "find it"}],
                "tools": [{
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }],
                "tool_choice": "auto",
                "models": ["different/model"],
                "provider": {"order": ["ExampleProvider"], "allow_fallbacks": False},
                "metadata": {"secret": "must-not-cross"},
            },
            {"X-Hawking-Client": "generic-test", "X-Hawking-Project": "project-a"},
        )
    assert status == 200
    output = json.loads(raw)
    assert output["model"] == "openai/gpt-5.6-luna"
    assert output["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "lookup"
    assert output["hawking"]["provider"] == "openrouter"
    request = _FakeOpenRouter.requests[0]
    assert request.model == "openai/gpt-5.6-luna"
    assert len(request.tools) == 1
    assert request.transport_options["provider"]["order"] == ["ExampleProvider"]
    assert "models" not in request.transport_options
    assert "secret" not in json.dumps(request.metadata)
    payload = request.to_payload()
    assert "transport_options" not in payload
    assert payload["tool_choice"] == "auto"


def test_responses_maps_input_and_returns_responses_shape(monkeypatch):
    with _server(monkeypatch) as server:
        status, _headers, raw = _request(
            server,
            "POST",
            "/v1/responses",
            {
                "model": "openrouter:openai/gpt-5.6-luna",
                "instructions": "be concise",
                "input": "hello",
                "max_output_tokens": 32,
            },
        )
    assert status == 200
    output = json.loads(raw)
    assert output["object"] == "response"
    assert output["model"] == "openai/gpt-5.6-luna"
    assert output["output"][0]["content"][0]["text"] == "remote answer"
    request = _FakeOpenRouter.requests[0]
    assert [message["role"] for message in request.messages] == ["system", "user"]
    assert request.max_tokens == 32


def test_chat_stream_is_incremental_sse_and_never_executes_client_tool(monkeypatch):
    with _server(monkeypatch) as server:
        status, headers, raw = _request(
            server,
            "POST",
            "/v1/chat/completions",
            {
                "model": "openai/gpt-5.6-luna",
                "messages": [{"role": "user", "content": "stream"}],
                "stream": True,
                "stream_options": {"include_usage": True},
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
            },
        )
    text = raw.decode()
    assert status == 200
    assert headers["Content-Type"].startswith("text/event-stream")
    assert "streamed " in text and "answer" in text
    assert '"tool_calls"' in text
    assert '"usage"' in text
    assert "data: [DONE]" in text
    assert len(_FakeOpenRouter.requests) == 1


def test_responses_stream_emits_standard_response_events(monkeypatch):
    with _server(monkeypatch) as server:
        status, headers, raw = _request(
            server,
            "POST",
            "/v1/responses",
            {
                "model": "openai/gpt-5.6-luna",
                "input": "stream this response",
                "stream": True,
            },
        )
    text = raw.decode()
    assert status == 200
    assert headers["Content-Type"].startswith("text/event-stream")
    assert "event: response.created" in text
    assert "event: response.output_text.delta" in text
    assert "event: response.completed" in text
    assert "streamed " in text and "answer" in text
    assert "data: [DONE]" in text


class _ToolSpec:
    name = "repo.read"
    description = "Read an approved repository file."
    mutation = "read_only"
    input_schema = {
        "type": "object",
        "required": ["path"],
        "properties": {"path": {"type": "string"}},
    }


class _ToolResult:
    ok = True
    value = {"path": "README.md", "content": "observed"}
    error = None

    def to_dict(self):
        return {"ok": self.ok, "value": self.value, "error": self.error}


class _Registry:
    context = SimpleNamespace(permissions={"read_only"})

    def discover(self, include_aliases=False):
        del include_aliases
        return [{
            "name": "repo.read",
            "description": _ToolSpec.description,
            "mutation": "read_only",
        }]

    def get(self, name):
        return _ToolSpec() if name == "repo.read" else None

    def invoke(self, name, arguments):
        assert name == "repo.read"
        assert arguments == {"path": "README.md"}
        return _ToolResult()


class _HWebToolSpec:
    name = "source.owner"
    description = "Locate the canonical source owner."
    mutation = "read_only"
    input_schema = {
        "type": "object",
        "required": ["symbol"],
        "properties": {"symbol": {"type": "string"}},
    }


class _HWebRegistry:
    context = SimpleNamespace(permissions={"read_only"})

    def discover(self, include_aliases=False):
        del include_aliases
        return [{
            "name": "source.owner",
            "description": _HWebToolSpec.description,
            "mutation": "read_only",
        }]

    def get(self, name):
        return _HWebToolSpec() if name == "source.owner" else None

    def invoke(self, name, arguments):
        assert name == "source.owner"
        assert arguments == {"symbol": "h web"}
        return _ToolResult()


class _HWebStreamingProvider(_FakeOpenRouter):
    contexts = []

    def _workunit_tool_schemas(self, context):
        self.__class__.contexts.append(dict(context))
        return (({
            "type": "function",
            "function": {
                "name": "source_owner",
                "description": "Locate source owner.",
                "parameters": _HWebToolSpec.input_schema,
            },
        },), ["source.owner"])

    def generate_stream_with_workunit_tools(self, request, context, schemas, names):
        self.__class__.requests.append(request)
        assert schemas and names == ["source.owner"]
        result = context["tool_dispatch"]("source.owner", {"symbol": "h web"})
        observation = result.to_dict()
        yield {
            "type": "tool",
            "trace": {
                "tool": "source.owner", "wire_tool": "source_owner",
                "dispatched": True, "ok": observation["ok"],
            },
        }
        yield {"type": "delta", "text": "current ", "tool_calls": None}
        yield {"type": "delta", "text": "source owner", "tool_calls": None}
        yield {
            "type": "done",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            "receipt": self.last_receipt,
            "tool_trace": [{"tool": "source.owner", "dispatched": True, "ok": True}],
            "tool_admission": {"tool_count": 1, "effect_ceiling": "READ"},
        }


def test_hweb_production_stream_projects_read_tools_for_auto_and_pinned(monkeypatch):
    """Exercise the real H-Web route, not its helper or a generic gateway call."""
    import hawking.remote_cognition as remote_cognition

    _HWebStreamingProvider.requests = []
    _HWebStreamingProvider.contexts = []
    with tempfile.TemporaryDirectory() as tmp:
        with _server(monkeypatch, registry=_HWebRegistry(), state_root=tmp) as server:
            # _server installs the generic fake; replace it only for this
            # production H-Web route regression.
            monkeypatch.setattr(remote_cognition, "OpenRouterProvider", _HWebStreamingProvider)
            for model, session_id in (
                ("openai/gpt-5.6-luna", "pinned-read"),
                ("hawking/auto", "auto-read"),
            ):
                status, headers, raw = _request(
                    server,
                    "POST",
                    "/hawking/web/chat",
                    {
                        "session_id": session_id,
                        "model": model,
                        "message": (
                            "Find the current canonical implementation of h web "
                            "using Hawking's current source tools."
                        ),
                        "stream": True,
                    },
                )
                assert status == 200, raw.decode()
                assert headers["Content-Type"].startswith("text/event-stream")
                rendered = raw.decode()
                assert "current " in rendered and "source owner" in rendered
                assert '"tool":"source.owner"' in rendered
                assert '"effect_ceiling":"READ"' in rendered

    assert len(_HWebStreamingProvider.contexts) == 2
    for context in _HWebStreamingProvider.contexts:
        assert context["tool_write_authority"] is False
        assert "source.owner" in context["allowed_tool_names"]
        assert "repo.edit" not in context["allowed_tool_names"]


def test_streaming_hawking_loop_keeps_provider_deltas_across_tool_continuation():
    """A real provider transport receives schemas and streams after a tool."""
    requests = []

    def urlopen(request, timeout):
        del timeout
        payload = json.loads(request.data.decode())
        requests.append(payload)
        if len(requests) == 1:
            tool_name = payload["tools"][0]["function"]["name"]
            return _SSEResponse([
                ("data: " + json.dumps({
                    "id": "tool-turn",
                    "choices": [{"delta": {"tool_calls": [{
                        "index": 0, "id": "call_owner", "type": "function",
                        "function": {"name": tool_name, "arguments": '{"symbol":"h web"}'},
                    }]}}],
                }) + "\n").encode(),
                b"\n",
                b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n',
                b"\n", b"data: [DONE]\n", b"\n",
            ])
        return _SSEResponse([
            b'data: {"id":"final-turn","choices":[{"delta":{"content":"grounded "}}]}\n', b"\n",
            b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}\n', b"\n",
            b"data: [DONE]\n", b"\n",
        ])

    provider = OpenRouterProvider(
        "openai/gpt-5.6-luna",
        cost_policy=RemoteCostPolicy(max_cost_usd=1),
        key_resolver=lambda: "test-key",
        urlopen=urlopen,
        max_attempts=1,
    )
    context = {
        "tool_registry": _HWebRegistry(),
        "tool_write_authority": False,
        "mutation_lock_held": False,
        "allowed_tool_names": ["source.owner"],
        "tool_dispatch": lambda name, args: _HWebRegistry().invoke(name, args),
        "max_tool_calls": 2,
    }
    schemas, names = provider._workunit_tool_schemas(context)
    request = GenerationRequest(
        model="openai/gpt-5.6-luna",
        messages=({"role": "user", "content": "find h web"},),
        request_id="streaming-hweb-tool-regression",
    )
    events = list(provider.generate_stream_with_workunit_tools(
        request, context, tuple(schemas), names
    ))

    assert len(requests) == 2
    assert all(item.get("tools") for item in requests)
    assert [item["text"] for item in events if item["type"] == "delta"] == [
        "grounded ", "answer"
    ]
    assert any(item["type"] == "tool" and item["trace"]["tool"] == "source.owner"
               for item in events)
    assert events[-1]["tool_admission"]["effect_ceiling"] == "READ"


class _JSONResponse:
    def __init__(self, value, *, status=200, headers=None):
        self.value = value
        self.status = status
        self.headers = headers or {}

    def read(self):
        return json.dumps(self.value).encode()

    def close(self):
        return None


class _SSEResponse:
    status = 200
    headers = {"x-request-id": "sse-request"}

    def __init__(self, lines):
        self.lines = iter(lines)

    def readline(self):
        return next(self.lines, b"")

    def close(self):
        return None


def test_http_200_empty_length_is_rejected_as_semantic_failure(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    provider = OpenRouterProvider(
        "openai/gpt-5.6-luna",
        cost_policy=RemoteCostPolicy(max_cost_usd=1),
        urlopen=lambda _request, timeout: _JSONResponse({
            "id": "empty-200",
            "choices": [{
                "message": {"role": "assistant", "content": None},
                "finish_reason": "length",
            }],
        }),
        max_attempts=1,
    )
    request = GenerationRequest(
        model="openai/gpt-5.6-luna",
        messages=({"role": "user", "content": "probe"},),
        tools=({
            "type": "function",
            "function": {
                "name": "route_probe",
                "parameters": {"type": "object"},
            },
        },),
        transport_options={"tool_choice": "required"},
        metadata={"cost_authorization": {"max_usd": 1}},
    )
    with pytest.raises(RemoteCognitionError) as raised:
        provider.generate(request)
    assert raised.value.code == "REMOTE_SEMANTIC_INCOMPLETE"
    assert raised.value.status_code == 502


def test_typed_workunit_surfaces_http_200_missing_tool_as_status_not_502(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    provider = OpenRouterProvider(
        "openai/gpt-5.6-luna",
        cost_policy=RemoteCostPolicy(max_cost_usd=1),
        urlopen=lambda _request, timeout: _JSONResponse({
            "id": "missing-tool-200",
            "choices": [{
                "message": {"role": "assistant", "content": None},
                "finish_reason": "stop",
            }],
        }),
        max_attempts=1,
    )
    request = GenerationRequest(
        model="openai/gpt-5.6-luna",
        messages=({"role": "user", "content": "emit one edit"},),
        tools=({
            "type": "function",
            "function": {"name": "repo_edit", "parameters": {"type": "object"}},
        },),
        transport_options={
            "tool_choice": "required",
            "hawking_semantic_miss_as_response": True,
        },
        metadata={"cost_authorization": {"max_usd": 1}},
    )

    result = provider.generate(request)

    status = json.loads(result.text)
    assert result.finish_reason == "stop"
    assert status == {
        "provider_semantic_code": "REMOTE_SEMANTIC_INCOMPLETE",
        "reason": "required_tool_call_missing",
        "schema": "hawking.provider.semantic_miss.v1",
        "status": "MUTATION_PAYLOAD_MISSING",
    }
    assert result.raw["remote_cognition"]["status"] == "semantic_incomplete"
    assert result.usage["hawking_semantic_miss"] == status


def test_http_200_typed_json_wrapped_by_provider_is_not_mislabeled_502(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    provider = OpenRouterProvider(
        "openai/gpt-5.6-luna",
        cost_policy=RemoteCostPolicy(max_cost_usd=1),
        urlopen=lambda _request, timeout: _JSONResponse({
            "id": "wrapped-typed-200",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        "```json\n"
                        '{"status":"MUTATION_PROPOSED","schema":"HAWKING_EDIT_V1",'
                        '"file":"mod.py","mode":"anchor_edit",'
                        '"anchor_before":"VALUE = 1","replacement":"VALUE = 2"}'
                        "\n```"
                    ),
                },
                "finish_reason": "stop",
            }],
        }),
        max_attempts=1,
    )
    request = GenerationRequest(
        model="openai/gpt-5.6-luna",
        messages=({"role": "user", "content": "emit one edit"},),
        response_schema={"type": "json_object"},
        metadata={"cost_authorization": {"max_usd": 1}},
    )
    result = provider.generate(request)
    assert "MUTATION_PROPOSED" in result.text
    assert result.finish_reason == "stop"


def test_typed_workunit_request_fast_fails_upstream_5xx_for_route_recovery(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []

    def urlopen(_request, timeout):
        calls.append(timeout)
        raise urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            502,
            "upstream failure",
            {},
            None,
        )

    provider = OpenRouterProvider(
        "openai/gpt-5.6-luna",
        cost_policy=RemoteCostPolicy(max_cost_usd=1),
        urlopen=urlopen,
        sleeper=lambda _seconds: None,
        max_attempts=3,
    )
    request = GenerationRequest(
        model="openai/gpt-5.6-luna",
        messages=({"role": "user", "content": "probe"},),
        transport_options={"hawking_fast_failover": True},
        metadata={"cost_authorization": {"max_usd": 1}},
    )
    with pytest.raises(RemoteCognitionError) as raised:
        provider.generate(request)
    assert raised.value.status_code == 502
    assert len(calls) == 1


def test_http_200_required_tool_call_remains_a_valid_semantic_completion(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    provider = OpenRouterProvider(
        "openai/gpt-5.6-luna",
        cost_policy=RemoteCostPolicy(max_cost_usd=1),
        urlopen=lambda _request, timeout: _JSONResponse({
            "id": "tool-200",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_probe",
                        "type": "function",
                        "function": {"name": "route_probe", "arguments": "{}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }),
        max_attempts=1,
    )
    request = GenerationRequest(
        model="openai/gpt-5.6-luna",
        messages=({"role": "user", "content": "probe"},),
        tools=({
            "type": "function",
            "function": {
                "name": "route_probe",
                "parameters": {"type": "object"},
            },
        },),
        transport_options={"tool_choice": "required"},
        metadata={"cost_authorization": {"max_usd": 1}},
    )
    result = provider.generate(request)
    assert result.finish_reason == "tool_calls"
    assert result.raw["choices"][0]["message"]["tool_calls"]


def test_http_200_tool_finish_without_callable_payload_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    provider = OpenRouterProvider(
        "openai/gpt-5.6-luna",
        cost_policy=RemoteCostPolicy(max_cost_usd=1),
        urlopen=lambda _request, timeout: _JSONResponse({
            "id": "malformed-tool-200",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{}],
                },
                "finish_reason": "tool_calls",
            }],
        }),
        max_attempts=1,
    )
    request = GenerationRequest(
        model="openai/gpt-5.6-luna",
        messages=({"role": "user", "content": "probe"},),
        tools=({
            "type": "function",
            "function": {
                "name": "route_probe",
                "parameters": {"type": "object"},
            },
        },),
        transport_options={"tool_choice": "required"},
        metadata={"cost_authorization": {"max_usd": 1}},
    )
    with pytest.raises(RemoteCognitionError) as raised:
        provider.generate(request)
    assert raised.value.code == "REMOTE_SEMANTIC_INCOMPLETE"
    assert raised.value.status_code == 502


def test_openrouter_default_concurrency_is_unbounded_but_explicit_policy_still_works(monkeypatch):
    monkeypatch.delenv("HAWKING_OPENROUTER_CONCURRENCY", raising=False)
    assert OpenRouterGatewayPolicy.from_environment().concurrency is None
    monkeypatch.setenv("HAWKING_OPENROUTER_CONCURRENCY", "3")
    assert OpenRouterGatewayPolicy.from_environment().concurrency == 3
    monkeypatch.setenv("HAWKING_OPENROUTER_CONCURRENCY", "unlimited")
    assert OpenRouterGatewayPolicy.from_environment().concurrency is None


def test_hawking_workunit_uses_same_openrouter_provider_with_authorized_tools(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    requests = []

    def urlopen(request, timeout):
        del timeout
        payload = json.loads(request.data.decode())
        requests.append(payload)
        if len(requests) == 1:
            value = {
                "id": "remote-tool-call",
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "repo_read",
                                "arguments": '{"path":"README.md"}',
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        else:
            value = {
                "id": "remote-final",
                "choices": [{
                    "message": {
                        "content": '{"observations":["tool result seen"],"status":"observed"}',
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        return _JSONResponse(value)

    provider = OpenRouterProvider(
        "openai/gpt-5.6-luna",
        cost_policy=RemoteCostPolicy(max_cost_usd=1),
        urlopen=urlopen,
        sleeper=lambda _seconds: None,
        max_attempts=1,
    )
    unit = SimpleNamespace(id="WU-remote-tools", description="inspect README")
    result = provider.execute_workunit(unit, {
        "objective": "inspect README",
        "tool_registry": _Registry(),
        "allowed_tool_names": ["repo.read"],
        "hawking_tools": True,
        "max_tool_calls": 2,
    })
    assert result["status"] == "observed"
    assert result["tool_call_count"] == 1
    assert result["tool_trace"][0]["tool"] == "repo.read"
    assert result["tool_trace"][0]["dispatched"] is True
    assert requests[0]["tools"][0]["function"]["name"] == "repo_read"
    assert any(message.get("role") == "tool" for message in requests[1]["messages"])


def test_workunit_returns_refusals_for_tool_calls_over_budget(monkeypatch):
    """Every provider call id needs an output, including a bounded refusal."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    requests = []

    def urlopen(request, timeout):
        del timeout
        payload = json.loads(request.data.decode())
        requests.append(payload)
        if len(requests) == 1:
            calls = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "repo_read",
                        "arguments": '{"path":"README.md"}',
                    },
                }
                for i in range(3)
            ]
            value = {
                "id": "remote-three-tools",
                "choices": [{
                    "message": {"content": None, "tool_calls": calls},
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        else:
            value = {
                "id": "remote-three-tools-final",
                "choices": [{
                    "message": {
                        "content": '{"observations":["all call ids answered"],"status":"observed"}',
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        return _JSONResponse(value)

    provider = OpenRouterProvider(
        "openai/gpt-5.6-luna",
        cost_policy=RemoteCostPolicy(max_cost_usd=1),
        urlopen=urlopen,
        sleeper=lambda _seconds: None,
        max_attempts=1,
    )
    unit = SimpleNamespace(id="WU-three-tools", description="bounded read")
    result = provider.execute_workunit(unit, {
        "objective": "bounded read",
        "tool_registry": _Registry(),
        "allowed_tool_names": ["repo.read"],
        "hawking_tools": True,
        "max_tool_calls": 2,
    })
    assert result["status"] == "observed"
    assert result["tool_call_count"] == 3
    assert [row["dispatched"] for row in result["tool_trace"]] == [True, True, False]
    assert result["tool_trace"][2]["error"] == "tool call budget exhausted"
    tool_messages = [message for message in requests[1]["messages"] if message.get("role") == "tool"]
    assert len(tool_messages) == 3


def test_openrouter_policy_routes_privacy_and_provider_denylists(monkeypatch):
    monkeypatch.setenv("HAWKING_OPENROUTER_ALLOWED_PROVIDERS", "ProviderA,ProviderB")
    monkeypatch.setenv("HAWKING_OPENROUTER_DENIED_PROVIDERS", "ProviderB")
    monkeypatch.setenv("HAWKING_OPENROUTER_REQUIRE_ZDR", "true")
    monkeypatch.setenv("HAWKING_OPENROUTER_SORT", "latency")
    policy = OpenRouterGatewayPolicy.from_environment()
    options = policy.transport_options()
    assert options["provider"]["order"] == ["ProviderA"]
    assert options["provider"]["ignore"] == ["ProviderB"]
    assert options["provider"]["data_collection"] == "deny"
    assert options["provider"]["sort"] == "latency"
    assert options["provider"]["allow_fallbacks"] is False


def test_default_model_provider_preferences_are_pinned_with_same_model_fallback():
    policy = OpenRouterGatewayPolicy.from_environment()
    deepseek = policy.transport_options("deepseek/deepseek-v4.1-flash")
    qwen = policy.transport_options("qwen/qwen3.8-flash")
    assert deepseek["provider"]["order"] == ["together"]
    assert deepseek["provider"]["allow_fallbacks"] is True
    assert qwen["provider"]["order"] == ["alibaba"]
    assert qwen["provider"]["allow_fallbacks"] is True


def test_openrouter_stream_parser_and_bounded_errors(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    request = GenerationRequest(
        model="openai/gpt-5.6-luna",
        messages=({"role": "user", "content": "hello"},),
        metadata={"cost_authorization": {"max_usd": 1}},
    )
    stream_response = _SSEResponse([
        b'data: {"id":"stream-1","choices":[{"delta":{"content":"hi "}}]}\n',
        b"\n",
        b'data: {"choices":[{"delta":{"content":"there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n',
        b"\n",
        b"data: [DONE]\n",
        b"\n",
    ])
    provider = OpenRouterProvider(
        request.model,
        cost_policy=RemoteCostPolicy(max_cost_usd=1),
        urlopen=lambda _request, timeout: stream_response,
        max_attempts=1,
    )
    events = list(provider.generate_stream(request))
    assert [event["text"] for event in events if event["type"] == "delta"] == ["hi ", "there"]
    assert events[-1]["remote_request_id"] == "sse-request"

    rate_limited = OpenRouterProvider(
        request.model,
        cost_policy=RemoteCostPolicy(max_cost_usd=1),
        urlopen=lambda _request, timeout: _JSONResponse(
            {"error": "slow down"}, status=429, headers={"Retry-After": "0"}
        ),
        sleeper=lambda _seconds: None,
        max_attempts=2,
    )
    with pytest.raises(RemoteRateLimitError):
        rate_limited.generate(request)

    cancelled = Event()
    cancelled.set()
    provider = OpenRouterProvider(
        request.model,
        cost_policy=RemoteCostPolicy(max_cost_usd=1),
        urlopen=lambda *_args, **_kwargs: pytest.fail("cancelled request reached transport"),
        max_attempts=1,
    )
    cancelled_request = GenerationRequest(
        model=request.model,
        messages=request.messages,
        metadata={"cancel_event": cancelled, "cost_authorization": {"max_usd": 1}},
    )
    with pytest.raises(RemoteCancelledError):
        provider.generate(cancelled_request)
