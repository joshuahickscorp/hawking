#!/usr/bin/env python3
"""Protected HCLI model-call-path checks.

Plain python3 + assert. No pytest fixtures. Must also pass under pytest.
Must exit non-zero on failure and print one line per check.

Covers the structured-output call path that was truncating on Qwen3 thinking
tokens (finish_reason=length) and reporting only
"Model did not return a valid structured JSON object".

Run:
    python3 tools/headless/hcli_callpath_test.py
    pytest tools/headless/hcli_callpath_test.py -q
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parents[2]

from hcli.config import Config  # noqa: E402
from hcli.engine import Engine, EngineError  # noqa: E402
from hcli.events import EventBus  # noqa: E402
from hcli.workspace import Workspace  # noqa: E402

ANSWER = {
    "kind": "answer",
    "content": "ok",
    "operations": [],
    "tests": [],
}


@contextlib.contextmanager
def _env(**kwargs: Optional[str]):
    saved: Dict[str, Optional[str]] = {}
    try:
        for key, value in kwargs.items():
            saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


class _Pool:
    def __init__(self, port: int = 18765) -> None:
        runtime = type("Runtime", (), {})()
        runtime.index = 0
        runtime.pid = 1
        runtime.port = port
        runtime.active = True
        self.runtimes = [runtime]


class _Stub:
    def __init__(self, body: Any) -> None:
        self.body = body
        self.calls = []

    def __call__(self, endpoint, payload, timeout):
        self.calls.append(
            {
                "endpoint": endpoint,
                "payload": payload,
                "timeout": timeout,
            }
        )
        if isinstance(self.body, Exception):
            raise self.body
        if callable(self.body):
            return self.body(endpoint, payload, timeout)
        return self.body


def _engine(root: Path) -> Engine:
    cfg = Config(str(root), global_path=str(root / "global-config.json"))
    pool = _Pool()
    return Engine(
        workspace=Workspace(str(root)),
        event_bus=EventBus(),
        runtime_provider=lambda: pool,
        runtime_state_provider=lambda: pool,
        runtime_count=1,
        model_name="local",
        config=cfg,
    )


def _completion(
    content: str,
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
) -> Dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def check_payload_shape():
    """Structured calls think off + schema on; flipping config reverses both."""
    with tempfile.TemporaryDirectory() as tmp:
        with _env(
            HCLI_ENABLE_THINKING=None,
            HCLI_RESPONSE_SCHEMA=None,
            HCLI_MODEL_TOKENS=None,
        ):
            engine = _engine(Path(tmp))
            payload = engine._build_model_payload("say ok")
            thinking = payload.get("chat_template_kwargs") or {}
            assert thinking.get("enable_thinking") is False, (
                "structured call payload missing "
                f"chat_template_kwargs.enable_thinking=false: "
                f"chat_template_kwargs={thinking!r} "
                f"keys={sorted(payload)}"
            )
            fmt = payload.get("response_format")
            assert isinstance(fmt, dict), (
                "structured call payload missing response_format: "
                f"keys={sorted(payload)}"
            )
            schema = ((fmt.get("json_schema") or {}).get("schema")) or {}
            required = set(schema.get("required") or [])
            assert {
                "kind",
                "content",
                "operations",
                "tests",
            } <= required, (
                "response_format schema required fields "
                f"missing kind/content/operations/tests: {required!r}"
            )

            with _env(
                HCLI_ENABLE_THINKING="true",
                HCLI_RESPONSE_SCHEMA="off",
            ):
                flipped = engine._build_model_payload("say ok")
            assert (
                flipped.get("chat_template_kwargs") or {}
            ).get("enable_thinking") is True, (
                "flipping enable_thinking did not re-enable thinking: "
                f"{(flipped.get('chat_template_kwargs') or {})!r}"
            )
            assert "response_format" not in flipped, (
                "flipping response_schema=off did not drop response_format: "
                f"keys={sorted(flipped)}"
            )

            engine.config.save_project(
                {
                    "enable_thinking": True,
                    "response_schema": "off",
                }
            )
            via_file = engine._build_model_payload("say ok")
            assert (
                via_file.get("chat_template_kwargs") or {}
            ).get("enable_thinking") is True, via_file
            assert "response_format" not in via_file, via_file


def check_length_failure_is_diagnosable():
    """finish_reason=length must name the token budget and prompt size."""
    with tempfile.TemporaryDirectory() as tmp:
        with _env(HCLI_MODEL_TOKENS="6500"):
            engine = _engine(Path(tmp))
            stub = _Stub(
                _completion(
                    '{"kind": "answer", "content": "truncated',
                    finish_reason="length",
                    prompt_tokens=7257,
                    completion_tokens=6500,
                )
            )
            engine._post_completion = stub
            with _env(HCLI_MODEL_TOKENS="6500"):
                try:
                    engine._call_model("mission")
                    raise AssertionError("length failure did not raise")
                except EngineError as exc:
                    err = str(exc)
            assert "6500-token" in err, err
            assert "7257-token" in err, err
            assert "completion budget" in err, err

            engine2 = _engine(Path(tmp))
            engine2._post_completion = stub
            result = engine2.execute("mission")
            assert result.get("status") == "failed", result
            assert "6500-token" in str(result.get("error") or ""), result
            receipt = json.loads(
                Path(result["receipt"]).read_text(encoding="utf-8")
            )
            calls = receipt.get("model_calls") or []
            assert calls, receipt
            assert calls[0].get("finish_reason") == "length", receipt


def check_telemetry_reaches_receipt():
    """A successful call writes model_calls with the measured fields."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        stub = _Stub(
            _completion(
                json.dumps(ANSWER),
                finish_reason="stop",
                prompt_tokens=111,
                completion_tokens=22,
            )
        )
        engine._post_completion = stub
        result = engine.execute("say ok")
        assert result.get("status") == "completed", result
        receipt = json.loads(
            Path(result["receipt"]).read_text(encoding="utf-8")
        )
        calls = receipt.get("model_calls") or []
        assert len(calls) == 1, (
            "receipt missing model_calls telemetry: "
            f"keys={sorted(receipt)}"
        )
        call = calls[0]
        assert call.get("endpoint"), call
        assert "127.0.0.1" in str(call.get("endpoint")), call
        assert call.get("finish_reason") == "stop", call
        assert call.get("prompt_tokens") == 111, call
        assert call.get("completion_tokens") == 22, call
        assert isinstance(call.get("wall_s"), (int, float)), call
        assert call.get("wall_s") >= 0, call


def check_degradation_path():
    """response_schema=off drops response_format and still completes."""
    with tempfile.TemporaryDirectory() as tmp:
        with _env(HCLI_RESPONSE_SCHEMA="off"):
            engine = _engine(Path(tmp))
            payload = engine._build_model_payload("say ok")
            assert "response_format" not in payload, (
                "response_schema=off still sent response_format: "
                f"keys={sorted(payload)}"
            )
            stub = _Stub(_completion(json.dumps(ANSWER)))
            engine._post_completion = stub
            result = engine.execute("say ok")
        assert result.get("status") == "completed", result
        assert result.get("kind") == "answer", result
        assert stub.calls, "stub was not invoked"
        sent = stub.calls[0]["payload"]
        assert "response_format" not in sent, sent


def check_jinja_detection():
    """Thinking-off + <think> in the reply is a loud --jinja failure."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        leaked = (
            "<think>secret chain of thought</think>\n"
            + json.dumps(ANSWER)
        )
        stub = _Stub(_completion(leaked, finish_reason="stop"))
        engine._post_completion = stub
        try:
            engine._call_model("say ok")
            raise AssertionError(
                "ignored chat_template_kwargs did not fail loudly"
            )
        except EngineError as exc:
            err = str(exc)
        assert "--jinja" in err, err
        assert "enable_thinking" in err, err

        engine2 = _engine(Path(tmp))
        engine2._post_completion = stub
        result = engine2.execute("say ok")
        assert result.get("status") == "failed", result
        assert "--jinja" in str(result.get("error") or ""), result


CHECKS = [
    ("payload_shape", check_payload_shape),
    ("length_failure_diagnosable", check_length_failure_is_diagnosable),
    ("telemetry_reaches_receipt", check_telemetry_reaches_receipt),
    ("degradation_path", check_degradation_path),
    ("jinja_detection", check_jinja_detection),
]


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"ok {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    return 1 if failed else 0


def test_hcli_callpath():
    """pytest entry: the same checks as running this file directly."""
    rc = main()
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
