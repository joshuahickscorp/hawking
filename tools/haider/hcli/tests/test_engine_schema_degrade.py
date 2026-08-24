"""Engine structured-output degradation: never silently drop schema.

When the backend cannot enforce response_format the engine must not send
the field, must validate against HCLI_RESULT_SCHEMA, retry a bounded
number of times on SchemaViolation, and write a receipt that cannot be
mistaken for the llama.cpp enforced path.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]

from hcli.backends import StructuredOutputExhausted
from hcli.config import Config
from hcli.engine import Engine, HCLI_RESULT_SCHEMA
from hcli.events import EventBus
from hcli.workspace import Workspace


VALID = {
    "kind": "answer",
    "content": "ok",
    "operations": [],
    "tests": [],
}


def _openai(text, finish_reason="stop", prompt_tokens=9, completion_tokens=4):
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": text},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


class _Backend:
    def __init__(self, response_format=False, grammar=False):
        self._rf = bool(response_format)
        self._grammar = bool(grammar)

    def supports(self, feature):
        if feature in {
            "response_format",
            "json_schema",
            "response_format_json_schema",
        }:
            return self._rf
        if feature in {"grammar", "grammar_gbnf"}:
            return self._grammar
        return True


def _engine(root: Path, backend=None) -> Engine:
    cfg = Config(str(root), global_path=str(root / "global-config.json"))
    runtime = type("Runtime", (), {})()
    runtime.index = 0
    runtime.pid = 1
    runtime.port = 18765
    runtime.active = True
    runtime.backend = backend
    pool = type("Pool", (), {})()
    pool.runtimes = [runtime]
    return Engine(
        workspace=Workspace(str(root)),
        event_bus=EventBus(),
        runtime_provider=lambda: pool,
        runtime_state_provider=lambda: pool,
        runtime_count=1,
        model_name="local",
        config=cfg,
    )


def _receipt(result):
    return json.loads(Path(result["receipt"]).read_text(encoding="utf-8"))


class TestEngineSchemaDegrade(unittest.TestCase):
    def test_degraded_path_does_not_send_response_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp), backend=_Backend())
            captured = []

            def stub(endpoint, payload, timeout):
                captured.append(payload)
                return _openai(json.dumps(VALID))

            engine._post_completion = stub
            parsed = engine._call_model("say ok")
        self.assertEqual(parsed["kind"], "answer")
        self.assertEqual(len(captured), 1)
        sent = captured[0]
        self.assertNotIn("response_format", sent)
        self.assertNotIn("grammar", sent)
        user = sent["messages"][-1]["content"]
        self.assertIn("MUST satisfy this JSON Schema", user)
        self.assertIn('"kind"', user)
        so = engine._last_call_plan["structured_output"]
        self.assertEqual(so["mode"], "degraded")
        self.assertFalse(so["response_format_sent"])
        self.assertEqual(so["attempts"], 1)
        self.assertEqual(so["retries"], 0)
        self.assertFalse(so["exhausted"])

    def test_schema_violation_is_rejected_and_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp), backend=_Backend())
            captured = []
            replies = [
                _openai("this is not json at all {"),
                _openai(json.dumps(VALID)),
            ]

            def stub(endpoint, payload, timeout):
                captured.append(payload)
                return replies[len(captured) - 1]

            engine._post_completion = stub
            result = engine.execute("say ok")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(captured), 2)
            self.assertNotIn("response_format", captured[0])
            self.assertNotIn("response_format", captured[1])
            self.assertIn("MUST satisfy this JSON Schema", captured[0]["messages"][-1]["content"])
            self.assertIn("Attempt 1 was rejected", captured[1]["messages"][-1]["content"])
            receipt = _receipt(result)
            so = receipt["structured_output"]
            self.assertEqual(so["mode"], "degraded")
            self.assertFalse(so["response_format_sent"])
            self.assertEqual(so["attempts"], 2)
            self.assertEqual(so["retries"], 1)
            self.assertFalse(so["exhausted"])
            self.assertEqual(len(receipt["model_calls"]), 2)

    def test_extra_properties_are_rejected_not_sanitized_away(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp), backend=_Backend())
            extra = dict(VALID)
            extra["surprise"] = True
            captured = []
            replies = [
                _openai(json.dumps(extra)),
                _openai(json.dumps(VALID)),
            ]

            def stub(endpoint, payload, timeout):
                captured.append(payload)
                return replies[len(captured) - 1]

            engine._post_completion = stub
            parsed = engine._call_model("say ok")
        self.assertEqual(len(captured), 2)
        self.assertEqual(parsed, VALID)
        self.assertIn("additional property", captured[1]["messages"][-1]["content"])

    def test_retry_exhaustion_is_explicit_failure_with_last_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp), backend=_Backend())
            malformed = "I decline to emit JSON. Here is a poem instead."

            def stub(endpoint, payload, timeout):
                return _openai(malformed)

            engine._post_completion = stub
            with patch.dict(
                os.environ, {"HCLI_STRUCTURED_OUTPUT_ATTEMPTS": "2"}, clear=False
            ):
                result = engine.execute("say ok")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_type"], "StructuredOutputExhausted")
            self.assertIn("rejected after 2 attempts", result["error"])
            self.assertIn("JSON object", result["error"])
            receipt = _receipt(result)
            so = receipt["structured_output"]
            self.assertEqual(so["mode"], "degraded")
            self.assertFalse(so["response_format_sent"])
            self.assertTrue(so["exhausted"])
            self.assertEqual(so["attempts"], 2)
            self.assertEqual(so["retries"], 1)
            self.assertEqual(so["max_attempts"], 2)
            self.assertTrue(so["last_violation"])
            self.assertIn(so["last_violation"], result["error"])
            self.assertEqual(len(receipt["model_calls"]), 2)

    def test_retry_is_bounded_not_unbounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp), backend=_Backend())
            calls = {"n": 0}

            def stub(endpoint, payload, timeout):
                calls["n"] += 1
                if calls["n"] > 8:
                    raise AssertionError("unbounded structured-output retry")
                return _openai("not json")

            engine._post_completion = stub
            with patch.dict(
                os.environ, {"HCLI_STRUCTURED_OUTPUT_ATTEMPTS": "3"}, clear=False
            ):
                with self.assertRaises(StructuredOutputExhausted) as ctx:
                    engine._call_model("say ok")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(ctx.exception.attempts, 3)
        self.assertEqual(len(ctx.exception.errors), 3)

    def test_receipt_enforced_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp), backend=_Backend(response_format=True))
            captured = []

            def stub(endpoint, payload, timeout):
                captured.append(payload)
                return _openai(json.dumps(VALID))

            engine._post_completion = stub
            result = engine.execute("say ok")
            self.assertEqual(result["status"], "completed")
            self.assertIn("response_format", captured[0])
            schema = captured[0]["response_format"]["json_schema"]["schema"]
            self.assertEqual(
                set(schema["required"]), set(HCLI_RESULT_SCHEMA["required"])
            )
            receipt = _receipt(result)
            so = receipt["structured_output"]
            self.assertEqual(so, {"mode": "enforced", "response_format_sent": True})
            self.assertNotIn("attempts", so)
            self.assertNotIn("retries", so)
            self.assertNotIn("exhausted", so)

    def test_receipt_degraded_shape_distinct_from_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp), backend=_Backend())

            def stub(endpoint, payload, timeout):
                return _openai(json.dumps(VALID))

            engine._post_completion = stub
            result = engine.execute("say ok")
            so = _receipt(result)["structured_output"]
            self.assertEqual(so["mode"], "degraded")
            self.assertFalse(so["response_format_sent"])
            self.assertIn("attempts", so)
            self.assertIn("retries", so)
            self.assertIn("max_attempts", so)
            self.assertFalse(so["exhausted"])

    def test_enable_thinking_false_on_degraded_unless_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp), backend=_Backend())
            engine.config.save_project({"enable_thinking": True})
            captured = []

            def stub(endpoint, payload, timeout):
                captured.append(payload)
                return _openai(json.dumps(VALID))

            engine._post_completion = stub
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HCLI_ENABLE_THINKING", None)
                engine._call_model("say ok")
                engine._call_model("say ok", enable_thinking=True)
        self.assertFalse(
            captured[0]["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertTrue(
            captured[1]["chat_template_kwargs"]["enable_thinking"]
        )

    def test_no_backend_stays_on_enforced_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp), backend=None)
            captured = []

            def stub(endpoint, payload, timeout):
                captured.append(payload)
                return _openai(json.dumps(VALID))

            engine._post_completion = stub
            engine._call_model("say ok")
        self.assertIn("response_format", captured[0])
        self.assertEqual(
            engine._last_call_plan["structured_output"]["mode"], "enforced"
        )


if __name__ == "__main__":
    unittest.main()
