from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]

from hcli.config import Config, coerce_bool, coerce_on_off
from hcli.engine import (
    Engine,
    EngineError,
    HCLI_RESULT_SCHEMA,
    _MAX_TOKENS_CEILING,
    _MAX_TOKENS_FLOOR,
)
from hcli.events import EventBus
from hcli.workspace import Workspace


def _engine(root: Path) -> Engine:
    cfg = Config(str(root), global_path=str(root / "global-config.json"))
    runtime = type("Runtime", (), {})()
    runtime.index = 0
    runtime.pid = 1
    runtime.port = 18765
    runtime.active = True
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


class TestCallPolicyConfig(unittest.TestCase):
    def test_coerce_bool(self):
        self.assertFalse(coerce_bool(None, default=False))
        self.assertTrue(coerce_bool(None, default=True))
        self.assertTrue(coerce_bool("true"))
        self.assertTrue(coerce_bool("ON"))
        self.assertFalse(coerce_bool("false"))
        self.assertFalse(coerce_bool("0"))
        self.assertTrue(coerce_bool(True))
        self.assertFalse(coerce_bool(False))

    def test_coerce_on_off(self):
        self.assertEqual(coerce_on_off(None), "on")
        self.assertEqual(coerce_on_off("off"), "off")
        self.assertEqual(coerce_on_off("0"), "off")
        self.assertEqual(coerce_on_off(False), "off")
        self.assertEqual(coerce_on_off(True), "on")

    def test_env_wins_over_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(tmp, global_path=str(Path(tmp) / "g.json"))
            cfg.save_project(
                {"enable_thinking": False, "response_schema": "on"}
            )
            with patch.dict(
                os.environ,
                {
                    "HCLI_ENABLE_THINKING": "true",
                    "HCLI_RESPONSE_SCHEMA": "off",
                },
                clear=False,
            ):
                self.assertTrue(cfg.enable_thinking(default=False))
                self.assertFalse(cfg.response_schema_on(default=True))

    def test_model_tokens_source_env_config_derived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = _engine(root)
            with patch.dict(os.environ, {"HCLI_MODEL_TOKENS": "1234"}, clear=False):
                n, source = engine.config.model_tokens()
                self.assertEqual(n, 1234)
                self.assertEqual(source, "env")
                payload = engine._build_model_payload("x")
                self.assertEqual(payload["max_tokens"], 1234)
                self.assertEqual(
                    engine._last_call_plan["max_tokens_source"], "env"
                )

            env = os.environ.copy()
            env.pop("HCLI_MODEL_TOKENS", None)
            with patch.dict(os.environ, env, clear=True):
                # re-apply nothing; isolated Config file still empty
                engine.config.save_project({"model_tokens": 2222})
                payload = engine._build_model_payload("x")
                self.assertEqual(payload["max_tokens"], 2222)
                self.assertEqual(
                    engine._last_call_plan["max_tokens_source"], "config"
                )

                engine.config.save_project({})
                (root / ".hcli" / "config.json").write_text("{}", encoding="utf-8")
                with patch.dict(
                    os.environ, {"HCLI_CTX_SIZE": "4096"}, clear=False
                ):
                    os.environ.pop("HCLI_MODEL_TOKENS", None)
                    payload = engine._build_model_payload("x")
                self.assertEqual(
                    engine._last_call_plan["max_tokens_source"], "derived"
                )
                self.assertGreaterEqual(
                    payload["max_tokens"], _MAX_TOKENS_FLOOR
                )
                self.assertLessEqual(
                    payload["max_tokens"], _MAX_TOKENS_CEILING
                )


class TestCallPath(unittest.TestCase):
    def test_default_payload_is_structured(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HCLI_ENABLE_THINKING", None)
                os.environ.pop("HCLI_RESPONSE_SCHEMA", None)
                engine = _engine(Path(tmp))
                payload = engine._build_model_payload("say ok")
            self.assertFalse(
                payload["chat_template_kwargs"]["enable_thinking"]
            )
            required = payload["response_format"]["json_schema"]["schema"][
                "required"
            ]
            self.assertEqual(
                set(required),
                set(HCLI_RESULT_SCHEMA["required"]),
            )

    def test_per_call_override_re_enables_thinking(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp))
            payload = engine._build_model_payload(
                "plan",
                enable_thinking=True,
                response_schema=False,
            )
            self.assertTrue(
                payload["chat_template_kwargs"]["enable_thinking"]
            )
            self.assertNotIn("response_format", payload)

    def test_length_error_names_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp))

            def stub(endpoint, payload, timeout):
                return {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "content": '{"kind": "answer", "content": "'
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 400,
                        "completion_tokens": payload["max_tokens"],
                    },
                }

            engine._post_completion = stub
            with patch.dict(
                os.environ, {"HCLI_MODEL_TOKENS": "6500"}, clear=False
            ):
                with self.assertRaises(EngineError) as ctx:
                    engine._call_model("mission")
            msg = str(ctx.exception)
            self.assertIn("6500-token", msg)
            self.assertIn("400-token", msg)

    def test_receipt_model_calls_and_error_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp))

            def stub(endpoint, payload, timeout):
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "kind": "answer",
                                        "content": "ok",
                                        "operations": [],
                                        "tests": [],
                                    }
                                )
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 4,
                    },
                }

            engine._post_completion = stub
            result = engine.execute("say ok")
            self.assertEqual(result["status"], "completed")
            receipt = json.loads(
                Path(result["receipt"]).read_text(encoding="utf-8")
            )
            self.assertNotIn("error", receipt)
            call = receipt["model_calls"][0]
            self.assertEqual(call["finish_reason"], "stop")
            self.assertEqual(call["prompt_tokens"], 9)
            self.assertEqual(call["completion_tokens"], 4)
            self.assertIn("endpoint", call)
            self.assertIn("wall_s", call)

    def test_jinja_leak_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp))

            def stub(endpoint, payload, timeout):
                return {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": (
                                    "<think>nope</think>"
                                    + json.dumps(
                                        {
                                            "kind": "answer",
                                            "content": "ok",
                                            "operations": [],
                                            "tests": [],
                                        }
                                    )
                                )
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 4,
                    },
                }

            engine._post_completion = stub
            with self.assertRaises(EngineError) as ctx:
                engine._call_model("say ok")
            self.assertIn("--jinja", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
