from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]

from hcli.backends import (
    CompletionResult,
    MlxServerBackend,
    SchemaViolation,
    StructuredOutputExhausted,
    extract_json_object,
    is_mlx_model_dir,
    make_structured_output_contract,
    mlx_quantisation_label,
    mlx_server_binary,
    schema_from_response_format,
    schema_instruction,
    validate_against_schema,
)


MINI_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["answer", "mutation"]},
        "content": {"type": "string"},
        "operations": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["op", "path"],
                "additionalProperties": False,
            },
        },
        "tests": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kind", "content", "operations", "tests"],
    "additionalProperties": False,
}

VALID_OBJECT = {
    "kind": "answer",
    "content": "ok",
    "operations": [],
    "tests": [],
}

MLX_HELP = """
usage: mlx_lm.server [-h] [--model MODEL] [--host HOST] [--port PORT]
                     [--chat-template-args CHAT_TEMPLATE_ARGS]
                     [--decode-concurrency DECODE_CONCURRENCY]
                     [--prompt-concurrency PROMPT_CONCURRENCY]
                     [--prefill-step-size PREFILL_STEP_SIZE]
                     [--prompt-cache-size PROMPT_CACHE_SIZE]
                     [--prompt-cache-bytes PROMPT_CACHE_BYTES]
"""


def _fake_mlx_dir(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                },
                "text_config": {"max_position_embeddings": 262144},
            }
        ),
        encoding="utf-8",
    )
    (root / "model.safetensors").write_bytes(b"x" * 128)
    return str(root)


class _FakeMlxHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return

    def do_GET(self):
        if self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            body = {}
        self.server.requests.append({"path": self.path, "body": body})
        text = self.server.reply_text
        payload = {
            "id": "cmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
            },
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class _FakeMlxServer(ThreadingHTTPServer):
    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.requests = []
        self.reply_text = "ok"


def _start_fake_server() -> _FakeMlxServer:
    server = _FakeMlxServer(("127.0.0.1", 0), _FakeMlxHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.thread = thread
    return server


class TestMlxCommandAndIdentity(unittest.TestCase):
    def test_command_uses_real_mlx_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _fake_mlx_dir(Path(tmp) / "4bit")
            backend = MlxServerBackend(
                model_path=model,
                port=8099,
                n_slots=2,
                binary="mlx_lm.server",
                max_tokens=512,
            )
            cmd = backend.command()
        self.assertEqual(cmd[0], "mlx_lm.server")
        self.assertIn("--model", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], os.path.realpath(model))
        self.assertIn("--host", cmd)
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")
        self.assertIn("--port", cmd)
        self.assertEqual(cmd[cmd.index("--port") + 1], "8099")
        self.assertIn("--max-tokens", cmd)
        self.assertIn("--decode-concurrency", cmd)
        self.assertEqual(cmd[cmd.index("--decode-concurrency") + 1], "2")
        self.assertIn("--prompt-concurrency", cmd)
        self.assertIn("--prompt-cache-size", cmd)
        self.assertIn("--chat-template-args", cmd)
        args = json.loads(cmd[cmd.index("--chat-template-args") + 1])
        self.assertEqual(args.get("enable_thinking"), False)
        for banned in ("--json-schema", "--grammar", "--jinja", "--ctx-size"):
            self.assertNotIn(banned, cmd)

    def test_identity_names_the_4bit_affine_g64_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _fake_mlx_dir(Path(tmp) / "4bit")
            backend = MlxServerBackend(
                model_path=model, port=9, binary="mlx_lm.server"
            )
            ident = backend.identity()
            self.assertEqual(ident["backend"], "mlx_lm_server")
            self.assertEqual(ident["quantisation"], "4bit-affine-g64")
            self.assertEqual(ident["context"], 262144)
            self.assertEqual(ident["n_slots"], 1)
            self.assertEqual(ident["decode_concurrency"], 1)
            self.assertIn(
                "does not reserve llama.cpp-style slots", ident["slots_note"]
            )
            self.assertIn("4bit-affine-g64", ident["model_identity"])
            self.assertGreater(ident["model_bytes"], 0)
            self.assertEqual(mlx_quantisation_label(model), "4bit-affine-g64")
            self.assertTrue(is_mlx_model_dir(model))
            self.assertFalse(is_mlx_model_dir(str(Path(tmp) / "missing")))


class TestMlxSupports(unittest.TestCase):
    def test_supports_follows_help_text_not_the_comment_table(self):
        backend = MlxServerBackend(
            model_path="/no-model", port=1, binary="mlx_lm.server"
        )
        with patch(
            "hcli.backends.mlx_help_text", return_value=MLX_HELP
        ):
            self.assertFalse(backend.supports("response_format"))
            self.assertFalse(backend.supports("json_schema"))
            self.assertFalse(backend.supports("response_format_json_schema"))
            self.assertFalse(backend.supports("grammar"))
            self.assertFalse(backend.supports("grammar_gbnf"))
            self.assertTrue(backend.supports("chat_template_kwargs"))
            self.assertTrue(
                backend.supports("chat_template_kwargs_enable_thinking")
            )
            self.assertTrue(backend.supports("prefix_cache"))
            self.assertTrue(backend.supports("prompt_prefix_cache"))
            self.assertTrue(backend.supports("slots"))
            self.assertTrue(backend.supports("continuous_batching_slots"))

    def test_supports_false_when_flags_absent(self):
        backend = MlxServerBackend(
            model_path="/no-model", port=1, binary="mlx_lm.server"
        )
        with patch(
            "hcli.backends.mlx_help_text",
            return_value="usage: mlx_lm.server [-h]",
        ):
            self.assertFalse(backend.supports("response_format"))
            self.assertFalse(backend.supports("grammar"))
            self.assertFalse(backend.supports("chat_template_kwargs"))
            self.assertFalse(backend.supports("prefix_cache"))
            self.assertFalse(backend.supports("slots"))

    def test_supports_against_installed_binary(self):
        try:
            binary = mlx_server_binary()
        except RuntimeError:
            self.skipTest("mlx_lm.server not on PATH")
        backend = MlxServerBackend(
            model_path="/no-model", port=1, binary=binary
        )
        self.assertFalse(backend.supports("response_format"))
        self.assertFalse(backend.supports("grammar"))
        # Skip only when the PROBE failed, never when the feature answer is
        # inconvenient. Under a sandbox `--help` dies in Metal detection and
        # returns a crash string; the previous skip read that as "the server
        # does not advertise chat_template_kwargs", which is false on this box
        # -- `--chat-template-args` is in --help, and a live request with
        # `enable_thinking: false` measurably changes the answer.
        if not backend.help_probe_usable():
            self.skipTest("mlx_lm.server --help did not answer in this environment")
        self.assertTrue(backend.supports("chat_template_kwargs"))
        self.assertTrue(backend.supports("prefix_cache"))
        self.assertTrue(backend.supports("slots"))


class TestMlxCompleteStripsUnsupportedFields(unittest.TestCase):
    def setUp(self):
        self.server = _start_fake_server()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)

    def test_complete_never_sends_response_format_or_grammar(self):
        self.server.reply_text = json.dumps(VALID_OBJECT)
        with tempfile.TemporaryDirectory() as tmp:
            model = _fake_mlx_dir(Path(tmp) / "4bit")
            backend = MlxServerBackend(
                model_path=model,
                port=self.server.server_address[1],
                binary="mlx_lm.server",
            )
            with patch(
                "hcli.backends.mlx_help_text", return_value=MLX_HELP
            ):
                result = backend.complete(
                    {
                        "model": "local",
                        "messages": [
                            {"role": "user", "content": "reply json"}
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "hcli_result",
                                "strict": True,
                                "schema": MINI_SCHEMA,
                            },
                        },
                        "grammar": "root ::= object",
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                )
        self.assertEqual(len(self.server.requests), 1)
        sent = self.server.requests[0]["body"]
        self.assertEqual(
            self.server.requests[0]["path"], "/v1/chat/completions"
        )
        self.assertNotIn("response_format", sent)
        self.assertNotIn("grammar", sent)
        self.assertEqual(
            sent.get("chat_template_kwargs"), {"enable_thinking": False}
        )
        self.assertIn("response_format", result.degraded)
        self.assertIn("grammar", result.degraded)
        self.assertIn("JSON Schema", sent["messages"][-1]["content"])
        self.assertEqual(sent.get("model"), "default_model")
        self.assertEqual(result.text, json.dumps(VALID_OBJECT))
        self.assertEqual(result.finish_reason, "stop")

    def test_complete_rewrites_engine_local_model_name(self):
        self.server.reply_text = "ok"
        backend = MlxServerBackend(
            model_path="/models/qwen/4bit",
            port=self.server.server_address[1],
            binary="mlx_lm.server",
        )
        with patch(
            "hcli.backends.mlx_help_text", return_value=MLX_HELP
        ):
            backend.complete(
                {
                    "model": "local",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 8,
                }
            )
        self.assertEqual(self.server.requests[0]["body"]["model"], "default_model")

    def test_ready_probes_health(self):
        backend = MlxServerBackend(
            model_path="/no-model",
            port=self.server.server_address[1],
            binary="mlx_lm.server",
        )
        self.assertTrue(backend.ready(timeout=2))


class TestStructuredOutputContract(unittest.TestCase):
    def test_schema_from_response_format(self):
        schema = schema_from_response_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "hcli_result",
                    "strict": True,
                    "schema": MINI_SCHEMA,
                },
            }
        )
        self.assertEqual(schema, MINI_SCHEMA)
        self.assertEqual(
            schema_from_response_format({"type": "json_object"}),
            {"type": "object"},
        )

    def test_instruction_names_the_schema(self):
        text = schema_instruction(MINI_SCHEMA)
        self.assertIn("JSON object", text)
        self.assertIn('"kind"', text)
        self.assertIn("rejected", text.lower())

    def test_validate_accepts_satisfying_object(self):
        backend = MlxServerBackend(
            model_path="/no-model", port=1, binary="mlx_lm.server"
        )
        with patch(
            "hcli.backends.mlx_help_text", return_value=MLX_HELP
        ):
            contract = make_structured_output_contract(backend, MINI_SCHEMA)
        self.assertIsNotNone(contract)
        parsed = contract.validate(json.dumps(VALID_OBJECT))
        self.assertEqual(parsed, VALID_OBJECT)
        wrapped = (
            "<think>scratch</think>\n```json\n"
            + json.dumps(VALID_OBJECT)
            + "\n```"
        )
        self.assertEqual(contract.validate(wrapped), VALID_OBJECT)

    def test_validate_rejects_malformed_and_schema_invalid(self):
        backend = MlxServerBackend(
            model_path="/no-model", port=1, binary="mlx_lm.server"
        )
        with patch(
            "hcli.backends.mlx_help_text", return_value=MLX_HELP
        ):
            contract = make_structured_output_contract(backend, MINI_SCHEMA)
        with self.assertRaises(SchemaViolation) as malformed:
            contract.validate("this is not json at all {")
        self.assertIn("JSON object", str(malformed.exception))
        missing = json.dumps({"kind": "answer", "content": "ok"})
        with self.assertRaises(SchemaViolation) as missing_fields:
            contract.validate(missing)
        self.assertIn("missing required property", missing_fields.exception.reason)
        extra = json.dumps(
            {
                "kind": "answer",
                "content": "ok",
                "operations": [],
                "tests": [],
                "surprise": True,
            }
        )
        with self.assertRaises(SchemaViolation) as extra_prop:
            contract.validate(extra)
        self.assertIn("additional property", extra_prop.exception.reason)
        bad_enum = json.dumps(
            {
                "kind": "poem",
                "content": "ok",
                "operations": [],
                "tests": [],
            }
        )
        with self.assertRaises(SchemaViolation) as enum_err:
            contract.validate(bad_enum)
        self.assertIn("not one of", enum_err.exception.reason)

    def test_validate_against_schema_direct(self):
        self.assertIsNone(validate_against_schema(VALID_OBJECT, MINI_SCHEMA))
        self.assertIsNotNone(
            validate_against_schema({"kind": "answer"}, MINI_SCHEMA)
        )
        with self.assertRaises(SchemaViolation):
            extract_json_object("")

    def test_native_backend_returns_no_contract(self):
        class _Native:
            def supports(self, feature):
                return True

        self.assertIsNone(
            make_structured_output_contract(_Native(), MINI_SCHEMA)
        )

    def test_apply_strips_response_format_and_injects_instruction(self):
        backend = MlxServerBackend(
            model_path="/no-model", port=1, binary="mlx_lm.server"
        )
        with patch(
            "hcli.backends.mlx_help_text", return_value=MLX_HELP
        ):
            contract = make_structured_output_contract(backend, MINI_SCHEMA)
        payload = {
            "messages": [{"role": "user", "content": "do the thing"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": MINI_SCHEMA},
            },
            "grammar": "root ::= object",
        }
        applied = contract.apply(payload)
        self.assertNotIn("response_format", applied)
        self.assertNotIn("grammar", applied)
        self.assertIn("JSON Schema", applied["messages"][-1]["content"])
        self.assertIn("do the thing", applied["messages"][-1]["content"])

    def test_enforce_retries_then_accepts(self):
        backend = MlxServerBackend(
            model_path="/no-model", port=1, binary="mlx_lm.server"
        )
        with patch(
            "hcli.backends.mlx_help_text", return_value=MLX_HELP
        ):
            contract = make_structured_output_contract(
                backend, MINI_SCHEMA, max_attempts=3
            )
        replies = [
            CompletionResult(raw={}, text="not json {"),
            CompletionResult(
                raw={},
                text=json.dumps({"kind": "answer", "content": "still broken"}),
            ),
            CompletionResult(raw={"ok": True}, text=json.dumps(VALID_OBJECT)),
        ]
        seen = []

        def complete_fn(payload, timeout=None):
            seen.append(payload)
            return replies[len(seen) - 1]

        result = contract.enforce(
            complete_fn,
            {"messages": [{"role": "user", "content": "go"}]},
        )
        self.assertEqual(len(seen), 3)
        self.assertEqual(json.loads(result.text), VALID_OBJECT)
        self.assertIn("response_format", result.degraded)
        self.assertIn("structured_output_prompt_validation", result.degraded)
        self.assertIn("Attempt 1 was rejected", seen[1]["messages"][-1]["content"])
        self.assertEqual(result.raw["_structured"], VALID_OBJECT)

    def test_enforce_exhaustion_is_explicit_failure(self):
        backend = MlxServerBackend(
            model_path="/no-model", port=1, binary="mlx_lm.server"
        )
        with patch(
            "hcli.backends.mlx_help_text", return_value=MLX_HELP
        ):
            contract = make_structured_output_contract(
                backend, MINI_SCHEMA, max_attempts=2
            )
        malformed = "I decline to emit JSON. Here is a poem instead."

        def complete_fn(payload, timeout=None):
            return CompletionResult(raw={"choices": []}, text=malformed)

        with self.assertRaises(StructuredOutputExhausted) as ctx:
            contract.enforce(
                complete_fn,
                {"messages": [{"role": "user", "content": "go"}]},
            )
        err = ctx.exception
        self.assertEqual(err.attempts, 2)
        self.assertEqual(err.last_text, malformed)
        self.assertEqual(len(err.errors), 2)
        self.assertIn("rejected after 2 attempts", err.reason)
        self.assertIn(err.reason, str(err))
        self.assertTrue(err.reason)
        self.assertNotEqual(err.reason, "")


if __name__ == "__main__":
    unittest.main()
