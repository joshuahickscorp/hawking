"""A reply cut off mid-object must be RETRIED and COUNTED, then bounded.

Measured from a live mission receipt (.hcli/receipts/8039ee5a-*.json):

    structured_output: {"attempts": 0, "max_attempts": 3, "retries": 0,
                        "mode": "degraded", ...}
    final model_call:  {"max_tokens": 6310, "completion_tokens": 2048,
                        "finish_reason": "length"}

attempts=0 against max_attempts=3 is a retry budget that was never spent.
Engine._complete_with_schema_contract raised EngineError on a length
truncation, which escaped StructuredOutputContract.enforce instead of
counting as the schema violation it is -- so the model was never asked
again, and never told to answer more briefly.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hcli.backends import CompletionResult, SchemaViolation, StructuredOutputExhausted
from hcli.backends import StructuredOutputContract, schema_instruction
from hcli.config import Config
from hcli.engine import Engine
from hcli.events import EventBus
from hcli.workspace import Workspace


VALID = {
    "kind": "answer",
    "content": "ok",
    "operations": [],
    "tests": [],
    "tool_calls": [],
}

# What a 2048-token reply against a 6310-token budget actually looks like:
# a real object that stops mid-string with nothing closed.
TRUNCATED = '{"kind": "answer", "content": "the repository contains'


def _openai(text, finish_reason="stop", prompt_tokens=4096, completion_tokens=2048):
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"content": text}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


class _NoConstrainedDecoding:
    """A native-shaped backend: no response_format, no grammar."""

    def supports(self, feature):
        if feature in {
            "response_format",
            "json_schema",
            "response_format_json_schema",
            "grammar",
            "grammar_gbnf",
        }:
            return False
        return True


def _engine(root: Path) -> Engine:
    cfg = Config(str(root), global_path=str(root / "global-config.json"))
    runtime = type("Runtime", (), {})()
    runtime.index = 0
    runtime.pid = 1
    runtime.port = 18765
    runtime.active = True
    runtime.backend = _NoConstrainedDecoding()
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


class TestStructuredOutputTruncationRetry(unittest.TestCase):
    def test_structured_retry_is_local_not_a_full_context_replay(self):
        schema = {
            "type": "object",
            "required": ["kind"],
            "properties": {"kind": {"type": "string"}},
            "additionalProperties": False,
        }
        contract = StructuredOutputContract(
            schema=schema, instruction=schema_instruction(schema), max_attempts=2
        )
        original = {
            "model": "local",
            "messages": [
                {"role": "system", "content": "stable system prefix"},
                {"role": "user", "content": "ODYSSEY EVIDENCE " * 1200},
            ],
            "max_tokens": 2048,
        }
        sent = []

        def complete(payload, _timeout):
            sent.append(payload)
            if len(sent) == 1:
                raise SchemaViolation(
                    "missing required property 'kind'",
                    text='{"content":"rejected scientific fragment"}',
                )
            return CompletionResult(
                raw={"choices": []}, text=json.dumps({"kind": "answer"})
            )

        result = contract.enforce(complete, original)

        self.assertEqual(result.schema_attempts, 2)
        self.assertEqual(sent[1]["messages"][0], sent[0]["messages"][0])
        retry_prompt = sent[1]["messages"][-1]["content"]
        self.assertLess(len(retry_prompt), len(sent[0]["messages"][-1]["content"]) // 4)
        self.assertIn("missing required property 'kind'", retry_prompt)
        self.assertIn("rejected scientific fragment", retry_prompt)
        self.assertLess(sent[1]["max_tokens"], sent[0]["max_tokens"])

    def test_length_truncation_is_retried_and_counted(self):
        sent = []
        replies = [
            _openai(TRUNCATED, finish_reason="length"),
            _openai(json.dumps(VALID)),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp))

            def stub(endpoint, payload, timeout):
                sent.append(payload)
                return replies[len(sent) - 1]

            engine._post_completion = stub
            parsed = engine._call_model("say ok")

        self.assertEqual(parsed, VALID)
        # The retry actually happened.
        self.assertEqual(len(sent), 2)

        so = engine._last_call_plan["structured_output"]
        # The whole point: attempts is no longer 0.
        self.assertGreater(so["attempts"], 0)
        self.assertEqual(so["attempts"], 2)
        self.assertEqual(so["retries"], 1)
        self.assertFalse(so["exhausted"])
        assert len(so["structured_attempt_budgets"]) == 2
        assert so["structured_attempt_budgets"][0]["requested_max_tokens"] > 0
        assert so["structured_attempt_budgets"][1]["completion_tokens"] == 2048

        # The model was TOLD it was truncated and asked for less, not simply
        # asked again. A retry that repeats the same request repeats the same
        # 2048-token ramble.
        retry_prompt = sent[1]["messages"][-1]["content"]
        self.assertIn("never closed the JSON object", retry_prompt)
        self.assertIn("answer far more briefly", retry_prompt)
        self.assertIn("2048 tokens", retry_prompt)

        # And the receipt says plainly why constrained decoding was not used,
        # rather than listing the two features it did not have as if they
        # were features it had.
        self.assertEqual(so["constrained_decoding"], "unavailable")
        self.assertIn("response_format=False", so["constrained_decoding_reason"])
        self.assertIn("grammar=False", so["constrained_decoding_reason"])
        self.assertEqual(
            so["degraded_features"], ["response_format", "grammar"]
        )

    def test_retry_budget_is_bounded_and_fails_hard(self):
        """Never a loop, and never a fake success at the end of one."""
        calls = {"n": 0}

        with tempfile.TemporaryDirectory() as tmp:
            engine = _engine(Path(tmp))

            def stub(endpoint, payload, timeout):
                calls["n"] += 1
                if calls["n"] > 8:
                    raise AssertionError("unbounded truncation retry")
                return _openai(TRUNCATED, finish_reason="length")

            engine._post_completion = stub
            with patch.dict(
                os.environ, {"HCLI_STRUCTURED_OUTPUT_ATTEMPTS": "2"}, clear=False
            ):
                with self.assertRaises(StructuredOutputExhausted) as ctx:
                    engine._call_model("say ok")

        self.assertEqual(calls["n"], 2)
        self.assertEqual(ctx.exception.attempts, 2)
        self.assertEqual(len(ctx.exception.errors), 2)
        # The truncation diagnosis survives to the failure, so the receipt
        # still names the real ceiling instead of a generic parse error.
        self.assertIn("never closed the JSON object", ctx.exception.errors[-1])

        so = engine._last_call_plan["structured_output"]
        self.assertTrue(so["exhausted"])
        self.assertEqual(so["attempts"], 2)
        self.assertEqual(so["max_attempts"], 2)
        self.assertIn("never closed the JSON object", so["last_violation"])


if __name__ == "__main__":
    unittest.main()
