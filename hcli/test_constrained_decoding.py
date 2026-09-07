"""Constrained decoding on the native transport: prove it, don't claim it.

WORK ITEM 1 (settled from code, not the feature list): the native JSONL wire
protocol (HawkingNativeConnector.complete_payload -> ResidentProcess.request)
builds its own request dict from scratch --

    {"id": ..., "prompt": ..., "max_new_tokens": ..., "max_seq_len": ...}

-- and never forwards response_format or grammar even when the caller's
OpenAI-shaped payload carries them. There is no grammar/logit-mask channel in
the wire protocol at all, so NoeticNativeBackend.supports() correctly reports
both False for a profile with no declared capabilities (the live sealed-3.14
profile has none), StructuredOutputContract degrades honestly, and the
receipt already says so (test_structured_output_retry.py covers that half).

WORK ITEM 3/4 (this file): degrading is correct, but three identical-looking
degraded attempts still died on a 4620-token reply the model itself chose to
stop (finish_reason="stop") with a JSON string it never closed -- measured on
.hcli/receipts/10464271-*.json: "Unterminated string starting at: line 3
column 14 (char 37)". Retrying the exact same instruction three times bought
three malformed replies. This file proves:

  (a) a reply whose ONLY defect is an unclosed trailing string is repaired
      deterministically -- no field content invented, no retry spent -- but
      ONLY when the repaired object fully satisfies the schema; a reply that
      breaks early (required fields never reached, the live-receipt shape)
      is correctly left alone rather than silently accepted with the missing
      fields defaulted, which would be exactly the silent pass this whole
      contract exists to prevent.
  (b) the retry that DOES get spent asks for the right thing: a decode-class
      violation (broken JSON syntax) gets "escape your quotes, keep it
      short, close every field" instead of the generic "satisfy the schema"
      instruction, which does not describe what actually went wrong.
"""
from __future__ import annotations

import json
import unittest

from hcli.backends import (
    CompletionResult,
    SchemaViolation,
    StructuredOutputContract,
    StructuredOutputExhausted,
    _close_unterminated_string,
    _is_decode_violation,
    extract_json_object,
    schema_instruction,
)
from hcli.engine import HCLI_RESULT_SCHEMA
from hcli.hawking_native import HawkingNativeConfig, HawkingNativeConnector


VALID = {
    "kind": "answer",
    "content": "ok",
    "operations": [],
    "tests": [],
    "tool_calls": [],
}


# ---------------------------------------------------------------------------
# Work item 1: the wire protocol itself, not the feature list.
# ---------------------------------------------------------------------------

class _FakeResident:
    """Stands in for ResidentProcess: captures the body it was asked to send."""

    def __init__(self, reply_text: str) -> None:
        self.sent = None
        self._reply_text = reply_text

    def request(self, body, timeout):  # noqa: D401 - test double
        self.sent = dict(body)
        return {"generated_text": self._reply_text, "new_token_ids": [0]}

    def health(self):
        return {"ready": True}


def _connector(resident: _FakeResident) -> HawkingNativeConnector:
    connector = HawkingNativeConnector.__new__(HawkingNativeConnector)
    connector.config = HawkingNativeConfig(
        binary="/nonexistent/resident",
        resident_binary="/nonexistent/resident",
        artifact_root="/nonexistent/artifact",
        tokenizer="/nonexistent/tokenizer.json",
        max_seq_len=8192,
        generation={"max_new_tokens": 2048},
        mode="resident",
    )
    connector.resident = resident
    connector.restart_count = 0
    connector._render = lambda payload: type(
        "Rendered", (), {
            "text": "RENDERED PROMPT",
            "prompt_tokens": 10,
            "thinking_requested": False,
            "thinking_qualified": False,
            "token_count_source": "test",
        },
    )()
    return connector


class TestNativeWireProtocolCannotCarryConstrainedDecoding(unittest.TestCase):
    def test_response_format_and_grammar_never_reach_the_resident(self):
        resident = _FakeResident(json.dumps(VALID))
        connector = _connector(resident)
        payload = {
            "messages": [{"role": "user", "content": "go"}],
            "response_format": {"type": "json_schema", "json_schema": HCLI_RESULT_SCHEMA},
            "grammar": "root ::= object",
            "max_tokens": 512,
        }

        connector.complete_payload(payload, timeout=5.0)

        self.assertIsNotNone(resident.sent, "the request never reached the resident")
        self.assertNotIn(
            "response_format", resident.sent,
            "the wire protocol has no field for it, yet response_format reached "
            f"the resident anyway: {resident.sent}",
        )
        self.assertNotIn(
            "grammar", resident.sent,
            f"the wire protocol has no field for it, yet grammar reached the "
            f"resident anyway: {resident.sent}",
        )
        # The protocol is this small on purpose -- prove the whole shape, not
        # just the two missing keys, so a future field added elsewhere can't
        # sneak through unnoticed.
        self.assertEqual(
            set(resident.sent), {"id", "prompt", "max_new_tokens", "max_seq_len"},
        )


# ---------------------------------------------------------------------------
# Work item 3(b): deterministic repair, bounded to when it can't be a silent
# pass.
# ---------------------------------------------------------------------------

class TestDeterministicTruncationRepair(unittest.TestCase):
    def test_short_answer_can_omit_mode_specific_empty_arrays(self):
        contract = StructuredOutputContract(
            schema=HCLI_RESULT_SCHEMA,
            instruction=schema_instruction(HCLI_RESULT_SCHEMA),
        )
        result = contract.validate('{"kind":"answer","content":"ok"}')
        self.assertEqual(result, {"kind": "answer", "content": "ok"})

    def test_trailing_unterminated_string_is_closed_without_a_retry(self):
        """content is the LAST field written; everything else is complete."""
        text = (
            '{"kind": "answer", "operations": [], "tests": [], '
            '"tool_calls": [], "content": "the repo has a src dir and a '
            'lot of history that never'
        )
        diag: list = []
        parsed = extract_json_object(text, diag)

        self.assertEqual(parsed["kind"], "answer")
        self.assertTrue(parsed["content"].startswith("the repo has a src dir"))
        self.assertTrue(
            any(d.startswith("TRUNCATION_REPAIR: ") for d in diag),
            f"the repair happened but was not recorded: {diag}",
        )

        contract = StructuredOutputContract(
            schema=HCLI_RESULT_SCHEMA, instruction=schema_instruction(HCLI_RESULT_SCHEMA)
        )
        result = contract.validate(text)
        self.assertEqual(result["kind"], "answer")
        self.assertEqual(len(contract.truncation_repairs), 1)

    def test_early_break_is_NOT_silently_accepted(self):
        """The measured live-mission shape: content is huge and EARLY, the
        rest of the object (operations/tests/tool_calls) never arrives.

        A repair that defaulted the missing required arrays to [] would
        report success on an object the model never finished -- exactly the
        silent pass StructuredOutputExhausted exists to forbid. This must
        still raise.
        """
        text = (
            '{"kind": "answer", "content": "the repo has a src dir and a '
            'lot of history spanning years of work that never'
        )
        contract = StructuredOutputContract(
            schema=HCLI_RESULT_SCHEMA, instruction=schema_instruction(HCLI_RESULT_SCHEMA)
        )
        with self.assertRaises(SchemaViolation) as ctx:
            contract.validate(text)
        # The failure names BOTH what happened: the syntax was repaired...
        self.assertIn("TRUNCATION_REPAIR", str(ctx.exception))
        # ...and what is still missing, so the retry knows what to add.
        self.assertIn("missing required property", str(ctx.exception))
        # And the repair must not have been credited as used.
        self.assertEqual(contract.truncation_repairs, [])

    def test_close_unterminated_string_only_fires_on_that_exact_error(self):
        """Any other JSONDecodeError must not be treated as recoverable."""
        try:
            json.loads('{"a": 1,}')  # trailing comma -- a real syntax error
        except json.JSONDecodeError as exc:
            self.assertIsNone(_close_unterminated_string('{"a": 1,}', exc))
        else:
            self.fail("expected a JSONDecodeError")


# ---------------------------------------------------------------------------
# Work item 3(a): the retry that IS spent asks for the right thing.
# ---------------------------------------------------------------------------

class TestDecodeAwareRetryPrompt(unittest.TestCase):
    def test_decode_failure_gets_escaping_and_brevity_guidance(self):
        sent = []
        replies = ["not json at all {{{", json.dumps(VALID)]

        def complete_fn(payload, timeout=None):
            sent.append(payload)
            return CompletionResult(raw={}, text=replies[len(sent) - 1], finish_reason="stop")

        contract = StructuredOutputContract(
            schema=HCLI_RESULT_SCHEMA,
            instruction=schema_instruction(HCLI_RESULT_SCHEMA),
            max_attempts=2,
        )
        result = contract.enforce(complete_fn, {"messages": [{"role": "user", "content": "go"}]})

        self.assertEqual(len(sent), 2)
        retry_prompt = sent[1]["messages"][-1]["content"]
        self.assertIn("broke JSON syntax", retry_prompt)
        self.assertIn("Escape every quote", retry_prompt)
        self.assertIn("kind, content", retry_prompt)
        self.assertNotIn(
            "Return exactly one JSON object that satisfies the "
            "schema and nothing else",
            retry_prompt,
        )
        self.assertEqual(result.text, json.dumps(VALID))

    def test_schema_shape_failure_keeps_the_generic_message(self):
        """A valid-JSON-but-wrong-shape reply is a different failure class."""
        sent = []
        replies = [json.dumps({"kind": "answer"}), json.dumps(VALID)]

        def complete_fn(payload, timeout=None):
            sent.append(payload)
            return CompletionResult(raw={}, text=replies[len(sent) - 1], finish_reason="stop")

        contract = StructuredOutputContract(
            schema=HCLI_RESULT_SCHEMA,
            instruction=schema_instruction(HCLI_RESULT_SCHEMA),
            max_attempts=2,
        )
        contract.enforce(complete_fn, {"messages": [{"role": "user", "content": "go"}]})

        retry_prompt = sent[1]["messages"][-1]["content"]
        self.assertIn(
            "Return exactly one JSON object that satisfies the "
            "schema and nothing else",
            retry_prompt,
        )
        self.assertNotIn("broke JSON syntax", retry_prompt)

    def test_exhausting_all_attempts_on_the_measured_failure_still_raises(self):
        """The live-mission failure, replayed 3 times, must still fail hard."""
        text = (
            '{"kind": "answer", "content": "the repo has a src dir and a '
            'lot of history spanning years of work that never'
        )

        def complete_fn(payload, timeout=None):
            return CompletionResult(raw={}, text=text, finish_reason="stop")

        contract = StructuredOutputContract(
            schema=HCLI_RESULT_SCHEMA,
            instruction=schema_instruction(HCLI_RESULT_SCHEMA),
            max_attempts=3,
        )
        with self.assertRaises(StructuredOutputExhausted) as ctx:
            contract.enforce(complete_fn, {"messages": [{"role": "user", "content": "go"}]})
        self.assertEqual(ctx.exception.attempts, 3)
        self.assertIn("TRUNCATION_REPAIR", ctx.exception.errors[-1])


class TestProviderNativeToolValues(unittest.TestCase):
    def test_native_boolean_tool_value_reaches_typed_argument_boundary(self):
        """Claude-shaped JSON must not burn retries on ``true`` vs ``"true"``."""
        reply = json.dumps({
            "kind": "tool_use",
            "content": "inspect",
            "operations": [],
            "tests": [],
            "tool_calls": [{
                "tool": "fs.list",
                "arguments": [
                    {"name": "path", "value": "hcli"},
                    {"name": "recursive", "value": True},
                ],
            }],
        })
        contract = StructuredOutputContract(
            schema=HCLI_RESULT_SCHEMA,
            instruction=schema_instruction(HCLI_RESULT_SCHEMA),
            max_attempts=3,
        )
        result = contract.enforce(
            lambda _payload, _timeout=None: CompletionResult(
                raw={}, text=reply, finish_reason="stop"
            ),
            {"messages": [{"role": "user", "content": "go"}]},
        )

        assert result.schema_attempts == 1
        assert result.raw["_structured"]["tool_calls"][0]["arguments"][1]["value"] == "true"
        assert result.raw["_structured_value_repairs"]
        assert "structured_output_value_repair" in result.degraded


class TestIsDecodeViolation(unittest.TestCase):
    def test_classifies_the_measured_live_error_as_decode(self):
        self.assertTrue(_is_decode_violation(
            "response is not a JSON object (outermost object failed to "
            "decode: Unterminated string starting at: line 3 column 14 "
            "(char 37))"
        ))

    def test_does_not_classify_a_missing_field_as_decode(self):
        self.assertFalse(_is_decode_violation("$: missing required property 'operations'"))


if __name__ == "__main__":
    unittest.main()
