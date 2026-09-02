"""Native JSONL grammar channel for response_format.type == json_object.

The resident JSON constraint is well-formed JSON only (no schema keys,
types, or enums). This file pins the Python request shape: send
grammar="json" if and only if the OpenAI-shaped payload asks for a
json_object, and surface the resident's grammar_enforced flag.
"""
from __future__ import annotations

from hcli.hawking_native import HawkingNativeConfig, HawkingNativeConnector


class _FakeResident:
    def __init__(self, reply_text: str = "{}") -> None:
        self.sent = None
        self._reply_text = reply_text

    def request(self, body, timeout):  # noqa: ARG002 - test double
        self.sent = dict(body)
        return {
            "generated_text": self._reply_text,
            "new_token_ids": [0],
            "grammar_enforced": body.get("grammar") == "json",
        }

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
        "Rendered",
        (),
        {
            "text": "RENDERED PROMPT",
            "prompt_tokens": 10,
            "thinking_requested": False,
            "thinking_qualified": False,
            "token_count_source": "test",
        },
    )()
    return connector


def test_json_object_response_format_sends_grammar_json():
    resident = _FakeResident()
    connector = _connector(resident)
    result = connector.complete_payload(
        {
            "messages": [{"role": "user", "content": "go"}],
            "response_format": {"type": "json_object"},
            "max_tokens": 512,
        },
        timeout=5.0,
    )
    assert resident.sent is not None
    assert resident.sent["grammar"] == "json"
    assert result["hawking"]["grammar_enforced"] is True


def test_grammar_is_omitted_without_json_object_response_format():
    resident = _FakeResident()
    connector = _connector(resident)
    result = connector.complete_payload(
        {
            "messages": [{"role": "user", "content": "go"}],
            "max_tokens": 512,
        },
        timeout=5.0,
    )
    assert resident.sent is not None
    assert "grammar" not in resident.sent
    assert result["hawking"]["grammar_enforced"] is False


def test_json_schema_response_format_does_not_send_grammar():
    resident = _FakeResident()
    connector = _connector(resident)
    connector.complete_payload(
        {
            "messages": [{"role": "user", "content": "go"}],
            "response_format": {"type": "json_schema", "json_schema": {"type": "object"}},
            "grammar": "root ::= object",
            "max_tokens": 512,
        },
        timeout=5.0,
    )
    assert resident.sent is not None
    assert "grammar" not in resident.sent
    assert set(resident.sent) == {"id", "prompt", "max_new_tokens", "max_seq_len"}
