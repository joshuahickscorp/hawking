"""A structured-output rejection must carry the reply it rejected.

Every failure receipt from the 553-minute unattended run said `response is not a
JSON object` and none of them said what the reply WAS. The offending text lived
on `StructuredOutputExhausted.last_text` and was dropped on the floor when the
record was built, so the whole failure class had to be inferred afterwards from a
`completion_tokens: 30` field -- which is how a diagnosis becomes a guess.

Bounded on purpose: an excerpt, with the full length beside it, and never the
prompt.
"""
from __future__ import annotations

from hcli.backends import StructuredOutputContract, schema_instruction
from hcli.engine import _REJECTED_EXCERPT_CHARS, _degraded_structured_record
from hcli.engine import HCLI_RESULT_SCHEMA as SCHEMA


def _contract():
    return StructuredOutputContract(schema=SCHEMA, instruction=schema_instruction(SCHEMA))


def test_the_rejected_reply_is_recorded():
    rec = _degraded_structured_record(
        _contract(), attempts=3, exhausted=True, last_text="Sure, I can help with that."
    )
    assert rec["rejected_reply_excerpt"] == "Sure, I can help with that."
    assert rec["rejected_reply_chars"] == 27
    assert rec["rejected_reply_truncated_in_receipt"] is False


def test_a_long_reply_is_bounded_and_says_so():
    text = "x" * (_REJECTED_EXCERPT_CHARS + 500)
    rec = _degraded_structured_record(
        _contract(), attempts=3, exhausted=True, last_text=text
    )
    assert len(rec["rejected_reply_excerpt"]) == _REJECTED_EXCERPT_CHARS
    assert rec["rejected_reply_chars"] == len(text)
    assert rec["rejected_reply_truncated_in_receipt"] is True


def test_a_successful_call_records_no_excerpt():
    """No reply was rejected, so there is nothing to carry."""
    rec = _degraded_structured_record(_contract(), attempts=1, exhausted=False)
    assert "rejected_reply_excerpt" not in rec


def test_the_engine_passes_last_text_through_on_exhaustion():
    """The call site, not the helper. Grep found the record built in three places."""
    import inspect

    from hcli import engine

    src = inspect.getsource(engine.Engine)
    assert "last_text=exc.last_text" in src, (
        "the exhaustion handler no longer forwards the rejected reply"
    )
