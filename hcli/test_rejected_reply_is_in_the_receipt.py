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
    # BOUNDED, not exactly equal: the excerpt now keeps both ends with an
    # elision marker between them, so it carries a few characters of
    # bookkeeping. What the receipt must not do is grow with the reply.
    assert len(rec["rejected_reply_excerpt"]) <= _REJECTED_EXCERPT_CHARS + 64
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


def test_the_receipt_keeps_the_END_of_a_long_reply_not_only_its_head():
    """The live path, not the helper.

    A reply's `operations` sit at the end and its `content` prose at the start,
    so a head-only excerpt spent the whole budget on prose and cut off at the
    word "operations" -- the one part a rejection about an operation needs.
    Measured: a 2,221-character reply rejected three times for a bracket error
    inside new_text, and the receipt preserved 800 characters ending at
    '"op": "replace",'.
    """
    text = "HEAD" + "p" * 4000 + '"operations":[{"op":"replace","new_text":"BROKEN("}]'
    rec = _degraded_structured_record(
        _contract(), attempts=3, exhausted=True, last_text=text
    )
    excerpt = rec["rejected_reply_excerpt"]
    assert excerpt.startswith("HEAD")
    assert excerpt.endswith('"new_text":"BROKEN("}]'), excerpt[-60:]
    assert rec["rejected_reply_chars"] == len(text)
