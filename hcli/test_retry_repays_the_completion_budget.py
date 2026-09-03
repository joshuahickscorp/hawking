"""A retry that grows the prompt must shrink the completion budget.

`max_tokens` is derived ONCE, from the prompt the caller built, against a fixed
context window. Every retry appends its rejection reason to that payload, so the
prompt grew while the budget computed against it did not.

Measured on one goal: three attempts carrying max_tokens 3243 against prompts of
4776, 4876 and 5090 tokens. The first two fit the 8192 window. The third asked
for 8333, the runtime truncated the reply mid-object, and the truncated reply was
then rejected as malformed JSON -- the retry mechanism defeating itself while the
model's actual patch was correct.
"""
from __future__ import annotations

from hcli.backends import append_user_text


def test_a_note_costs_the_completion_budget_what_it_costs_the_prompt():
    payload = {"messages": [{"role": "user", "content": "goal"}], "max_tokens": 3243}
    append_user_text(payload, "y" * 400)
    assert payload["max_tokens"] == 3243 - 100, payload["max_tokens"]


def test_the_prompt_string_form_is_charged_too():
    """Not every backend takes messages."""
    payload = {"prompt": "goal", "max_tokens": 1000}
    append_user_text(payload, "z" * 200)
    assert payload["max_tokens"] == 950


def test_a_skipped_append_is_not_charged():
    """skip_if means nothing was added, so nothing is owed."""
    payload = {
        "messages": [{"role": "user", "content": "already has MARKER here"}],
        "max_tokens": 900,
    }
    append_user_text(payload, "x" * 4000, skip_if="MARKER")
    assert payload["max_tokens"] == 900


def test_the_budget_never_goes_to_zero_or_negative():
    """A huge note must leave a reply possible, not make the call pointless."""
    payload = {"messages": [{"role": "user", "content": "g"}], "max_tokens": 300}
    append_user_text(payload, "q" * 100_000)
    assert payload["max_tokens"] == 256


def test_a_payload_without_a_budget_is_left_alone():
    payload = {"messages": [{"role": "user", "content": "g"}]}
    append_user_text(payload, "note")
    assert "max_tokens" not in payload
    assert payload["messages"][-1]["content"] == "gnote"


def test_the_note_still_actually_reaches_the_model():
    """The whole point of the retry is that the model is told what was wrong."""
    payload = {"messages": [{"role": "user", "content": "goal"}], "max_tokens": 500}
    append_user_text(payload, " REASON: anchor did not match")
    assert "REASON: anchor did not match" in payload["messages"][-1]["content"]
