"""The sealed body loops, and the detector could not see it.

Four structure-only questions were put to sealed-3.14. All four produced real
content and then collapsed into phrase repetition. is_degenerate() returned
False on every one of them, so nothing downstream could have known.

These tests pin the fix against the ACTUAL captured replies, not synthetic
strings, and against the negative controls that matter for the contract: prose
and JSON. Source files are not an input this function ever sees.
"""
from __future__ import annotations

import json

import pytest

from tools.future._common import REPO
from tools.future.resident_provider import (
    SHINGLE_NOVELTY_FLOOR,
    degenerate_prefix,
    is_degenerate,
)

RAW = REPO / "receipts" / "future" / "_G111_RESIDENT_REPLIES_raw.json"


def _replies():
    if not RAW.is_file():
        pytest.skip(f"{RAW} absent")
    return {r["name"]: r["reply"] for r in json.loads(RAW.read_text())["replies"]}


def test_the_captured_loops_are_now_detected():
    r = _replies()
    for name in ("q1_structure", "q2_program", "q4_moe"):
        assert is_degenerate(r[name]) is True, f"{name} loops and must be caught"


def test_the_clean_answer_is_not_flagged():
    """q3 repeats by enumeration, not collapse. Flagging it would lose science."""
    assert is_degenerate(_replies()["q3_subbit"]) is False


def test_the_prefix_before_the_loop_is_salvaged():
    """The body produces real content THEN loops; discarding it all is wrong."""
    r = _replies()
    for name in ("q2_program", "q4_moe"):
        pre = degenerate_prefix(r[name])
        assert 0 < len(pre) < len(r[name])
        assert not is_degenerate(pre), "the salvaged prefix must itself be clean"
        assert len(pre) > 1000, "the real content must survive the cut"


def test_a_clean_reply_is_returned_whole():
    r = _replies()["q3_subbit"]
    assert degenerate_prefix(r) == r


def test_ordinary_prose_is_not_flagged():
    assert is_degenerate((REPO / "README.md").read_text()[:6000]) is False
    assert is_degenerate("Paris") is False
    assert is_degenerate(
        " ".join(f"Sentence {i} covers a distinct facet." for i in range(60))
    ) is False


def test_a_json_reply_is_not_flagged():
    """S030 asks the resident for JSON. Repeated keys must not read as a loop."""
    payload = json.dumps(
        {"hypotheses": [{"id": i, "claim": "x", "why": "y"} for i in range(40)]},
        indent=1,
    )
    assert is_degenerate(payload) is False


def test_the_floor_is_documented_and_separates_the_observed_cases():
    assert 0.5 < SHINGLE_NOVELTY_FLOOR < 0.7


def test_empty_and_none_are_not_degenerate():
    assert is_degenerate("") is False
    assert is_degenerate(None) is False  # type: ignore[arg-type]
    assert degenerate_prefix("") == ""
