"""Two rounds produced a byte-identical closing prompt from different observations.

Round 8: 5 successful observations, 3 tools. Round 9: 5 successful and 1 failed, 4 tools.
Both final calls were 1,553 prompt tokens and 6,292 rendered characters. Identical. A
prompt that does not vary with the observations it is supposed to report on does not
contain them, and both rounds answered "No observations were provided; no measurement was
made or recorded."

The engine has TWO closing paths and they disagree about how observations travel:

  budget-exhausted   passes history=conversation_history + [observations turn]   -> works
  post-loop closure  passes trailing=final_trailing and NO history               -> empty

`trailing` is the LEGACY field. The fit ladder says so in its own comment -- "Real tool
rounds use chat history, not the legacy ``trailing`` field" -- so the post-loop path hands
its observations to a channel the current builder no longer reads.

This is why the previous fix did not help. b3938ab03 stopped the ladder from SHEDDING
observations on the decision turn, which was a real defect, but shedding was never
reached: there was nothing to shed.

The rounds that DID report reported from an in-loop reply, not from the closing turn --
round 7's 566-token report was call 5 of 5, produced while the conversation was still
attached.
"""
from __future__ import annotations

from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "engine.py"


def _closing_branch(src: str) -> str:
    """The post-loop closure branch, from its prompt build to its call."""
    i = src.index("final_trailing = self._observations_block")
    return src[i - 1200:src.index("finally:", i)]


def test_the_post_loop_closure_passes_the_conversation():
    branch = _closing_branch(ENGINE.read_text())
    assert "history=" in branch, (
        "the post-loop closing call passes no history, so the model is asked to report on "
        "observations it was never given")


def test_both_closing_paths_use_the_same_channel():
    """One path on history and one on a legacy field is how they drifted apart."""
    src = ENGINE.read_text()
    # every _call_model that runs with tools closed must carry history
    closed_calls = src.count("self._tools_closed_for_round = True")
    with_history = sum(
        1 for chunk in src.split("self._tools_closed_for_round = True")[1:]
        if "history=" in chunk[:2500]
    )
    assert with_history == closed_calls, (
        f"{closed_calls} closing paths, only {with_history} pass history")


def test_the_legacy_trailing_field_is_not_the_only_carrier():
    branch = _closing_branch(ENGINE.read_text())
    if "trailing=" in branch:
        assert "history=" in branch, (
            "trailing is the legacy channel the fit ladder no longer reads; it must not be "
            "the only way observations reach the closing turn")
