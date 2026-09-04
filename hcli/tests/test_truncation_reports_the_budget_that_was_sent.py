"""Tell the model the budget it ACTUALLY had, not the one the plan remembers.

Measured, receipt 5b2e060a:

    plan max_tokens 2048 | PAYLOAD received 1446 | GRANTED 1446 | gen 1446 | budget
    plan max_tokens 2048 | PAYLOAD received 1338 | GRANTED 1338 | gen 1338 | budget

The engine SENT 1446, the connector granted exactly that, the resident produced
exactly that, and the receipt recorded 2048 -- the pre-reduction plan. The
retry instruction built from it then read:

    "model produced 1338 tokens against a 2048-token completion budget ...
     the runtime stopped 710 tokens SHORT of the budget, so the real ceiling is
     the runtime's, not this budget; answer far more briefly"

Every clause after the comma is false. The runtime did not stop short -- it
delivered its budget exactly. And "answer far more briefly" is advice against a
ceiling the model had already hit, so three attempts were spent on a correction
that could not work.

`_truncation_message`'s own docstring describes this disease in the opposite
direction: "the native adapter was capping an explicit 6310-token request at its
2048 default, and every receipt said 'hit the 6310-token completion budget' -- so
the real ceiling was invisible in the only artifact anyone reads."
"""
from __future__ import annotations

from hcli.engine import _truncation_message


def test_exhausting_the_budget_is_not_reported_as_stopping_short():
    msg = _truncation_message(1338, 1338, 4135)
    assert "SHORT of the budget" not in msg, (
        "completion equals the budget, so the budget was exhausted; claiming the "
        f"runtime stopped short misdiagnoses it: {msg!r}"
    )
    assert "1338-token completion budget" in msg


def test_a_genuine_short_stop_is_still_reported():
    """The guard must not blind the case it was built for."""
    msg = _truncation_message(2048, 900, 4135)
    assert "SHORT of the budget" in msg
    assert "1148" in msg


def test_the_engine_prefers_the_granted_budget_over_the_plan():
    """The number has to reach the MESSAGE, not merely exist in a receipt.

    A helper that reports correctly when handed the right budget is worthless if
    the call site keeps handing it the stale one.
    """
    import inspect
    from hcli import engine as eng

    src = inspect.getsource(eng.Engine)
    assert "_last_granted_max_tokens" in src, (
        "the engine records max_new_tokens_granted in the receipt but the "
        "truncation message still reads plan['max_tokens'], which is the "
        "pre-reduction value"
    )
    # and the call sites must actually consult it
    hits = src.count("_last_granted_max_tokens")
    assert hits >= 3, (
        f"_last_granted_max_tokens appears {hits} times; it must be SET where the "
        "native metadata is read and PREFERRED at both truncation call sites"
    )
