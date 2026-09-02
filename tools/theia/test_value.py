"""H.1 type boundary: the value function cannot mark a result true."""
from __future__ import annotations

from fractions import Fraction

import pytest

from tools.theia.value import (
    REQUIRED_FACTORS,
    DeclaredFactor,
    ScheduleScore,
    ValueRefused,
    VerifiedResult,
    accept_as_verified,
    bounty_value,
    value_inputs_from_mapping,
)


def _inputs(**overrides):
    raw = {
        name: {"value": 1, "source": f"test:{name}"}
        for name in REQUIRED_FACTORS
    }
    raw.update(overrides)
    return value_inputs_from_mapping(raw)


def test_value_function_cannot_mark_a_result_true():
    score = bounty_value(_inputs())
    assert type(score) is ScheduleScore
    assert score.to_json_dict()["declares_result_true"] is False
    with pytest.raises(TypeError, match="never declares"):
        bool(score)
    with pytest.raises(TypeError, match="never declares"):
        accept_as_verified(score)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="never declares"):
        VerifiedResult.from_schedule(score)


def test_missing_factor_is_refused_not_defaulted():
    raw = {
        name: {"value": 1, "source": f"test:{name}"}
        for name in REQUIRED_FACTORS
        if name != "risk"
    }
    with pytest.raises(ValueRefused, match="missing"):
        value_inputs_from_mapping(raw)


def test_zero_cost_is_refused():
    with pytest.raises(ValueRefused, match="positive"):
        DeclaredFactor(value=Fraction(0), name="risk", source="zero")


def test_ungrounded_factor_is_refused():
    with pytest.raises(ValueRefused, match="empty source"):
        DeclaredFactor(value=Fraction(1), name="risk", source="")


def test_schedule_is_the_h1_ratio():
    score = bounty_value(
        _inputs(
            information_gain={"value": 4, "source": "n_scars"},
            transfer_value={"value": 4, "source": "laws"},
        )
    )
    assert score.value == Fraction(16)
