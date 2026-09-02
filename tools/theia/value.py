"""H.1 bounty value function.

The score schedules work. It never declares the result true. That is a
type boundary: `bounty_value` returns `ScheduleScore` and cannot produce
`VerifiedResult`. `accept_as_verified` TypeErrors on a schedule score.

Missing factors REFUSE (complete_ebpw doctrine: a calculator with a missing
input is a guess). A zero or negative cost term REFUSES — it would send the
score to infinity and look like a truth claim about cheapness.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping


NUMERATOR_FACTORS = (
    "verified_reward",
    "probability_of_success",
    "information_gain",
    "transfer_value",
    "strategic_relevance",
)
DENOMINATOR_FACTORS = (
    "wall_time",
    "compute_cost",
    "human_cost",
    "risk",
    "opportunity_cost",
)
REQUIRED_FACTORS = NUMERATOR_FACTORS + DENOMINATOR_FACTORS


class ValueRefused(ValueError):
    """An H.1 factor is missing, ungrounded, or non-positive."""


@dataclass(frozen=True)
class DeclaredFactor:
    value: Fraction
    name: str
    source: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueRefused("DeclaredFactor.name is required")
        if not self.source:
            raise ValueRefused(
                f"{self.name} has an empty source; an ungrounded factor is a guess"
            )
        if self.value <= 0:
            raise ValueRefused(
                f"{self.name}={self.value} must be positive; zero/negative is "
                "undefined under H.1 (would inflate or invert the schedule)"
            )


@dataclass(frozen=True)
class ValueInputs:
    verified_reward: DeclaredFactor
    probability_of_success: DeclaredFactor
    information_gain: DeclaredFactor
    transfer_value: DeclaredFactor
    strategic_relevance: DeclaredFactor
    wall_time: DeclaredFactor
    compute_cost: DeclaredFactor
    human_cost: DeclaredFactor
    risk: DeclaredFactor
    opportunity_cost: DeclaredFactor


@dataclass(frozen=True)
class ScheduleScore:
    """H.1 output. Schedules work. Not a verification verdict."""

    value: Fraction
    inputs: ValueInputs
    formula: str = (
        "verified_reward * probability_of_success * information_gain * "
        "transfer_value * strategic_relevance / "
        "(wall_time * compute_cost * human_cost * risk * opportunity_cost)"
    )

    def __bool__(self) -> bool:  # noqa: D105
        raise TypeError(
            "the H.1 value function schedules work and never declares the "
            "result true; a ScheduleScore has no boolean truth"
        )

    def to_json_dict(self) -> dict[str, Any]:
        factors = {}
        for name in REQUIRED_FACTORS:
            f: DeclaredFactor = getattr(self.inputs, name)
            factors[name] = {
                "numerator": f.value.numerator,
                "denominator": f.value.denominator,
                "source": f.source,
            }
        return {
            "value_numerator": self.value.numerator,
            "value_denominator": self.value.denominator,
            "formula": self.formula,
            "declares_result_true": False,
            "type": type(self).__name__,
            "factors": factors,
        }


@dataclass(frozen=True)
class VerifiedResult:
    """Produced only by independent verification, never by bounty_value."""

    kind: str
    artifact: str
    detail: dict[str, Any]

    @classmethod
    def from_schedule(cls, score: ScheduleScore) -> "VerifiedResult":
        raise TypeError(
            "the H.1 value function schedules work and never declares the "
            f"result true; refused to build VerifiedResult from {type(score).__name__}"
        )


def accept_as_verified(obj: VerifiedResult) -> VerifiedResult:
    if type(obj) is not VerifiedResult:
        raise TypeError(
            "the H.1 value function schedules work and never declares the "
            f"result true; accept_as_verified got {type(obj).__name__}"
        )
    return obj


def value_inputs_from_mapping(raw: Mapping[str, Mapping[str, Any]]) -> ValueInputs:
    """Fail closed on a missing factor. Same doctrine as complete_ebpw.cost."""
    missing = [k for k in REQUIRED_FACTORS if k not in raw]
    if missing:
        raise ValueRefused(
            f"H.1 factors missing {missing}; refusing rather than defaulting to 1"
        )
    built: dict[str, DeclaredFactor] = {}
    for name in REQUIRED_FACTORS:
        row = raw[name]
        if not isinstance(row, Mapping) or "value" not in row or "source" not in row:
            raise ValueRefused(f"{name} must be {{value, source}}")
        val = row["value"]
        if isinstance(val, Fraction):
            frac = val
        elif isinstance(val, int):
            frac = Fraction(val)
        elif isinstance(val, Mapping) and "numerator" in val:
            frac = Fraction(int(val["numerator"]), int(val["denominator"]))
        else:
            raise ValueRefused(f"{name}.value must be int or Fraction, not {type(val).__name__}")
        built[name] = DeclaredFactor(value=frac, name=name, source=str(row["source"]))
    return ValueInputs(**built)


def bounty_value(inputs: ValueInputs) -> ScheduleScore:
    numer = Fraction(1)
    denom = Fraction(1)
    for name in NUMERATOR_FACTORS:
        numer *= getattr(inputs, name).value
    for name in DENOMINATOR_FACTORS:
        denom *= getattr(inputs, name).value
    return ScheduleScore(value=numer / denom, inputs=inputs)
