"""Rank candidate measurements by what they can DECIDE, not by how much they move.

The FPGA partition lane produced a result worth generalising. Sweeping the model's
three unknowns gave:

    r (FPGA rate / Apple rate)   14.2x speedup swing   cannot reach APPLE_ONLY
    t (link rate / Apple rate)    2.0x speedup swing   reaches APPLE_ONLY
    s (setup cost)                2.0x speedup swing   reaches APPLE_ONLY

Ranked by swing, r is measured first and answers nothing: it does not appear in
the break-even, so no value of it can decide whether the architecture exists. It
sizes a win it cannot create. The link decides.

So the ordering rule is:

    A MEASUREMENT IS WORTH ITS WALL TIME IN PROPORTION TO ITS ABILITY TO CHANGE
    THE DECISION, NOT ITS ABILITY TO CHANGE THE NUMBER.

An input that cannot cross a decision boundary anywhere in its plausible range is
not an experiment. It is a refinement, and refinements wait.

This module is deliberately domain-free: hand it named inputs, a plausible range
per input, and a function that returns a DECISION plus a magnitude, and it returns
the measurement order. It knows nothing about FPGAs.

It also answers the question that saves the most time of all: whether ANY
measurement can change the decision. When none can, the decision is already
determined by what is known, and the honest output is "run nothing".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Mapping, Sequence

#: What a `decide` callable must return.
DECISION_KEY = "decision"
MAGNITUDE_KEY = "magnitude"


class DecisionContract(ValueError):
    """The decide() callable did not return a decision and a magnitude."""


@dataclass(frozen=True)
class Measurement:
    """One candidate measurement, and what it is worth."""
    name: str
    range_swept: tuple[Any, Any]
    decisions_reachable: tuple[Hashable, ...]
    changes_decision: bool
    magnitude_min: float
    magnitude_max: float
    magnitude_swing: float
    measure_order: int = 0
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "range_swept": list(self.range_swept),
            "decisions_reachable": [str(d) for d in self.decisions_reachable],
            "changes_decision": self.changes_decision,
            "magnitude_min": self.magnitude_min,
            "magnitude_max": self.magnitude_max,
            "magnitude_swing": self.magnitude_swing,
            "measure_order": self.measure_order,
            "note": self.note,
        }


@dataclass(frozen=True)
class Ranking:
    """The measurement order, plus whether measuring anything is worth it."""
    measurements: tuple[Measurement, ...]
    baseline_decision: Hashable
    any_measurement_changes_the_decision: bool
    inputs_held_at: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_decision": str(self.baseline_decision),
            "any_measurement_changes_the_decision": self.any_measurement_changes_the_decision,
            "verdict": (
                "measure in this order; the first entries can change the decision"
                if self.any_measurement_changes_the_decision else
                "RUN NOTHING: no input can cross a decision boundary anywhere in its "
                "plausible range, so the decision is already determined and every "
                "candidate measurement is a refinement"
            ),
            "inputs_held_at": dict(self.inputs_held_at),
            "measurements": [m.as_dict() for m in self.measurements],
        }


def _median(values: Sequence[Any]) -> Any:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _call(decide: Callable[..., Mapping[str, Any]], point: dict[str, Any]) -> tuple[Hashable, float]:
    out = decide(**point)
    if not isinstance(out, Mapping) or DECISION_KEY not in out or MAGNITUDE_KEY not in out:
        raise DecisionContract(
            f"decide() must return a mapping with {DECISION_KEY!r} and {MAGNITUDE_KEY!r}; "
            f"got {type(out).__name__} with keys {sorted(out) if isinstance(out, Mapping) else '-'}"
        )
    return out[DECISION_KEY], float(out[MAGNITUDE_KEY])


def rank_measurements(
    inputs: Mapping[str, Sequence[Any]],
    decide: Callable[..., Mapping[str, Any]],
    *,
    hold_at: Mapping[str, Any] | None = None,
    notes: Mapping[str, str] | None = None,
) -> Ranking:
    """Order candidate measurements by decision power, then by magnitude swing.

    `inputs` maps each unknown to the values it plausibly takes. `decide` receives
    one value per input by keyword and returns {"decision": hashable, "magnitude": float}.

    Each input is swept with the others held fixed, which is a one-at-a-time
    design: it finds inputs that can move the decision ALONE. It will not find a
    pair that only crosses a boundary together, and that limit is deliberate -- a
    joint sweep costs the product of the grids and the ordering question rarely
    needs it. Say so rather than implying full coverage.

    `hold_at` states that fixed point. It defaults to each input's grid median,
    which is only a reasonable baseline when the grid is centred on plausible
    values: a grid invented to span a wide range has a median that is itself a
    strong claim, and holding an unknown there silently changes every other
    input's apparent power. State the baseline when you know it.
    """
    if not inputs:
        raise ValueError("no candidate measurements to rank")
    for name, values in inputs.items():
        if len(list(values)) < 2:
            raise ValueError(f"input {name!r} needs at least two values to be swept")

    held = {name: _median(list(values)) for name, values in inputs.items()}
    for name, value in (hold_at or {}).items():
        if name not in inputs:
            raise ValueError(f"hold_at names {name!r}, which is not a candidate measurement")
        held[name] = value
    baseline_decision, _ = _call(decide, dict(held))

    rows: list[Measurement] = []
    for name, values in inputs.items():
        vals = list(values)
        decisions: list[Hashable] = []
        magnitudes: list[float] = []
        for v in vals:
            point = dict(held)
            point[name] = v
            d, m = _call(decide, point)
            decisions.append(d)
            magnitudes.append(m)
        distinct = tuple(dict.fromkeys(decisions))
        lo, hi = min(magnitudes), max(magnitudes)
        rows.append(Measurement(
            name=name,
            range_swept=(sorted(vals)[0], sorted(vals)[-1]),
            decisions_reachable=distinct,
            changes_decision=len(distinct) > 1,
            magnitude_min=lo,
            magnitude_max=hi,
            # Ratio when the quantity is positive, else absolute spread. A swing
            # of 0/0 is not infinite information, it is no information.
            magnitude_swing=(hi / lo) if lo > 0 else (hi - lo),
            note=(notes or {}).get(name, ""),
        ))

    # THE ORDERING RULE. Decision power first; magnitude only breaks ties among
    # inputs that are equally (in)capable of deciding anything.
    rows.sort(key=lambda m: (m.changes_decision, len(m.decisions_reachable), m.magnitude_swing),
              reverse=True)
    ranked = tuple(
        Measurement(**{**m.__dict__, "measure_order": i}) for i, m in enumerate(rows, 1)
    )
    return Ranking(
        measurements=ranked,
        baseline_decision=baseline_decision,
        any_measurement_changes_the_decision=any(m.changes_decision for m in ranked),
        inputs_held_at=held,
    )
