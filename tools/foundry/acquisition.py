#!/usr/bin/env python3.12
"""Acquisition function: what to try next, structurally under the one-bit ceiling.

The acquisition function is the one component whose whole job is to invent new
candidates, so it is the one most able to smuggle a higher rate back in. It cannot:

  - Candidate is frozen and validates in __post_init__, so a >1 BPW candidate cannot be
    CONSTRUCTED, let alone returned;
  - propose() refuses an above-ceiling request instead of clamping it, so asking for a
    1.2 anchor is an error and not a silent success;
  - the emit path asserts the ceiling one more time on the way out.

Ordering is expected capability gain at a FIXED ceiling, never rate. F1 measured that
the raw-weight rate-distortion limit binds (Qwen3-235B: complete 1.0075 BPW collapsed
6/6, complete 0.4930 BPW collapsed 6/6), so the families that CHANGE the source lead.

A rate only ever moves DOWN, and only once every family is exhausted at the current
rate: that is gravity_potency.check_rate_discipline, and this module proposes nothing
that would violate it.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gravity_potency import (  # noqa: E402
    DEAD_METHOD_FAMILIES, METHOD_FAMILY_ORDER, ONE_BIT_CEILING, RATE_LADDER,
    on_ladder, parse_bpw, show_rate,
)

SCHEMA_PROPOSAL = "hawking.foundry.acquisition_proposal.v1"


class CeilingViolation(AssertionError):
    """Something tried to leave, or enter, above complete BPW 1/1."""


# Expected capability gain at a fixed ceiling. Priors, not measurements: they seed the
# order only, and a real parent-vs-packed forward selects. Descending order here IS
# METHOD_FAMILY_ORDER; _assert_order_matches_gain keeps the two from drifting.
FAMILY_EXPECTED_GAIN: dict[str, float] = {
    "quantization_aware_training": 0.62,
    "compressibility_training": 0.55,
    "distillation": 0.52,
    "learned_sharing": 0.40,
    "structured_pruning": 0.34,
    "representation_geometry": 0.26,
    "allocation": 0.18,
}

FAMILY_RATIONALE: dict[str, str] = {
    "quantization_aware_training":
        "train the weights to BE the one-bit code; not bound by the rate-distortion "
        "limit of the original weights that F1 hit at complete 1.0075 BPW",
    "compressibility_training":
        "train the source to be compressible at the ceiling instead of coding a source "
        "that is not",
    "distillation":
        "distil the parent into a natively sub-bit student; the student's weights never "
        "have to survive quantization of the parent's",
    "learned_sharing":
        "sharing trained IN. Post-hoc sharing is dead: mean pairwise expert cosine 1e-4, "
        "so there is nothing to subtract that was not trained to be shared",
    "structured_pruning":
        "change the model, not the code: fewer weights at the same complete budget buys "
        "bits per surviving weight",
    "representation_geometry":
        "transform the weight space before coding. Raw-weight PQ/VQ is falsified; a "
        "learned or rotated space is a different family",
    "allocation":
        "organ- and routing-aware allocation at a fixed ceiling. Uniform allocation is "
        "falsified, and allocation alone did not close the gap at F1",
}


def _assert_order_matches_gain() -> None:
    ranked = tuple(sorted(FAMILY_EXPECTED_GAIN, key=lambda f: -FAMILY_EXPECTED_GAIN[f]))
    if ranked != tuple(METHOD_FAMILY_ORDER):
        raise AssertionError(
            "acquisition order drifted from METHOD_FAMILY_ORDER: proposals must be "
            f"orderable by expected gain. gain order {ranked} vs law {METHOD_FAMILY_ORDER}")


_assert_order_matches_gain()


@dataclasses.dataclass(frozen=True)
class Candidate:
    """A proposal. Cannot exist above the ceiling: the filter is the constructor."""

    method_family: str
    rate: Fraction
    expected_capability_gain: float
    rationale: str

    def __post_init__(self) -> None:
        rate = parse_bpw(self.rate)
        object.__setattr__(self, "rate", rate)
        if rate > ONE_BIT_CEILING:
            raise CeilingViolation(
                f"candidate complete BPW {show_rate(rate)} is above the one-bit ceiling "
                f"{show_rate(ONE_BIT_CEILING)}; complete_artifact_bits / "
                "original_weight_count must be <= 1/1 with nothing excluded as overhead")
        if not on_ladder(rate):
            raise CeilingViolation(
                f"candidate rate {show_rate(rate)} is not on the exact rate ladder")
        if self.method_family in DEAD_METHOD_FAMILIES:
            raise CeilingViolation(
                f"method family {self.method_family!r} is falsified (F1 raw-weight PQ/VQ "
                "collapsed 6/6 at complete 1.0075 and 0.4930 BPW)")
        if self.method_family not in METHOD_FAMILY_ORDER:
            raise CeilingViolation(
                f"method family {self.method_family!r} is not a materially distinct family; "
                f"choose from {list(METHOD_FAMILY_ORDER)}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "method_family": self.method_family,
            "complete_bpw": show_rate(self.rate),
            "expected_capability_gain": self.expected_capability_gain,
            "rationale": self.rationale,
        }


def next_rate_below(rate: Fraction) -> Optional[Fraction]:
    """The next legal rate DOWN the ladder. There is no next rate up, by construction."""
    lower = [r for r in RATE_LADDER if r < rate]
    return max(lower) if lower else None


def propose(state: Optional[dict[str, Any]] = None, *, limit: int = 3) -> list[Candidate]:
    """Next candidates, best expected capability gain first, all at or under the ceiling.

    state: {"rate": "1/1", "exhausted": [families done at that rate],
            "families": [explicit ask, still filtered]}

    An above-ceiling request raises rather than being quietly clamped: a caller that
    asks for 1.2 has to see the refusal.
    """
    state = state or {}
    rate = parse_bpw(state.get("rate", ONE_BIT_CEILING))
    if rate > ONE_BIT_CEILING:
        raise CeilingViolation(
            f"refused: acquisition was asked to work at complete BPW {show_rate(rate)}, above "
            f"the one-bit ceiling {show_rate(ONE_BIT_CEILING)}. Upward bracketing is rejected; "
            "the answer to a failure at the ceiling is a different METHOD at the ceiling")

    exhausted = set(state.get("exhausted") or ())
    asked = list(state.get("families") or METHOD_FAMILY_ORDER)
    families = [f for f in asked if f not in exhausted]
    if not families:
        lower = next_rate_below(rate)
        if lower is None:
            return []
        rate, families = lower, list(asked)

    out: list[Candidate] = []
    for family in sorted(families, key=lambda f: -FAMILY_EXPECTED_GAIN.get(f, 0.0)):
        if family in DEAD_METHOD_FAMILIES or family not in METHOD_FAMILY_ORDER:
            continue  # filtered at construction time, before a Candidate can exist
        out.append(Candidate(family, rate, FAMILY_EXPECTED_GAIN[family],
                             FAMILY_RATIONALE[family]))
        if len(out) >= limit:
            break
    return _emit(out)


def _emit(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Last gate on the way out. Belt and braces: the constructor already refused."""
    out = list(candidates)
    assert all(c.rate <= ONE_BIT_CEILING for c in out), (
        "acquisition emitted a candidate above the one-bit ceiling: "
        + ", ".join(show_rate(c.rate) for c in out if c.rate > ONE_BIT_CEILING))
    return out


def proposal(state: Optional[dict[str, Any]] = None, *, limit: int = 3) -> dict[str, Any]:
    """JSON-shaped proposal, for a program document or a review artifact."""
    state = state or {}
    candidates = propose(state, limit=limit)
    return {
        "schema": SCHEMA_PROPOSAL,
        "ceiling": show_rate(ONE_BIT_CEILING),
        "objective": "maximize capability subject to complete BPW <= 1/1",
        "rate": show_rate(candidates[0].rate) if candidates else None,
        "exhausted": sorted(state.get("exhausted") or ()),
        "candidates": [c.as_dict() for c in candidates],
        "rate_change_law": "downward only, after every method family is exhausted at the "
                           "current rate; upward never",
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rate", default="1/1", help="exact identity, e.g. 1/1 or 1/2")
    ap.add_argument("--exhausted", default="", help="comma separated method families")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args(argv)
    state = {"rate": args.rate,
             "exhausted": [f for f in args.exhausted.split(",") if f]}
    print(json.dumps(proposal(state, limit=args.limit), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
