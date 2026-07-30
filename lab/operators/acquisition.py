"""Recomposed science module acquisition (C-SCI-R1)."""
from __future__ import annotations
from lab.operators.gravity_potency import DEAD_METHOD_FAMILIES
from lab.operators.gravity_potency import METHOD_FAMILY_ORDER
from lab.operators.gravity_potency import ONE_BIT_CEILING
from lab.operators.gravity_potency import RATE_LADDER
from lab.operators.gravity_potency import on_ladder
from lab.operators.gravity_potency import parse_bpw
from lab.operators.gravity_potency import show_rate
import argparse
import dataclasses
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Optional
_HERE = Path(__file__).resolve().parent
SCHEMA_PROPOSAL = 'hawking.foundry.acquisition_proposal.v1'

class CeilingViolation(AssertionError):
    """Something tried to leave, or enter, above complete BPW 1/1."""
FAMILY_EXPECTED_GAIN: dict[str, float] = {'quantization_aware_training': 0.62, 'compressibility_training': 0.55, 'distillation': 0.52, 'learned_sharing': 0.4, 'structured_pruning': 0.34, 'representation_geometry': 0.26, 'allocation': 0.18}
FAMILY_RATIONALE: dict[str, str] = {'quantization_aware_training': 'train the weights to BE the one-bit code; not bound by the rate-distortion limit of the original weights that F1 hit at complete 1.0075 BPW', 'compressibility_training': 'train the source to be compressible at the ceiling instead of coding a source that is not', 'distillation': "distil the parent into a natively sub-bit student; the student's weights never have to survive quantization of the parent's", 'learned_sharing': 'sharing trained IN. Post-hoc sharing is dead: mean pairwise expert cosine 1e-4, so there is nothing to subtract that was not trained to be shared', 'structured_pruning': 'change the model, not the code: fewer weights at the same complete budget buys bits per surviving weight', 'representation_geometry': 'transform the weight space before coding. Raw-weight PQ/VQ is falsified; a learned or rotated space is a different family', 'allocation': 'organ- and routing-aware allocation at a fixed ceiling. Uniform allocation is falsified, and allocation alone did not close the gap at F1'}

def _assert_order_matches_gain() -> None:
    ranked = tuple(sorted(FAMILY_EXPECTED_GAIN, key=lambda f: -FAMILY_EXPECTED_GAIN[f]))
    if ranked != tuple(METHOD_FAMILY_ORDER):
        raise AssertionError(f'acquisition order drifted from METHOD_FAMILY_ORDER: proposals must be orderable by expected gain. gain order {ranked} vs law {METHOD_FAMILY_ORDER}')
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
        object.__setattr__(self, 'rate', rate)
        if rate > ONE_BIT_CEILING:
            raise CeilingViolation(f'candidate complete BPW {show_rate(rate)} is above the one-bit ceiling {show_rate(ONE_BIT_CEILING)}; complete_artifact_bits / original_weight_count must be <= 1/1 with nothing excluded as overhead')
        if not on_ladder(rate):
            raise CeilingViolation(f'candidate rate {show_rate(rate)} is not on the exact rate ladder')
        if self.method_family in DEAD_METHOD_FAMILIES:
            raise CeilingViolation(f'method family {self.method_family!r} is falsified (F1 raw-weight PQ/VQ collapsed 6/6 at complete 1.0075 and 0.4930 BPW)')
        if self.method_family not in METHOD_FAMILY_ORDER:
            raise CeilingViolation(f'method family {self.method_family!r} is not a materially distinct family; choose from {list(METHOD_FAMILY_ORDER)}')

    def as_dict(self) -> dict[str, Any]:
        return {'method_family': self.method_family, 'complete_bpw': show_rate(self.rate), 'expected_capability_gain': self.expected_capability_gain, 'rationale': self.rationale}

def next_rate_below(rate: Fraction) -> Optional[Fraction]:
    """The next legal rate DOWN the ladder. There is no next rate up, by construction."""
    lower = [r for r in RATE_LADDER if r < rate]
    return max(lower) if lower else None

def propose(state: Optional[dict[str, Any]]=None, *, limit: int=3) -> list[Candidate]:
    """Next candidates, best expected capability gain first, all at or under the ceiling.

    state: {"rate": "1/1", "exhausted": [families done at that rate],
            "families": [explicit ask, still filtered]}

    An above-ceiling request raises rather than being quietly clamped: a caller that
    asks for 1.2 has to see the refusal.
    """
    state = state or {}
    rate = parse_bpw(state.get('rate', ONE_BIT_CEILING))
    if rate > ONE_BIT_CEILING:
        raise CeilingViolation(f'refused: acquisition was asked to work at complete BPW {show_rate(rate)}, above the one-bit ceiling {show_rate(ONE_BIT_CEILING)}. Upward bracketing is rejected; the answer to a failure at the ceiling is a different METHOD at the ceiling')
    exhausted = set(state.get('exhausted') or ())
    asked = list(state.get('families') or METHOD_FAMILY_ORDER)
    families = [f for f in asked if f not in exhausted]
    if not families:
        lower = next_rate_below(rate)
        if lower is None:
            return []
        rate, families = (lower, list(asked))
    out: list[Candidate] = []
    for family in sorted(families, key=lambda f: -FAMILY_EXPECTED_GAIN.get(f, 0.0)):
        if family in DEAD_METHOD_FAMILIES or family not in METHOD_FAMILY_ORDER:
            continue
        out.append(Candidate(family, rate, FAMILY_EXPECTED_GAIN[family], FAMILY_RATIONALE[family]))
        if len(out) >= limit:
            break
    return _emit(out)

def _emit(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Last gate on the way out. Belt and braces: the constructor already refused."""
    out = list(candidates)
    assert all((c.rate <= ONE_BIT_CEILING for c in out)), 'acquisition emitted a candidate above the one-bit ceiling: ' + ', '.join((show_rate(c.rate) for c in out if c.rate > ONE_BIT_CEILING))
    return out

def proposal(state: Optional[dict[str, Any]]=None, *, limit: int=3) -> dict[str, Any]:
    """JSON-shaped proposal, for a program document or a review artifact."""
    state = state or {}
    candidates = propose(state, limit=limit)
    return {'schema': SCHEMA_PROPOSAL, 'ceiling': show_rate(ONE_BIT_CEILING), 'objective': 'maximize capability subject to complete BPW <= 1/1', 'rate': show_rate(candidates[0].rate) if candidates else None, 'exhausted': sorted(state.get('exhausted') or ()), 'candidates': [c.as_dict() for c in candidates], 'rate_change_law': 'downward only, after every method family is exhausted at the current rate; upward never'}
