"""Recomposed science module one_bit_ceiling (C-SCI-R1)."""
from __future__ import annotations

def parse_rate(text):
    from lab.operators import gravity_potency as _gp
    return _gp.parse_rate(text)


import json
from fractions import Fraction
from typing import Any

SCHEMA = 'hawking.foundry.one_bit_ceiling.v1'
CEILING = Fraction(1, 1)
COMPONENTS: tuple[str, ...] = ('indices', 'codebooks', 'scales', 'metadata', 'alignment', 'protected_islands', 'doctor', 'pass_through_tensors', 'packaging', 'runtime_tables')
RESERVE = 'metadata_alignment_reserve_bits'
REQUIRED_FIELDS: frozenset[str] = frozenset(COMPONENTS) | {RESERVE}

class CeilingError(AssertionError):
    """Base: this candidate may not be run as science."""

class IncompleteLedger(CeilingError):
    """A component was not declared. Undeclared is not zero."""

class CeilingViolation(CeilingError):
    """complete_bits / weights > 1."""

def _rate(value: Any) -> Fraction:
    """Exact rational. float keeps its exact binary value; no rounding anywhere."""
    return Fraction(value) if isinstance(value, float) else parse_rate(value)

def _exact(name: str, value: Any) -> Fraction:
    """int / Fraction / 'n/d' / {num,den} / float, converted with zero rounding.

    float is accepted but converted to its EXACT binary value, so 1e-9 of overage
    survives the conversion and trips the ceiling instead of vanishing.
    """
    if value is None:
        raise IncompleteLedger(f'{name}: declared but unset; undeclared is not zero')
    if isinstance(value, bool):
        raise IncompleteLedger(f'{name}: bool is not a bit count')
    try:
        q = _rate(value)
    except Exception as exc:
        raise IncompleteLedger(f'{name}: {exc}') from exc
    if q < 0:
        raise IncompleteLedger(f'{name}: negative bits ({q}); a component may not pay for another')
    return q

class CompleteByteLedger:
    """Itemized, exhaustive bit accounting for one candidate artifact.

    Every one of COMPONENTS plus the explicit reserve must be passed. Zero is a
    legal declaration; omission is not. Unknown keys are refused so nothing can
    hide in a "misc" bucket outside the ten named slots.
    """
    __slots__ = ('bits', 'note')

    def __init__(self, *, note: str='', **components: Any) -> None:
        given = set(components)
        missing = sorted(REQUIRED_FIELDS - given)
        unknown = sorted(given - REQUIRED_FIELDS)
        if missing:
            raise IncompleteLedger('incomplete ledger, undeclared components are NOT zero: ' + ', '.join(missing))
        if unknown:
            raise IncompleteLedger('unknown ledger components (every bit belongs in a named slot): ' + ', '.join(unknown))
        self.bits = {k: _exact(k, v) for k, v in components.items()}
        self.note = note

    def complete_bits(self) -> Fraction:
        """Every component plus the declared reserve. Exact."""
        return sum(self.bits.values(), Fraction(0))

    def itemized_bits(self) -> Fraction:
        """The ten components without the reserve headroom."""
        return sum((self.bits[c] for c in COMPONENTS), Fraction(0))

    def complete_bpw(self, original_weight_count: int) -> Fraction:
        w = int(original_weight_count)
        if w <= 0:
            raise IncompleteLedger(f'original_weight_count must be positive, got {w}')
        return self.complete_bits() / w

    def as_dict(self, original_weight_count: int | None=None) -> dict[str, Any]:
        out: dict[str, Any] = {'schema': SCHEMA, 'scope': 'whole_model', 'components': {k: str(v) for k, v in self.bits.items()}, 'itemized_bits': str(self.itemized_bits()), 'reserve_bits': str(self.bits[RESERVE]), 'complete_bits': str(self.complete_bits()), 'note': self.note}
        if original_weight_count is not None:
            bpw = self.complete_bpw(original_weight_count)
            out |= {'original_weight_count': int(original_weight_count), 'complete_bpw_exact': f'{bpw.numerator}/{bpw.denominator}', 'complete_bpw_float': float(bpw), 'legal': bpw <= CEILING}
        return out

def assert_complete_bpw_le_one(ledger: CompleteByteLedger, original_weight_count: int) -> dict[str, Any]:
    """Enforce the ceiling. Returns a receipt on pass, raises with the overage on fail."""
    bpw = ledger.complete_bpw(original_weight_count)
    if bpw > CEILING:
        over = bpw - CEILING
        over_bits = ledger.complete_bits() - int(original_weight_count)
        raise CeilingViolation(f'one-bit ceiling violated: complete {float(bpw):.9f} BPW (exact {bpw.numerator}/{bpw.denominator}) over {int(original_weight_count)} weights; overage {float(over):.9f} BPW = {float(over_bits):.0f} bits = {float(over_bits) / 8 / 1024 ** 2:.1f} MiB; rebudget to <= 1/1, do not raise the ceiling')
    return {'schema': SCHEMA, 'legal': True, 'complete_bpw_exact': f'{bpw.numerator}/{bpw.denominator}', 'complete_bpw_float': float(bpw), 'headroom_bits': str(int(original_weight_count) - ledger.complete_bits()), 'reserve_bits': str(ledger.bits[RESERVE]), 'scope': 'whole_model'}

def is_legal_candidate(spec: dict[str, Any]) -> tuple[bool, list[str]]:
    """Candidate validation. (legal, reasons). Reasons are empty iff legal.

    spec:
      original_weight_count : int, the PARENT weight count (whole model)
      ledger                : CompleteByteLedger, or a dict of the eleven fields
      reported_bpw          : optional, must equal the ledger rate exactly
      target_bpw            : optional, refused if > 1 (no upward bracketing)
      scope                 : optional, must be "whole_model" if present
      expert_only_bpw / payload_only_bpw / *_only_bpw : recorded, never accepted
                              as the whole-model rate
    """
    reasons: list[str] = []
    spec = dict(spec or {})
    ledger = spec.get('ledger')
    if isinstance(ledger, dict):
        try:
            ledger = CompleteByteLedger(**ledger)
        except CeilingError as exc:
            return (False, [str(exc)])
    if not isinstance(ledger, CompleteByteLedger):
        return (False, ['spec has no CompleteByteLedger; a byte plan without an itemized ledger is not a candidate'])
    scope = spec.get('scope', 'whole_model')
    if scope != 'whole_model':
        reasons.append(f'scope {scope!r}: the ceiling is whole-model only')
    partial = sorted((k for k in spec if k.endswith('_only_bpw')))
    if partial and 'reported_bpw' not in spec:
        reasons.append('partial-scope rate present (' + ', '.join(partial) + ') with no whole-model reported_bpw; a partial BPW may never stand in for the whole model')
    try:
        bpw = ledger.complete_bpw(spec.get('original_weight_count', 0))
    except CeilingError as exc:
        return (False, reasons + [str(exc)])
    for key in partial:
        try:
            claimed = _rate(spec[key])
        except Exception:
            reasons.append(f'{key}: unparseable rate {spec[key]!r}')
            continue
        if claimed < bpw:
            reasons.append(f'{key}={float(claimed):.6f} understates the whole-model {float(bpw):.6f} BPW; report the whole-model rate')
    if 'reported_bpw' in spec:
        reported = _rate(spec['reported_bpw'])
        if reported != bpw:
            reasons.append(f'reported_bpw {float(reported):.9f} != ledger {float(bpw):.9f}; the reported rate must be the complete whole-model rate')
    if 'target_bpw' in spec:
        target = _rate(spec['target_bpw'])
        if target > CEILING:
            reasons.append(f'target_bpw {float(target):.4f} > 1: upward bracketing is REJECTED (no 1.2 anchor, no 1.5, no 2.0, no 3.0, no automatic Escape Receipt)')
    try:
        assert_complete_bpw_le_one(ledger, spec['original_weight_count'])
    except CeilingViolation as exc:
        reasons.append(str(exc))
    return (not reasons, reasons)
A1_1P0_WEIGHTS = 235093634560
A1_1P0_TOTAL_BITS = 29606271552 * 8

def a1_1p0_ledger() -> CompleteByteLedger:
    zeros = {c: 0 for c in COMPONENTS}
    return CompleteByteLedger(**zeros | {'indices': A1_1P0_TOTAL_BITS}, metadata_alignment_reserve_bits=0, note='Qwen3-235B A1_1p0 sealed total; per-component split not sealed, other slots are lower bounds')

def a1_1p0_verdict() -> dict[str, Any]:
    legal, reasons = is_legal_candidate({'candidate_id': 'A1_1p0', 'parent': 'qwen3-235b', 'original_weight_count': A1_1P0_WEIGHTS, 'ledger': a1_1p0_ledger(), 'reported_bpw': f'{A1_1P0_TOTAL_BITS}/{A1_1P0_WEIGHTS}'})
    return {'candidate_id': 'A1_1p0', 'legal': legal, 'reasons': reasons}
