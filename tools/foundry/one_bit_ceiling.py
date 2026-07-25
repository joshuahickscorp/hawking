#!/usr/bin/env python3.12
"""The one-bit ceiling, as an enforced invariant.

LAW (operator, binding):
    "Hawking does not climb above one bit to discover where conventional
     quantization works. Hawking changes the representation, model, allocation
     and treatment until useful intelligence survives at one bit or below."

THE CEILING, for every scientific candidate:

    complete_artifact_bits / original_weight_count <= 1/1

COMPLETE means complete. Every component the law names has its own slot here and
every slot must be declared: indices, codebooks, scales, metadata, alignment,
protected_islands, doctor, pass_through_tensors, packaging, runtime_tables. There
is no "overhead" category that gets to be free, and there is no lump-sum entry
that lets a candidate skip itemization. A ledger that forgets to declare its
codebook is INCOMPLETE and is rejected, not silently treated as zero.

FORBIDDEN, permanently: any candidate above 1 BPW. No 1.2 safety anchor, no 1.5,
no 2.0, no 3.0, no automatic Escape Receipt. Upward bracketing is REJECTED. The
parent BF16 model is the quality reference; a compressed high-rate anchor is not
required and must not be scheduled.

An expert-only, payload-only or any other partial-scope BPW may NEVER be reported
as the whole-model BPW. If a candidate states a rate, it must equal the rate this
ledger computes, exactly.

Arithmetic is exact (int / Fraction) end to end. 1 + 1e-9 is a violation, not a
rounding tie. There is no tolerance, epsilon, or "close enough" branch, and none
may be added: the ceiling is 1/1, not 1/1 plus float slack.

Historical anchor: A1_1p0 on Qwen3-235B sealed at 29606271552 bytes over
235093634560 weights = 1.007471652 BPW. It is ILLEGAL under this ceiling and this
module rejects it. (It also collapsed 6/6 on the real forward, so the rebudget is
not a technicality being fixed on a working artifact.)
"""
from __future__ import annotations

import json
from fractions import Fraction
from typing import Any

from gravity_potency import parse_rate  # exact rational parsing, no decimals

SCHEMA = "hawking.foundry.one_bit_ceiling.v1"

# The ceiling itself. Exact, and not a tunable.
CEILING = Fraction(1, 1)

# Every component the law names. All ten are required on every ledger.
COMPONENTS: tuple[str, ...] = (
    "indices",                # the packed codes themselves
    "codebooks",              # centroids / dictionaries, amortized however you like, still billed
    "scales",                 # per-row, per-group, per-tensor scales and zero points
    "metadata",               # headers, shapes, dtypes, per-tensor descriptors
    "alignment",              # padding to whatever boundary the format demands
    "protected_islands",      # any tensor or slice kept at a higher rate
    "doctor",                 # recovery / repair bytes shipped with the artifact
    "pass_through_tensors",   # embeddings, norms, router, lm_head, anything left native
    "packaging",              # container, index, manifest, checksums
    "runtime_tables",         # anything the runtime must have resident to decode
)
RESERVE = "metadata_alignment_reserve_bits"
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
        raise IncompleteLedger(f"{name}: declared but unset; undeclared is not zero")
    if isinstance(value, bool):
        raise IncompleteLedger(f"{name}: bool is not a bit count")
    try:
        q = _rate(value)
    except Exception as exc:  # noqa: BLE001 - any parse failure is an incomplete declaration
        raise IncompleteLedger(f"{name}: {exc}") from exc
    if q < 0:
        raise IncompleteLedger(f"{name}: negative bits ({q}); a component may not pay for another")
    return q


class CompleteByteLedger:
    """Itemized, exhaustive bit accounting for one candidate artifact.

    Every one of COMPONENTS plus the explicit reserve must be passed. Zero is a
    legal declaration; omission is not. Unknown keys are refused so nothing can
    hide in a "misc" bucket outside the ten named slots.
    """

    __slots__ = ("bits", "note")

    def __init__(self, *, note: str = "", **components: Any) -> None:
        given = set(components)
        missing = sorted(REQUIRED_FIELDS - given)
        unknown = sorted(given - REQUIRED_FIELDS)
        if missing:
            raise IncompleteLedger(
                "incomplete ledger, undeclared components are NOT zero: "
                + ", ".join(missing)
            )
        if unknown:
            raise IncompleteLedger(
                "unknown ledger components (every bit belongs in a named slot): "
                + ", ".join(unknown)
            )
        self.bits = {k: _exact(k, v) for k, v in components.items()}
        self.note = note

    # ── totals ────────────────────────────────────────────────────────────────
    def complete_bits(self) -> Fraction:
        """Every component plus the declared reserve. Exact."""
        return sum(self.bits.values(), Fraction(0))

    def itemized_bits(self) -> Fraction:
        """The ten components without the reserve headroom."""
        return sum((self.bits[c] for c in COMPONENTS), Fraction(0))

    def complete_bpw(self, original_weight_count: int) -> Fraction:
        w = int(original_weight_count)
        if w <= 0:
            raise IncompleteLedger(f"original_weight_count must be positive, got {w}")
        return self.complete_bits() / w

    def as_dict(self, original_weight_count: int | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": SCHEMA,
            "scope": "whole_model",
            "components": {k: str(v) for k, v in self.bits.items()},
            "itemized_bits": str(self.itemized_bits()),
            "reserve_bits": str(self.bits[RESERVE]),
            "complete_bits": str(self.complete_bits()),
            "note": self.note,
        }
        if original_weight_count is not None:
            bpw = self.complete_bpw(original_weight_count)
            out |= {
                "original_weight_count": int(original_weight_count),
                "complete_bpw_exact": f"{bpw.numerator}/{bpw.denominator}",
                "complete_bpw_float": float(bpw),
                "legal": bpw <= CEILING,
            }
        return out


def assert_complete_bpw_le_one(ledger: CompleteByteLedger, original_weight_count: int) -> dict[str, Any]:
    """Enforce the ceiling. Returns a receipt on pass, raises with the overage on fail."""
    bpw = ledger.complete_bpw(original_weight_count)
    if bpw > CEILING:
        over = bpw - CEILING
        over_bits = ledger.complete_bits() - int(original_weight_count)
        raise CeilingViolation(
            f"one-bit ceiling violated: complete {float(bpw):.9f} BPW "
            f"(exact {bpw.numerator}/{bpw.denominator}) over {int(original_weight_count)} weights; "
            f"overage {float(over):.9f} BPW = {float(over_bits):.0f} bits "
            f"= {float(over_bits) / 8 / 1024 ** 2:.1f} MiB; "
            f"rebudget to <= 1/1, do not raise the ceiling"
        )
    return {
        "schema": SCHEMA,
        "legal": True,
        "complete_bpw_exact": f"{bpw.numerator}/{bpw.denominator}",
        "complete_bpw_float": float(bpw),
        "headroom_bits": str(int(original_weight_count) - ledger.complete_bits()),
        "reserve_bits": str(ledger.bits[RESERVE]),
        "scope": "whole_model",
    }


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

    ledger = spec.get("ledger")
    if isinstance(ledger, dict):
        try:
            ledger = CompleteByteLedger(**ledger)
        except CeilingError as exc:
            return False, [str(exc)]
    if not isinstance(ledger, CompleteByteLedger):
        return False, ["spec has no CompleteByteLedger; a byte plan without an itemized ledger is not a candidate"]

    scope = spec.get("scope", "whole_model")
    if scope != "whole_model":
        reasons.append(f"scope {scope!r}: the ceiling is whole-model only")

    partial = sorted(k for k in spec if k.endswith("_only_bpw"))
    if partial and "reported_bpw" not in spec:
        reasons.append(
            "partial-scope rate present (" + ", ".join(partial)
            + ") with no whole-model reported_bpw; a partial BPW may never stand in for the whole model"
        )

    try:
        bpw = ledger.complete_bpw(spec.get("original_weight_count", 0))
    except CeilingError as exc:
        return False, reasons + [str(exc)]

    for key in partial:
        try:
            claimed = _rate(spec[key])
        except Exception:  # noqa: BLE001 - unparseable partial rate is itself the finding
            reasons.append(f"{key}: unparseable rate {spec[key]!r}")
            continue
        if claimed < bpw:
            reasons.append(
                f"{key}={float(claimed):.6f} understates the whole-model {float(bpw):.6f} BPW; "
                "report the whole-model rate"
            )

    if "reported_bpw" in spec:
        reported = _rate(spec["reported_bpw"])
        if reported != bpw:
            reasons.append(
                f"reported_bpw {float(reported):.9f} != ledger {float(bpw):.9f}; "
                "the reported rate must be the complete whole-model rate"
            )

    if "target_bpw" in spec:
        target = _rate(spec["target_bpw"])
        if target > CEILING:
            reasons.append(
                f"target_bpw {float(target):.4f} > 1: upward bracketing is REJECTED "
                "(no 1.2 anchor, no 1.5, no 2.0, no 3.0, no automatic Escape Receipt)"
            )

    try:
        assert_complete_bpw_le_one(ledger, spec["original_weight_count"])
    except CeilingViolation as exc:
        reasons.append(str(exc))

    return (not reasons), reasons


# ── historical anchor ─────────────────────────────────────────────────────────
# Qwen3-235B A1_1p0, sealed receipt QWEN_GRAVITY/checkpoints/*__A1_1p0.json:
# total_artifact_bytes 29606271552, whole_model_bpw 1.007471652, collapse 6/6.
# The seal records the whole-model TOTAL, not a per-component split, so the split
# below books the entire sealed payload against `indices` and declares the other
# nine slots at their lower bound of zero. Real itemization can only ADD bits, so
# the rejection below holds a fortiori.
A1_1P0_WEIGHTS = 235093634560
A1_1P0_TOTAL_BITS = 29606271552 * 8  # 236850172416


def a1_1p0_ledger() -> CompleteByteLedger:
    zeros = {c: 0 for c in COMPONENTS}
    return CompleteByteLedger(
        **(zeros | {"indices": A1_1P0_TOTAL_BITS}),
        metadata_alignment_reserve_bits=0,
        note="Qwen3-235B A1_1p0 sealed total; per-component split not sealed, other slots are lower bounds",
    )


def a1_1p0_verdict() -> dict[str, Any]:
    legal, reasons = is_legal_candidate({
        "candidate_id": "A1_1p0",
        "parent": "qwen3-235b",
        "original_weight_count": A1_1P0_WEIGHTS,
        "ledger": a1_1p0_ledger(),
        "reported_bpw": f"{A1_1P0_TOTAL_BITS}/{A1_1P0_WEIGHTS}",
    })
    return {"candidate_id": "A1_1p0", "legal": legal, "reasons": reasons}


if __name__ == "__main__":
    print(json.dumps({
        "schema": SCHEMA,
        "ceiling": "1/1",
        "components": COMPONENTS,
        "reserve_field": RESERVE,
        "A1_1p0": a1_1p0_verdict(),
    }, indent=1))
