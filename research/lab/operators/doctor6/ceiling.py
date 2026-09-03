"""Ceiling enforcement for doctor6.

Default target is 1.0 and stays fail-closed via assert_complete_bpw_le_one.
target_bpw in (1.0, 1.5] is legal ONLY with a sealed specialization escape
receipt (issued by seal_with_ceiling). That is not a bypass: the 1.0 one-bit
law is unchanged, the abs hard ceiling stays 1.5, and a missing/tampered
escape still raises the same upward-bracketing refusal.

Raising CEILING itself is forbidden. Automatic unsigned escapes are forbidden.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from lab.operators.eco_common import seal_field, sealed
from lab.operators.one_bit_ceiling import (
    CEILING,
    COMPONENTS,
    RESERVE,
    SCHEMA,
    CeilingViolation,
    CompleteByteLedger,
    assert_complete_bpw_le_one,
)

TARGET_BPW = 1.0
ABS_HARD_CEILING_BPW = 1.5
ABS_HARD_CEILING = Fraction(3, 2)
MIXED_PRECISION_FLOOR_BPW = 1.34  # 1-bit + RHT + outlier (doctor table)
ESCAPE_SCHEMA = "hawking.doctor6.specialization_escape.v1"
SPECIALIZATION_JUSTIFICATION = (
    "Hawking specialization: every organ gets its own quantization bit-width "
    "so the param-weighted average effective-BPW is in [1.34, 1.5]. "
    "The mixed-precision floor is 1.34 (1-bit+RHT+outlier per the doctor "
    "table); a 1.0 allocation target sits below that floor and can never "
    "engage the water-fill. This sealed escape authorizes an operator-"
    "requested allocation target in (1.0, 1.5]. It is not a 1.2/2.0/3.0 "
    "quality anchor, not an automatic Escape Receipt, and does not weaken "
    "the default 1.0 one-bit seal."
)


def ledger_from_slots(
    slots_bits: dict[str, Any],
    *,
    reserve_bits: Any = 0,
    note: str = "doctor6",
) -> CompleteByteLedger:
    """Build a CompleteByteLedger from a partial slots dict (missing → 0)."""
    comps = {c: slots_bits.get(c, 0) for c in COMPONENTS}
    return CompleteByteLedger(
        **comps,
        **{RESERVE: reserve_bits},
        note=note,
    )


def issue_specialization_escape(
    *,
    target_bpw: float,
    justification: str = SPECIALIZATION_JUSTIFICATION,
    note: str = "doctor6",
) -> dict[str, Any]:
    """Issue a sealed escape authorizing an allocation target in (1.0, 1.5].

    The 1.0 one-bit ceiling is not modified. Tampering the receipt invalidates
    the seal (eco_common.sealed). target_bpw outside (1.0, 1.5] is refused.
    """
    t = float(target_bpw)
    if t <= float(CEILING) + 1e-15:
        raise CeilingViolation(
            f"specialization escape is not applicable at target_bpw {t}: "
            "the default 1.0 seal needs no escape"
        )
    if t > ABS_HARD_CEILING_BPW + 1e-12:
        raise CeilingViolation(
            f"target_bpw {t} > {ABS_HARD_CEILING_BPW}: abs hard ceiling; "
            "escape cannot raise this"
        )
    just = str(justification or "").strip()
    if not just:
        raise CeilingViolation("specialization escape requires a non-empty justification")
    body = {
        "schema": ESCAPE_SCHEMA,
        "kind": "specialization_mixed_precision_waterfill",
        "target_bpw": t,
        "one_bit_ceiling_unchanged": float(CEILING),
        "abs_hard_ceiling_bpw": ABS_HARD_CEILING_BPW,
        "mixed_precision_floor_bpw": MIXED_PRECISION_FLOOR_BPW,
        "justification": just,
        "not_a_bypass": True,
        "not_a_quality_anchor": True,
        "not_automatic": True,
        "note": str(note),
    }
    return seal_field(body, "sha256")


def escape_is_sealed(receipt: Any, *, target_bpw: float | None = None) -> bool:
    """True iff receipt is a tamper-evident specialization escape for this target."""
    if not isinstance(receipt, dict):
        return False
    if receipt.get("schema") != ESCAPE_SCHEMA:
        return False
    if receipt.get("not_a_bypass") is not True:
        return False
    if not sealed(receipt, "sha256"):
        return False
    try:
        t = float(receipt["target_bpw"])
    except (KeyError, TypeError, ValueError):
        return False
    if t <= float(CEILING) + 1e-15 or t > ABS_HARD_CEILING_BPW + 1e-12:
        return False
    if target_bpw is not None and abs(t - float(target_bpw)) > 1e-12:
        return False
    if not str(receipt.get("justification") or "").strip():
        return False
    return True


def _upward_refusal(target_bpw: float) -> CeilingViolation:
    return CeilingViolation(
        f"target_bpw {target_bpw} > 1: upward bracketing is REJECTED; "
        "raising the ceiling is not an available move"
    )


def enforce_ceiling(
    slots_bits: dict[str, Any],
    original_weight_count: int,
    *,
    reserve_bits: Any = 0,
    note: str = "doctor6",
    target_bpw: float = TARGET_BPW,
    escape_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the ceiling enforcer. Raises CeilingViolation on overage.

    Default (target_bpw <= 1.0): assert_complete_bpw_le_one. Fail-closed.
    target_bpw in (1.0, 1.5]: requires a sealed specialization escape, then
    fail-closed against the 1.5 abs hard ceiling. Without a valid escape the
    refusal is the same upward-bracketing error as before (not a bypass).
    target_bpw > 1.5: refused even with an escape.
    """
    t = float(target_bpw)
    n = int(original_weight_count)
    ledger = ledger_from_slots(slots_bits, reserve_bits=reserve_bits, note=note)

    if t > ABS_HARD_CEILING_BPW + 1e-12:
        raise CeilingViolation(
            f"target_bpw {t} > {ABS_HARD_CEILING_BPW}: abs hard ceiling; "
            "escape cannot raise this"
        )

    if t <= float(CEILING) + 1e-15:
        # Default path: 1.0 one-bit law. An escape cannot weaken this.
        receipt = assert_complete_bpw_le_one(ledger, n)
        receipt["enforcer_called"] = True
        receipt["target_bpw"] = t
        receipt["escape_applied"] = False
        receipt["abs_hard_ceiling_bpw_not_a_target"] = ABS_HARD_CEILING_BPW
        return receipt

    if not escape_is_sealed(escape_receipt, target_bpw=t):
        raise _upward_refusal(t)

    bpw = ledger.complete_bpw(n)
    if bpw > ABS_HARD_CEILING:
        over = bpw - ABS_HARD_CEILING
        over_bits = ledger.complete_bits() - ABS_HARD_CEILING * n
        raise CeilingViolation(
            f"escaped specialization ceiling {ABS_HARD_CEILING.numerator}/"
            f"{ABS_HARD_CEILING.denominator} violated: complete {float(bpw):.9f} "
            f"BPW (exact {bpw.numerator}/{bpw.denominator}) over {n} weights; "
            f"overage {float(over):.9f} BPW = {float(over_bits):.0f} bits; "
            "rebudget to <= 3/2, do not raise the abs hard ceiling"
        )

    if bpw <= CEILING:
        receipt = assert_complete_bpw_le_one(ledger, n)
        receipt["one_bit_seal_also_holds"] = True
    else:
        receipt = {
            "schema": SCHEMA,
            "legal": True,
            "complete_bpw_exact": f"{bpw.numerator}/{bpw.denominator}",
            "complete_bpw_float": float(bpw),
            "headroom_bits": str(ABS_HARD_CEILING * n - ledger.complete_bits()),
            "reserve_bits": str(ledger.bits[RESERVE]),
            "scope": "whole_model",
            "one_bit_seal_also_holds": False,
            "sealed_against": "3/2",
        }

    receipt["enforcer_called"] = True
    receipt["target_bpw"] = t
    receipt["escape_applied"] = True
    receipt["escape_receipt"] = dict(escape_receipt)
    receipt["abs_hard_ceiling_bpw"] = ABS_HARD_CEILING_BPW
    receipt["abs_hard_ceiling_bpw_not_a_target"] = ABS_HARD_CEILING_BPW
    return receipt


def project_slots_from_expert_payload(
    *,
    mean_expert_payload_bytes: float,
    expert_tensor_count: int,
    non_expert_payload_bytes: int,
    packaging_bytes: int,
    doctor_bytes: int = 0,
    pass_through_bytes: int = 0,
    protected_islands_bytes: int = 0,
    runtime_tables_bytes: int = 0,
) -> dict[str, int]:
    """Map projected byte slots onto the ten named ledger components (in BITS)."""
    expert_payload = int(round(float(mean_expert_payload_bytes) * expert_tensor_count))
    # Expert payload is the indices+codebooks+scales mass for experts.
    return {
        "indices": expert_payload * 8,
        "codebooks": 0,
        "scales": 0,
        "metadata": packaging_bytes * 8 // 4,  # split packaging floor across meta/align/pack
        "alignment": packaging_bytes * 8 // 4,
        "protected_islands": protected_islands_bytes * 8,
        "doctor": doctor_bytes * 8,
        "pass_through_tensors": (pass_through_bytes + non_expert_payload_bytes) * 8,
        "packaging": packaging_bytes * 8 // 2,
        "runtime_tables": runtime_tables_bytes * 8,
    }


def complete_bpw_from_bytes(total_bytes: int, original_weight_count: int) -> float:
    return 8.0 * total_bytes / max(int(original_weight_count), 1)
