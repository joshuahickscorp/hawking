"""Ceiling enforcement for doctor6.

Always call one_bit_ceiling.assert_complete_bpw_le_one before sealing.
Target <= 1.0; absolute hard ceiling 1.5 is NOT an available move for the target.
Raising the ceiling is forbidden.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any

from lab.operators.one_bit_ceiling import (
    CEILING,
    COMPONENTS,
    RESERVE,
    CeilingViolation,
    CompleteByteLedger,
    assert_complete_bpw_le_one,
)

TARGET_BPW = 1.0
ABS_HARD_CEILING_BPW = 1.5  # informational only; never the seal target


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


def enforce_ceiling(
    slots_bits: dict[str, Any],
    original_weight_count: int,
    *,
    reserve_bits: Any = 0,
    note: str = "doctor6",
    target_bpw: float = TARGET_BPW,
) -> dict[str, Any]:
    """Call the ceiling enforcer. Raises CeilingViolation on overage.

    Also refuses target_bpw > 1.0 (upward bracketing is not an available move).
    """
    if float(target_bpw) > float(CEILING) + 1e-15:
        raise CeilingViolation(
            f"target_bpw {target_bpw} > 1: upward bracketing is REJECTED; "
            "raising the ceiling is not an available move"
        )
    ledger = ledger_from_slots(slots_bits, reserve_bits=reserve_bits, note=note)
    receipt = assert_complete_bpw_le_one(ledger, int(original_weight_count))
    receipt["enforcer_called"] = True
    receipt["target_bpw"] = float(target_bpw)
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
