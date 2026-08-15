"""Whole-model BPW projection + optional gravity allocator when the ceiling binds."""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from lab.operators.doctor6.ceiling import (
    SPECIALIZATION_JUSTIFICATION,
    complete_bpw_from_bytes,
    enforce_ceiling,
    issue_specialization_escape,
    project_slots_from_expert_payload,
)

# Q30 admitted binary complete inventory (campaign sealed numbers).
SOURCE_WEIGHT_ELEMENTS = 30_532_122_624
NON_EXPERT_BINARY_PAYLOAD_BYTES = 216_732_884
EXPERT_ELEMENTS = 28_991_029_248
EXPERT_TENSOR_COUNT = 48 * 128 * 3  # 18432
MANIFEST_PACKAGING_BYTES_FLOOR = 19_726_976


def project_complete_bpw(
    *,
    mean_expert_payload_bytes: float,
    doctor_bytes: int = 0,
    extra_runtime_table_bytes: int = 0,
    pass_through_bytes: int = 0,
) -> dict[str, Any]:
    expert_payload = float(mean_expert_payload_bytes) * EXPERT_TENSOR_COUNT
    slots_bytes = {
        "expert_payload": int(round(expert_payload)),
        "non_expert_binary_payload": int(NON_EXPERT_BINARY_PAYLOAD_BYTES),
        "protected_islands": 0,
        "doctor": int(doctor_bytes),
        "pass_through_tensors": int(pass_through_bytes),
        "packaging_metadata_alignment_runtime_tables": int(
            MANIFEST_PACKAGING_BYTES_FLOOR + extra_runtime_table_bytes
        ),
    }
    total_bytes = sum(slots_bytes.values())
    complete_bpw = complete_bpw_from_bytes(total_bytes, SOURCE_WEIGHT_ELEMENTS)
    expert_local_bpw = (
        8.0 * float(mean_expert_payload_bytes) * EXPERT_TENSOR_COUNT / EXPERT_ELEMENTS
    )
    return {
        "slots_bytes": slots_bytes,
        "all_required_weight_artifact_bytes": int(total_bytes),
        "source_weight_elements": SOURCE_WEIGHT_ELEMENTS,
        "expert_local_bpw": float(expert_local_bpw),
        "complete_physical_bpw": float(complete_bpw),
        "under_1_0": bool(complete_bpw <= 1.0 + 1e-12),
        "under_abs_ceiling_1_5": bool(complete_bpw <= 1.5 + 1e-12),
        "note": (
            "Expert payload extrapolated from sample mean; non-experts stay "
            "binary HQ30G1B1; packaging floor from admitted binary ledger."
        ),
    }


def seal_with_ceiling(
    bill: dict[str, Any],
    *,
    target_bpw: float = 1.0,
    note: str = "doctor6",
    escape_receipt: dict[str, Any] | None = None,
    justification: str | None = None,
) -> dict[str, Any]:
    """Convert projected bill into ledger bits and enforce the ceiling.

    target_bpw <= 1.0: the one-bit seal (fail-closed).
    1.0 < target_bpw <= 1.5: issue (or reuse) a sealed specialization escape,
    then fail-closed against the 1.5 abs hard ceiling. This is the sanctioned
    path for --target-bpw 1.5, not a bypass of assert_complete_bpw_le_one.
    """
    slots_bytes = bill["slots_bytes"]
    slots_bits = project_slots_from_expert_payload(
        mean_expert_payload_bytes=slots_bytes["expert_payload"] / EXPERT_TENSOR_COUNT,
        expert_tensor_count=EXPERT_TENSOR_COUNT,
        non_expert_payload_bytes=slots_bytes["non_expert_binary_payload"],
        packaging_bytes=slots_bytes["packaging_metadata_alignment_runtime_tables"],
        doctor_bytes=slots_bytes.get("doctor", 0),
        pass_through_bytes=slots_bytes.get("pass_through_tensors", 0),
        protected_islands_bytes=slots_bytes.get("protected_islands", 0),
    )
    issued_escape = escape_receipt
    if float(target_bpw) > 1.0 + 1e-15 and issued_escape is None:
        issued_escape = issue_specialization_escape(
            target_bpw=target_bpw,
            justification=justification or SPECIALIZATION_JUSTIFICATION,
            note=note,
        )
    # enforce_ceiling raises CeilingViolation on fail (1.0 default, or 1.5
    # under a valid sealed escape, or missing/tampered escape).
    receipt = enforce_ceiling(
        slots_bits,
        SOURCE_WEIGHT_ELEMENTS,
        reserve_bits=0,
        note=note,
        target_bpw=target_bpw,
        escape_receipt=issued_escape,
    )
    return {
        "enforcer_called": True,
        "receipt": receipt,
        "escape_receipt": receipt.get("escape_receipt") or issued_escape,
        "escape_applied": bool(receipt.get("escape_applied")),
        "slots_bits": {k: int(v) for k, v in slots_bits.items()},
        "complete_physical_bpw": bill["complete_physical_bpw"],
        "legal": bool(receipt.get("legal")),
    }


def maybe_run_gravity_allocator(
    bill: dict[str, Any],
    *,
    target_bpw: float = 1.0,
    organ_sensitivities: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Wire gravity_global_allocator, gated on the ceiling actually binding.

    Today the budget is slack (complete ~0.9 vs 1.5 abs / 1.0 target) and
    value-density ordering is inert. When projected complete BPW would exceed
    target_bpw, run greedy_allocate to rebudget.
    """
    complete = float(bill["complete_physical_bpw"])
    binds = complete > float(target_bpw) + 1e-12
    out: dict[str, Any] = {
        "ceiling_binds": binds,
        "complete_physical_bpw": complete,
        "target_bpw": float(target_bpw),
        "allocator_invoked": False,
        "reason": (
            "ceiling binds — invoking marginal-utility allocator"
            if binds
            else (
                "budget slack: complete_bpw <= target; value-density ordering "
                "inert; allocator gated off (correct until ceiling binds)"
            )
        ),
    }
    if not binds:
        return out

    # Import allocator from tools/condense (restored).
    repo = Path(__file__).resolve().parents[3]
    condense = repo / "tools" / "condense"
    if str(condense) not in sys.path:
        sys.path.insert(0, str(condense))
    try:
        import gravity_global_allocator as gga  # type: ignore
    except Exception as exc:  # noqa: BLE001
        out["allocator_invoked"] = False
        out["error"] = f"allocator import failed: {exc}"
        return out

    sens = organ_sensitivities or {
        "expert_gate": 0.55,
        "expert_up": 0.55,
        "expert_down": 0.75,  # discriminating organ
        "attn": 0.4,
        "embed": 0.3,
    }
    # Minimal organ set for rebudget demonstration under binding ceiling.
    organs = []
    for name, param_count, s in (
        ("expert_gate", EXPERT_ELEMENTS // 3, sens.get("expert_gate", 0.55)),
        ("expert_up", EXPERT_ELEMENTS // 3, sens.get("expert_up", 0.55)),
        ("expert_down", EXPERT_ELEMENTS // 3, sens.get("expert_down", 0.75)),
        ("non_expert", SOURCE_WEIGHT_ELEMENTS - EXPERT_ELEMENTS, sens.get("attn", 0.4)),
    ):
        organs.append(
            gga.Organ(
                name=name,
                param_count=int(param_count),
                source_precision="bf16",
                sensitivity=float(s),
                activation_stat=0.5,
                router_or_expert_frequency=1.0 if name.startswith("expert") else 1.0,
                representation_options=[
                    gga.REP_PRODUCT_QUANT,
                    gga.REP_PQ_DOCTOR_LOWRANK,
                    gga.REP_SOURCE_NATIVE,
                ],
                doctor_reachable=name.startswith("expert"),
                min_protection_bpw=Fraction(1, 4),
                doctor_bpw=Fraction(0),
                importance_weight=float(s),
            )
        )
    budget_bytes = int(target_bpw * SOURCE_WEIGHT_ELEMENTS / 8.0)
    try:
        alloc = gga.greedy_allocate(organs, budget_bytes)
        out["allocator_invoked"] = True
        out["allocation"] = alloc.to_dict()
    except Exception as exc:  # noqa: BLE001
        out["allocator_invoked"] = True
        out["error"] = str(exc)
    return out
