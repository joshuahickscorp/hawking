"""doctor6 selftest: ceiling fail-closed, sealed escape, water-fill, compose floor, QAT."""
from __future__ import annotations

from typing import Any

import numpy as np

from lab.operators.doctor6.billing import project_complete_bpw, seal_with_ceiling
from lab.operators.doctor6.ceiling import (
    ABS_HARD_CEILING_BPW,
    enforce_ceiling,
    escape_is_sealed,
    issue_specialization_escape,
    project_slots_from_expert_payload,
)
from lab.operators.doctor6.compose import compose_organ_chain
from lab.operators.doctor6.verify import verify_ground_truth_pair
from lab.operators.eco_common import sealed
from lab.operators.mixed_precision_alloc import allocate, allocate_from_holdout
from lab.operators.one_bit_ceiling import CeilingViolation


def _legal_slots() -> tuple[dict[str, int], int]:
    from lab.operators.doctor6.billing import (
        EXPERT_TENSOR_COUNT,
        SOURCE_WEIGHT_ELEMENTS,
    )

    bill = project_complete_bpw(mean_expert_payload_bytes=120_000.0)
    slots = bill["slots_bytes"]
    bits = project_slots_from_expert_payload(
        mean_expert_payload_bytes=slots["expert_payload"] / EXPERT_TENSOR_COUNT,
        expert_tensor_count=EXPERT_TENSOR_COUNT,
        non_expert_payload_bytes=slots["non_expert_binary_payload"],
        packaging_bytes=slots["packaging_metadata_alignment_runtime_tables"],
        doctor_bytes=slots.get("doctor", 0),
        pass_through_bytes=slots.get("pass_through_tensors", 0),
        protected_islands_bytes=slots.get("protected_islands", 0),
    )
    return bits, SOURCE_WEIGHT_ELEMENTS


def check_ceiling() -> dict[str, Any]:
    bill_over = project_complete_bpw(mean_expert_payload_bytes=1e9)
    legal = project_complete_bpw(mean_expert_payload_bytes=120_000.0)
    msg = ""

    closed_1_0 = False
    try:
        seal_with_ceiling(bill_over, target_bpw=1.0, note="selftest")
    except CeilingViolation as exc:
        closed_1_0 = True
        msg = str(exc)

    legal_1_0 = False
    try:
        seal_with_ceiling(legal, target_bpw=1.0, note="selftest_legal")
        legal_1_0 = True
    except CeilingViolation as exc:
        msg += f" | legal failed: {exc}"

    slots, n = _legal_slots()
    no_escape_1_5 = False
    try:
        enforce_ceiling(slots, n, target_bpw=1.5, note="selftest_no_escape")
    except CeilingViolation as exc:
        no_escape_1_5 = "upward bracketing is REJECTED" in str(exc)

    sealed_ok = False
    escape_present = False
    try:
        out = seal_with_ceiling(legal, target_bpw=1.5, note="selftest_escape")
        esc = out.get("escape_receipt")
        escape_present = bool(esc) and sealed(esc, "sha256") and escape_is_sealed(
            esc, target_bpw=1.5
        )
        sealed_ok = bool(out.get("legal")) and bool(out.get("escape_applied"))
    except CeilingViolation as exc:
        msg += f" | sealed escape on legal bill failed: {exc}"

    over_1_5_closed = False
    try:
        seal_with_ceiling(bill_over, target_bpw=1.5, note="selftest_over_15")
    except CeilingViolation:
        over_1_5_closed = True

    tamper_closed = False
    try:
        esc = issue_specialization_escape(target_bpw=1.5, justification="unit")
        esc["justification"] = "tampered"
        enforce_ceiling(slots, n, target_bpw=1.5, escape_receipt=esc)
    except CeilingViolation as exc:
        tamper_closed = "upward bracketing is REJECTED" in str(exc)

    above_abs = False
    try:
        enforce_ceiling(slots, n, target_bpw=1.6, note="selftest_abs")
    except CeilingViolation as exc:
        above_abs = "abs hard ceiling" in str(exc)

    return {
        "over_ceiling_fail_closed": closed_1_0,
        "legal_bill_passes": legal_1_0,
        "target_1_5_without_escape_fail_closed": no_escape_1_5,
        "target_1_5_escape_sealed": sealed_ok and escape_present,
        "over_1_5_still_fail_closed": over_1_5_closed,
        "tampered_escape_fail_closed": tamper_closed,
        "above_abs_hard_ceiling_refused": above_abs,
        "abs_hard_ceiling_bpw": ABS_HARD_CEILING_BPW,
        "over_ceiling_message": msg[:500],
        "ok": bool(
            closed_1_0
            and legal_1_0
            and no_escape_1_5
            and sealed_ok
            and escape_present
            and over_1_5_closed
            and tamper_closed
            and above_abs
        ),
    }


def check_allocator() -> dict[str, Any]:
    # 16 equal-size organs: 1-bit floor is 1.34, slack to 1.5 is 0.16,
    # one 1→2 upgrade costs 1.0/16 = 0.0625, so the water-fill can spend.
    organs = []
    for i in range(14):
        organs.append(
            {
                "name": f"easy{i}.gate_proj.weight",
                "elems": 1000,
                "holdout_cosine": 0.995,
                "component": "gate_proj",
                "layer": i,
                "band": 0,
            }
        )
    organs.append(
        {
            "name": "mid.up_proj.weight",
            "elems": 1000,
            "holdout_cosine": 0.90,
            "component": "up_proj",
            "layer": 18,
            "band": 1,
        }
    )
    organs.append(
        {
            "name": "hard.down_proj.weight",
            "elems": 1000,
            "holdout_cosine": 0.64,
            "component": "down_proj",
            "layer": 37,
            "band": 3,
        }
    )
    out = allocate_from_holdout(
        organs, bits_set=(1, 2, 3, 4), target_bpw=1.5, layer_target=0.9857
    )
    alloc = out["allocation"]
    avg = float(out["achieved_avg_eff_bpw"])
    hard_gt_easy = int(alloc["hard.down_proj.weight"]) > int(
        alloc["easy0.gate_proj.weight"]
    )
    within = bool(avg <= 1.5 + 1e-9)
    rows = []
    for o in organs:
        deficit = max(0.9857 - float(o["holdout_cosine"]), 0.0) + 1e-6
        rows.append(
            {
                "name": o["name"],
                "elems": o["elems"],
                "holdout_cosine": o["holdout_cosine"],
                "sens": {b: deficit * (5 - b) for b in (1, 2, 3, 4)},
            }
        )
    _g_alloc, g_avg = allocate(rows, (1, 2, 3, 4), 1.5)
    return {
        "allocator_invoked": True,
        "within_budget": within,
        "achieved_avg_eff_bpw": avg,
        "hard_bits": int(alloc["hard.down_proj.weight"]),
        "easy_bits": int(alloc["easy0.gate_proj.weight"]),
        "hard_gets_more_bits": hard_gt_easy,
        "greedy_within_budget": bool(g_avg <= 1.5 + 1e-9),
        "ok": bool(within and hard_gt_easy and g_avg <= 1.5 + 1e-9),
    }


def check_compose_incumbent_floor() -> dict[str, Any]:
    rng = np.random.default_rng(1)
    W = rng.standard_normal((16, 32), dtype=np.float32) * 0.1
    # Tiny fit set so act-SVD overfits and holdout is worse than binary.
    X_fit = rng.standard_normal((6, 32), dtype=np.float32)
    X_hold = rng.standard_normal((24, 32), dtype=np.float32)
    result = compose_organ_chain(
        W=W,
        X_fit=X_fit,
        X_hold=X_hold,
        organ_key="synth.gate_proj.weight",
        component="gate_proj",
        sensitivity=0.2,
        seed=0xD0C70A,
        device="cpu",
        qat_steps=4,
        target_cos=0.9857,
        measure_all=True,
    )
    below = float(result["prescribed_cosine"]) + 1e-12 < float(result["incumbent_cosine"])
    return {
        "prescribed_cosine": float(result["prescribed_cosine"]),
        "incumbent_cosine": float(result["incumbent_cosine"]),
        "chain": list(result["chain"]),
        "n_prescribed_below_incumbent": int(below),
        "ok": (not below),
    }


def check_qat() -> dict[str, Any]:
    from lab.operators.doctor6.rungs import selfcheck_qat_fit_ge_calib

    try:
        rec = selfcheck_qat_fit_ge_calib(device="cpu")
        return rec
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}


def run_selftest() -> dict[str, Any]:
    gt = verify_ground_truth_pair()
    ceiling = check_ceiling()
    allocator = check_allocator()
    compose = check_compose_incumbent_floor()
    qat = check_qat()
    all_pass = bool(
        gt["all_pass"]
        and ceiling["ok"]
        and allocator["ok"]
        and compose["ok"]
        and qat.get("ok")
    )
    return {
        "coherence_ground_truth": gt,
        "ceiling": {
            "over_ceiling_fail_closed": ceiling["over_ceiling_fail_closed"],
            "legal_bill_passes": ceiling["legal_bill_passes"],
            "over_ceiling_message": ceiling["over_ceiling_message"],
            **{k: v for k, v in ceiling.items() if k != "over_ceiling_message"},
        },
        "allocator": allocator,
        "compose": compose,
        "qat": qat,
        "all_pass": all_pass,
    }
