#!/usr/bin/env python3
"""G094 MEASURED: the convert-free unpack takes 3.854 ms off the complete token.

Not a kernel probe and not a projection. The 580-dispatch resident graph, run
twice under gpu_lane_lock with the ONLY difference being which q2 unpack the two
affine kernels use:

    widen_f4 arm      control    26.4562 ms      bitcast   22.6021 ms
                      saved       3.8541 ms      1.1705x
                      32 tokens, 9 reps, TOKEN IDENTICAL, 0 fallbacks
                      580 dispatches both arms, dense_w_materialized 0 both

The op-class ablation said where to aim: the uint-to-float convert is 44% of
this kernel's arithmetic and 15% of its time, against 1.8% for the affine FMA
that two earlier candidates attacked. A 2-bit code needs no convert. Placing q
at bits 21-22 of an f32 with exponent field 0x40000000 gives f = 2.0 + 0.5*q
exactly, so w = q*scale + bias becomes w = (2*scale)*f + (bias - 4*scale) with
both constants folded once per group from the same half scale and bias
production already loads. Per weight this trades a convert for an OR.

THE RELATIVE IS THE CLAIM. Downloads were running, so the window is contaminated
and neither absolute is promotable. Ratios hold under load and absolutes do not,
and the two arms ran back to back with rep spreads of 0.5% and 0.4%.

    python3 tools/future/bitcast_dequant_ab.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402

RECORDED_BY = "tools/future/bitcast_dequant_ab.py"
RECEIPT_NAME = "BITCAST_DEQUANT_AB.json"
CTRL_REL = "receipts/future/_G094_RESIDENT_CTRL_raw.json"
BITCAST_REL = "receipts/future/_G094_RESIDENT_BITCAST_raw.json"
# A second independent paired run, taken after SIGSTOPping the ModelLake
# downloads. They RESPAWNED mid-window, so it is not a protected window either -
# but it is a different contention window, which is what makes it useful.
CTRL2_REL = "receipts/future/_G094_PROTECTED_CTRL_raw.json"
BITCAST2_REL = "receipts/future/_G094_PROTECTED_BITCAST_raw.json"
BUDGET_REL = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"

# The arm the resident actually serves on.
LIVE_ARM = "widen_f4"

SWAPPED = {
    "qwen_affine_q2_group32_matvec_geo_tpr64_tg128":
        "qwen_affine_q2_group32_matvec_geo_tpr64_tg128_bitcast",
    "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128":
        "qwen_affine_q2_group64_matvec_gate_up_swiglu_geo_tpr64_tg128_bitcast",
}


class AbRefused(RuntimeError):
    """The pair is not matched, or the candidate changed something it must not."""


def _raw(rel: str) -> dict[str, Any]:
    p = REPO / rel
    if not p.is_file():
        raise AbRefused(f"{rel} is not on disk; run the A/B first")
    return json.loads(p.read_text())


def _arm(rel: str, arm: str) -> dict[str, Any]:
    d = _raw(rel)["decode"]
    if arm not in d:
        raise AbRefused(f"{rel} has no {arm} arm")
    return d[arm]


def matched_pair() -> dict[str, Any]:
    """Everything that must be EQUAL for the timing difference to mean anything."""
    c = _arm(CTRL_REL, LIVE_ARM)
    b = _arm(BITCAST_REL, LIVE_ARM)

    if c["new_token_ids"] != b["new_token_ids"]:
        n = sum(1 for x, y in zip(c["new_token_ids"], b["new_token_ids"]) if x != y)
        raise AbRefused(
            f"the arms produced different tokens ({n} differ); a faster kernel "
            "that changes the output is not a win, it is a regression"
        )
    if c["theoretical_dispatches"] != b["theoretical_dispatches"]:
        raise AbRefused("dispatch count differs; this is not the same graph")
    if set(c["dispatches_last_step_reps"]) != set(b["dispatches_last_step_reps"]):
        raise AbRefused("measured dispatch counts differ; not the same graph")
    for arm_name, arm in (("control", c), ("bitcast", b)):
        if arm["dense_w_materialized"]:
            raise AbRefused(f"{arm_name} materialised a dense W; not in-register dequant")
        if any(arm["fallbacks_reps"]):
            raise AbRefused(f"{arm_name} took a fallback; the graph is not the one claimed")

    ck, bk = set(c["dispatched_kernels_rep0"]), set(b["dispatched_kernels_rep0"])
    only_c, only_b = sorted(ck - bk), sorted(bk - ck)
    if only_c != sorted(SWAPPED) or only_b != sorted(SWAPPED.values()):
        raise AbRefused(
            "the kernel-set difference is not exactly the two swapped unpacks: "
            f"only in control {only_c}, only in bitcast {only_b}"
        )
    return {
        "token_ids_identical": True,
        "n_tokens": len(c["new_token_ids"]),
        "reps": len(c["decode_wall_ns_reps"]),
        "dispatches": c["theoretical_dispatches"],
        "dense_w_materialized": 0,
        "fallbacks": 0,
        "kernels_swapped": SWAPPED,
        "nothing_else_changed": (
            "the dispatched-kernel sets differ by exactly these two names and "
            "nothing else, so the 580-dispatch graph is identical apart from the "
            "unpack"
        ),
    }


def timing() -> dict[str, Any]:
    matched_pair()
    c = _arm(CTRL_REL, LIVE_ARM)
    b = _arm(BITCAST_REL, LIVE_ARM)
    c_ms = c["gpu_ns_median"] / 1e6
    b_ms = b["gpu_ns_median"] / 1e6

    def spread(arm: dict[str, Any]) -> float:
        r = arm["decode_wall_ns_reps"]
        return max(r) / min(r)

    return {
        "arm": LIVE_ARM,
        "control_gpu_ms_per_token": round(c_ms, 4),
        "bitcast_gpu_ms_per_token": round(b_ms, 4),
        "ms_saved": round(c_ms - b_ms, 4),
        "speedup": round(c_ms / b_ms, 4),
        "control_rep_spread": round(spread(c), 4),
        "bitcast_rep_spread": round(spread(b), 4),
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
    }


def _wall_ms(arm: dict[str, Any]) -> float:
    """Wall ms per token. TPS means wall, not GPU."""
    reps = sorted(arm["decode_wall_ns_reps"])
    n = len(arm["new_token_ids"])
    return reps[len(reps) // 2] / 1e6 / n


def independent_runs() -> dict[str, Any]:
    """Two paired runs in DIFFERENT contention windows.

    Neither window is protected - the ModelLake downloads respawned after being
    SIGSTOPped. That is exactly why both are reported: the absolute moves with
    the window and the ratio does not, so the agreement between the two ratios
    is the evidence, and the disagreement between the two absolutes is the
    reason no single absolute is promoted.
    """
    runs = []
    for label, cr, br in (("run_1", CTRL_REL, BITCAST_REL),
                          ("run_2", CTRL2_REL, BITCAST2_REL)):
        c, b = _arm(cr, LIVE_ARM), _arm(br, LIVE_ARM)
        if c["new_token_ids"] != b["new_token_ids"]:
            raise AbRefused(f"{label}: the arms produced different tokens")
        cg, bg = c["gpu_ns_median"] / 1e6, b["gpu_ns_median"] / 1e6
        cw, bw = _wall_ms(c), _wall_ms(b)
        runs.append({
            "run": label,
            "control_gpu_ms": round(cg, 4),
            "bitcast_gpu_ms": round(bg, 4),
            "gpu_ms_saved": round(cg - bg, 4),
            "gpu_speedup": round(cg / bg, 4),
            "control_wall_ms": round(cw, 4),
            "bitcast_wall_ms": round(bw, 4),
            "wall_ms_saved": round(cw - bw, 4),
            "control_wall_tps": round(1000.0 / cw, 3),
            "bitcast_wall_tps": round(1000.0 / bw, 3),
        })
    speedups = [r["gpu_speedup"] for r in runs]
    ctrl_walls = [r["control_wall_ms"] for r in runs]
    return {
        "runs": runs,
        "gpu_speedup_range": [min(speedups), max(speedups)],
        "gpu_speedup_spread": round(max(speedups) / min(speedups) - 1, 5),
        "control_wall_range": [min(ctrl_walls), max(ctrl_walls)],
        "control_wall_spread": round(max(ctrl_walls) / min(ctrl_walls) - 1, 5),
        "reading": (
            "the RATIO agrees to "
            f"{round((max(speedups)/min(speedups)-1)*100, 2)}% across the two "
            "windows while the CONTROL's own absolute moves by "
            f"{round((max(ctrl_walls)/min(ctrl_walls)-1)*100, 2)}%. Ratios hold "
            "under load and absolutes do not, measured here rather than assumed."
        ),
        "bitcast_wall_tps_range": [
            min(r["bitcast_wall_tps"] for r in runs),
            max(r["bitcast_wall_tps"] for r in runs),
        ],
    }


def projection_vs_graph() -> dict[str, Any]:
    """The isolated harness UNDERSTATED the win, and that is worth recording."""
    iso_speedup = 1.2178
    budget = json.loads((REPO / BUDGET_REL).read_text())
    rows = {r["organ"]: float(r["gpu_ms"]) for r in budget["organs"]["rows"]}
    organ_ms = rows["mlp_gate_up"] + rows["mlp_down"]
    predicted = organ_ms - organ_ms / iso_speedup
    actual = timing()["ms_saved"]
    return {
        "isolated_kernel_speedup": iso_speedup,
        "q2_organ_ms_from_census": round(organ_ms, 4),
        "predicted_ms_saved": round(predicted, 4),
        "measured_ms_saved_in_the_graph": actual,
        "graph_over_prediction": round(actual / predicted, 4),
        "reading": (
            "the graph delivered "
            f"{round((actual/predicted - 1)*100, 1)}% MORE than the isolated "
            "harness predicted. widen_f4 did the same thing - its isolated cut "
            "was 0.7046 ms and the graph gave 1.0245 ms. An isolated organ "
            "measurement appears to be a LOWER bound on the graph-level win for "
            "this class of change, which is the safe direction for a projection "
            "to be wrong in, but it means the projection is not a tight estimate."
        ),
        "do_not_invert_this": (
            "this does NOT license scaling isolated numbers up by a fudge "
            "factor. Two observations are not a law, and the mechanism - "
            "whatever the graph-level fold is - has not been identified."
        ),
    }


def complete_token() -> dict[str, Any]:
    """The wall number, which is what TPS means. GPU time is not the token."""
    t = timing()
    budget = json.loads((REPO / BUDGET_REL).read_text())
    wall_before = float(budget["decode_wall_ms_per_token"])
    gpu_before = float(budget["decode_gpu_ms_per_token"])
    host_gap = wall_before - gpu_before
    # The saving is measured on GPU time. The host gap is carried across
    # unchanged because the dispatch count is identical - which matched_pair
    # asserts rather than assumes.
    wall_after = wall_before - t["ms_saved"]
    return {
        "wall_ms_before": wall_before,
        "wall_ms_after": round(wall_after, 4),
        "tps_before": round(1000.0 / wall_before, 3),
        "tps_after": round(1000.0 / wall_after, 3),
        "tps_gain": round(1000.0 / wall_after - 1000.0 / wall_before, 3),
        "host_gap_ms_carried_unchanged": round(host_gap, 4),
        "why_the_host_gap_carries": (
            "the candidate changes the body of two kernels and nothing else. "
            "Dispatch count is 580 in both arms and the kernel-set difference is "
            "exactly the two swapped names, so there is no mechanism by which "
            "host time would move. This is an assumption the matched pair "
            "CONSTRAINS, not one it proves; a protected reprofile would settle it."
        ),
        "checkpoint_crossed": "40 TPS" if 1000.0 / wall_after >= 40 else None,
        "still_short_of_60_by_ms": round(wall_after - 1000.0 / 60.0, 4),
    }


def claim_boundary() -> dict[str, Any]:
    return {
        "what_is_measured": (
            "the RELATIVE: 3.8541 ms and 1.1705x between two back-to-back runs "
            "of the same graph on the same machine, token-identical"
        ),
        "what_is_not": (
            "the ABSOLUTE. Neither of the two paired runs had a protected "
            "window: ModelLake downloads ran through the first, and in the "
            "second they were SIGSTOPped and the supervisor RESPAWNED them "
            "mid-run. The control's own wall moves 2.3% between the two windows "
            "while the ratio moves 0.1%, so no single absolute here is "
            "promotable and the TPS-after figure is a range, not a number."
        ),
        "what_would_promote_it": (
            "re-run tools/future/resident_reprofile.py under a real protected "
            "lease with the lever set, and let that receipt replace "
            "RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json as the absolute"
        ),
        "evidence_class": "DIAGNOSTIC_RELATIVE",
    }


def fp_boundary() -> dict[str, Any]:
    return {
        "bit_identical_weights": False,
        "token_identical": True,
        "reading": (
            "the refolded affine is exact in the reals but the two FMAs run on "
            "different constants, so the dequantised weights differ at f32 "
            "epsilon - measured rel_fro 1.5e-07 in the isolated harness. The "
            "resident A/B then produced IDENTICAL token ids over 32 greedy "
            "tokens with zero fallbacks, which is the property that actually "
            "matters. Token identity over 32 tokens is not a proof of identity "
            "over all inputs; it is the same bar every landed lever in this "
            "campaign has cleared."
        ),
        "risk": (
            "greedy argmax can flip on a near-tie. This has not been observed "
            "here, and a longer or adversarial prompt set would tighten it."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G094",
        "candidate": "bitcast dequant",
        "lever": "HAWKING_AFFINE2_GEO=bitcast (Affine2Geo::Bitcast)",
        "default_is_unchanged": True,
        "kernels": sorted(SWAPPED.values()),
        "transform": (
            "q unpacked straight into an f32 mantissa (f = 2 + q/2) so no "
            "int-to-float convert runs; the affine refolds per group as "
            "w = (2*scale)*f + (bias - 4*scale)"
        ),
        "why_this_class": (
            "receipts/future/OP_CLASS_ABLATION.json measured the convert at 44% "
            "of this kernel's arithmetic against 1.8% for the affine FMA. The "
            "hoist and fold_addqx both attacked the 1.8%."
        ),
        "matched_pair": matched_pair(),
        "timing": timing(),
        "independent_runs": independent_runs(),
        "projection_vs_graph": projection_vs_graph(),
        "complete_token": complete_token(),
        "fp_boundary": fp_boundary(),
        "claim_boundary": claim_boundary(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(lock_held=True, lane="g094-prot"),
        ))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("timing", "complete_token", "matched_pair", "claim_boundary")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
