#!/usr/bin/env python3
"""G094: the same convert-free unpack on q4, measured BIT-IDENTICAL at 1.1444x.

The q2 bitcast took 3.854 ms off the token. The scar that came with it says the
op-class split is kernel-specific and must be measured before the target is
chosen, so this does not assume q4 behaves the same - it measures it.

    production       599.9 GB/s   1.0000x
    bitcast          686.5        1.1444x    and BIT-IDENTICAL
    arm_a_stripped   935.8        1.5599x    all arithmetic removed

A nibble at bits 19-22 of an f32 with exponent field 0x40000000 gives exactly
f = 2.0 + q/8, so q = 8*(f-2) and production's (q - 8)*scale becomes
(8*scale)*f + (-24*scale), folded once per group. That removes the
int-to-float convert AND the -8 zero point.

WHY IT IS BIT-IDENTICAL AND THE Q2 VERSION IS NOT. Production computes
`sum += float(int(nibble) - 8) * scale * x`, and Metal contracts that to an FMA
by default, so both paths run the same instruction on values that are exactly
representable in f32. The q2 kernel's refold changes which constants the FMAs
carry, so it lands at f32 epsilon instead.

THE FIRST BUILD WAS FAST AND WRONG, AGAIN. It extracted the nibble by
pre-shifting the whole word - (packed << 19) >> 8i - which works for the q2
kernel's 16-bit packed word and DISCARDS BITS 13 AND UP of q4's 32-bit word of
eight nibbles. It measured 1.40x at rel_fro 0.877. The correct extraction is
one mask then one shift, and it measures 1.1444x.

    python3 tools/future/q4_bitcast_ab.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402

RECORDED_BY = "tools/future/q4_bitcast_ab.py"
RECEIPT_NAME = "Q4_BITCAST_AB.json"
RAW_REL = "receipts/future/_G094_Q4_BITCAST_raw.json"
# The resident A/B. Both arms have the q2 bitcast ON, so the difference is the
# q4 unpack alone and nothing else.
RES_ON_REL = "receipts/future/_G094_Q4RES_BOTH_raw.json"
RES_OFF_REL = "receipts/future/_G094_Q4RES_Q2ONLY_raw.json"
LIVE_ARM = "widen_f4"
BUDGET_REL = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"

STEADY_MAX_SPREAD = 1.10
# The organs the uniform-q4 matvecs run in. lm_head is EXCLUDED: it uses a
# different fused kernel that this candidate does not touch. DeltaNet is
# counted by its IN-PROJECTION only - the state update, rearrange and gated norm
# are different kernels, and counting the whole 5.5971 ms organ would overstate
# this projection by about 0.25 ms.
Q4_ORGANS = ("q4_remainder", "gqa_attention")
DN_INPROJ_MS = 3.5885
DN_INPROJ_SOURCE = (
    "receipts/future/_G094_RESIDENT_CTRL_raw.json isolated_components.dn_inproj"
)


class Q4Refused(RuntimeError):
    """An arm is missing, unsteady, or did not compute the right answer."""


def _raw() -> dict[str, Any]:
    p = REPO / RAW_REL
    if not p.is_file():
        raise Q4Refused(f"{RAW_REL} is not on disk; run the probe first")
    return json.loads(p.read_text())


def arms() -> dict[str, dict[str, Any]]:
    dn = _raw()["deltanet"]
    out = {}
    for name in ("production", "bitcast", "arm_a_stripped"):
        if name not in dn:
            raise Q4Refused(f"arm {name} is absent; not a matched comparison")
        spread = float(dn[name].get("rep_spread", 99.0))
        if spread > STEADY_MAX_SPREAD:
            raise Q4Refused(
                f"arm {name} has rep spread {spread:.4f} > {STEADY_MAX_SPREAD}; "
                "not in steady state"
            )
        out[name] = dn[name]
    return out


def measured() -> dict[str, Any]:
    a = arms()
    prod = float(a["production"]["effective_gb_s"])
    bc = float(a["bitcast"]["effective_gb_s"])
    arm_a = float(a["arm_a_stripped"]["effective_gb_s"])
    cmp_row = a["bitcast"]["output_compare"]
    if not bool(cmp_row["bit_identical"]):
        raise Q4Refused(
            "this receipt claims bit-identity in its own headline; the arm is "
            f"not bit-identical (rel_fro {cmp_row['rel_fro']:.3e}) so the claim "
            "must be rewritten before it is published"
        )
    return {
        "production_gb_s": prod,
        "bitcast_gb_s": bc,
        "arm_a_stripped_gb_s": arm_a,
        "speedup": round(bc / prod, 4),
        "arm_a_over_production": round(arm_a / prod, 4),
        "span_recovered": round((bc - prod) / (arm_a - prod), 4),
        "bit_identical": True,
        "rel_fro": float(cmp_row["rel_fro"]),
        "max_abs_err": float(cmp_row["max_abs_err"]),
        "loadavg": _raw().get("concurrent_load", {}).get("loadavg"),
        "why_bit_identical_here_and_not_on_q2": (
            "production computes sum += float(int(nibble) - 8) * scale * x and "
            "Metal contracts that to an FMA, so both paths run the same "
            "instruction on exactly representable values. The q2 refold changes "
            "which constants the FMAs carry, so it lands at f32 epsilon instead."
        ),
    }


def token_projection() -> dict[str, Any]:
    b = json.loads((REPO / BUDGET_REL).read_text())
    cur = float(b["decode_wall_ms_per_token"])
    rows = {r["organ"]: float(r["gpu_ms"]) for r in b["organs"]["rows"]}
    missing = [o for o in Q4_ORGANS if o not in rows]
    if missing:
        raise Q4Refused(f"budget has no rows for {missing}")
    organ_ms = sum(rows[o] for o in Q4_ORGANS) + DN_INPROJ_MS
    speedup = measured()["speedup"]
    saved = organ_ms - organ_ms / speedup
    return {
        "organs_this_kernel_runs": list(Q4_ORGANS) + ["deltanet_in_projection"],
        "organ_ms_today": round(organ_ms, 4),
        "deltanet_in_projection_ms": DN_INPROJ_MS,
        "deltanet_counted_by_component_because": (
            "the DeltaNet ORGAN is 5.5971 ms and only its in-projection is a q4 "
            "matvec; the state update, rearrange and gated norm are different "
            "kernels this candidate does not touch"
        ),
        "deltanet_source": DN_INPROJ_SOURCE,
        "lm_head_excluded_because": (
            "it runs qwen_uniform_q4_group64_final_norm_lm_head_simdgroup8, a "
            "different kernel this candidate does not touch"
        ),
        "measured_kernel_speedup": speedup,
        "ms_saved_if_it_lands": round(saved, 4),
        "evidence_class": "PROSPECTIVE",
        "below_the_materiality_threshold": round(saved, 4) < 1.0,
        "why_keep_it_anyway": (
            "0.97 ms is under the 1 ms bar S025 set for what deserves serious "
            "effort, and this candidate is already BUILT, token-identical and "
            "free to keep - the bar governs what to START, not what to discard "
            "after it is finished. It would not have justified being built on "
            "its own; it was built because the q2 ladder had already paid for "
            "the construction."
        ),
        "the_isolated_number_is_not_a_bound_in_either_direction": (
            "the q2 bitcast predicted 2.7985 ms and the graph delivered 3.8541; "
            "widen_f4 predicted 0.7046 and delivered 1.0245; this candidate "
            "predicted 0.9695 and delivered 0.6836. Two over, one under. An "
            "earlier version of this receipt called the isolated number a LOWER "
            "bound off the first two points; the third falsified it. Only the "
            "complete-token A/B counts."
        ),
        "what_would_make_it_measured": (
            "run the 580-graph twice with HAWKING_Q4_UNPACK=bitcast set and "
            "unset and compare complete-token wall time and token ids"
        ),
    }


def resident_measured() -> dict[str, Any]:
    """The complete-token A/B. Both arms carry the q2 bitcast; only q4 differs."""
    on = json.loads((REPO / RES_ON_REL).read_text())["decode"][LIVE_ARM]
    off = json.loads((REPO / RES_OFF_REL).read_text())["decode"][LIVE_ARM]
    if on["new_token_ids"] != off["new_token_ids"]:
        raise Q4Refused("the arms produced different tokens; this is a regression")
    if on["theoretical_dispatches"] != off["theoretical_dispatches"]:
        raise Q4Refused("dispatch count differs; not the same graph")
    if any(on["fallbacks_reps"]) or any(off["fallbacks_reps"]):
        raise Q4Refused("a fallback fired; the graph is not the one claimed")
    left = sorted(
        k for k in set(on["dispatched_kernels_rep0"])
        if ("uniform_q4" in k or "affine_q2" in k) and "bitcast" not in k
    )
    if left != ["qwen_uniform_q4_embedding_lookup"]:
        raise Q4Refused(
            f"production matvecs still dispatched with the lever on: {left}. "
            "Only the embedding lookup should remain - it is a gather, not a "
            "matvec, and has no bitcast sibling."
        )

    def wall(arm: dict[str, Any]) -> float:
        r = sorted(arm["decode_wall_ns_reps"])
        return r[len(r) // 2] / 1e6 / len(arm["new_token_ids"])

    g_off, g_on = off["gpu_ns_median"] / 1e6, on["gpu_ns_median"] / 1e6
    w_off, w_on = wall(off), wall(on)
    return {
        "arm": LIVE_ARM,
        "both_arms_have_q2_bitcast_on": True,
        "gpu_ms_without_q4_bitcast": round(g_off, 4),
        "gpu_ms_with_q4_bitcast": round(g_on, 4),
        "gpu_ms_saved": round(g_off - g_on, 4),
        "speedup": round(g_off / g_on, 4),
        "wall_ms_without": round(w_off, 4),
        "wall_ms_with": round(w_on, 4),
        "wall_tps_without": round(1000.0 / w_off, 3),
        "wall_tps_with": round(1000.0 / w_on, 3),
        "token_identical": True,
        "dispatches": on["theoretical_dispatches"],
        "reps": len(on["decode_wall_ns_reps"]),
        "n_tokens": len(on["new_token_ids"]),
        "only_production_kernel_left": left,
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "window": (
            "not protected; ModelLake downloads were live. The paired ratio is "
            "the claim, and the absolute is not promotable."
        ),
    }


def projection_vs_graph() -> dict[str, Any]:
    """The isolated projection OVER-predicted here, which breaks a pattern."""
    predicted = token_projection()["ms_saved_if_it_lands"]
    actual = resident_measured()["gpu_ms_saved"]
    return {
        "predicted_ms_saved": predicted,
        "measured_ms_saved_in_the_graph": actual,
        "graph_over_prediction": round(actual / predicted, 4),
        "the_lower_bound_pattern_did_not_hold": actual < predicted,
        "reading": (
            "the q2 bitcast and widen_f4 both delivered MORE in the graph than "
            "their isolated numbers predicted, and this receipt previously said "
            "an isolated organ measurement looks like a LOWER bound for this "
            f"class of change. Here it OVER-predicted: {predicted} ms projected, "
            f"{actual} measured, {round(actual/predicted, 3)}x. Three "
            "observations, two directions. There is no bound, and the earlier "
            "wording was a pattern read off two points."
        ),
        "why_it_may_differ": (
            "not established. One candidate is that these kernels are a smaller "
            "share of their organs than the census attribution assumes - the "
            "DeltaNet in-projection correction already moved this projection "
            "once - so the denominator, not the kernel, may be what is wrong. "
            "That is a hypothesis and it has not been tested."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G094",
        "candidate": "q4 bitcast dequant",
        "lever": "HAWKING_Q4_UNPACK=bitcast",
        "default_is_unchanged": True,
        "kernels": [
            "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128_bitcast",
            "qwen_uniform_q4_group64_matvec_qkv_geo_tpr64_tg128_bitcast",
            "qwen_uniform_q4_group64_matvec_pair_concat_geo_tpr64_tg128_bitcast",
        ],
        "transform": (
            "nibble unpacked into an f32 mantissa (f = 2 + q/8) so neither the "
            "int-to-float convert nor the -8 zero point runs; the affine "
            "refolds per group as (8*scale)*f + (-24*scale)"
        ),
        "measured": measured(),
        "token_projection": token_projection(),
        "resident_measured": resident_measured(),
        "projection_vs_graph": projection_vs_graph(),
        "the_scar_was_obeyed": (
            "REMOVING_ONE_OP_CLASS_IS_NOT_A_LEVER_WHEN_FOUR_SHARE_THE_COST says "
            "the split is kernel-specific. This did not assume q4 behaves like "
            "q2 - it measured q4's own production, candidate and stripped arms."
        ),
        "the_bug_that_the_output_compare_caught": (
            "the first build extracted the nibble with (packed << 19) >> 8i, "
            "which is correct for q2's 16-bit packed word and DISCARDS BITS 13 "
            "AND UP of q4's 32-bit word of eight nibbles. It measured 1.40x at "
            "rel_fro 0.877. Second time a bitcast candidate has been fast and "
            "wrong; second time the output comparison caught it before it left "
            "the harness."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(lock_held=True, lane="g094-q4b"),
        ))
        return 0
    print(json.dumps({k: doc[k] for k in ("measured", "token_projection")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
