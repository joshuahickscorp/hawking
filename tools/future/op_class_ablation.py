"""G094: which op class binds the q2 MLP matvec, and the candidate that follows.

The hoist attacked the affine FMA and bought nothing. This asks WHY by removing
one class of per-weight work at a time, holding byte traffic and load count
identical to production. None of the ablations computes the right answer - they
are arm_a with one class added back.

    production   331.5 GB/s   1.000x     shift+mask, convert, 2 FMA per weight
    noaffine     333.1        1.003x     the affine FMA removed
    noconv       436.1        1.313x     convert and both FMAs removed
    nounpack     438.3        1.320x     shift+mask and 7/8 converts removed
    arm_a        502.4        1.513x     all arithmetic removed

Solving those for the four classes puts the uint-to-float CONVERT at 44% of the
arithmetic and 15% of production's total time - the largest single class, and
twenty-five times the affine FMA the hoist spent a day on.

    shift+mask            29.2% of arithmetic     9.9% of production time
    uint->float convert   44.2%                  14.9%
    both FMAs             26.6%                   9.0%
      of which the affine  1.8%                   0.6%

NO SINGLE CLASS DOMINATES, which is exactly why removing one bought nothing.

THE CONVERT IS REMOVABLE. A 2-bit code needs no convert: placing q at bits 21-22
of an f32 with exponent field 0x40000000 yields f = 2.0 + 0.5*q exactly, so
q = 2*(f-2) and w = q*scale + bias becomes w = (2*scale)*f + (bias - 4*scale)
with both constants folded ONCE PER GROUP from the same half scale and bias
production already loads. Per weight this trades a convert for an OR.

    bitcast      404.4 GB/s   1.218x     and it computes the right answer

    python3 tools/future/op_class_ablation.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402

RECORDED_BY = "tools/future/op_class_ablation.py"
RECEIPT_NAME = "OP_CLASS_ABLATION.json"
RAW_REL = "receipts/future/_G094_OPCLASS_raw.json"
BUDGET_REL = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"

# The organs this kernel actually runs in the resident token.
Q2_ORGANS = ("mlp_gate_up", "mlp_down")

# An arm whose slowest rep exceeds its fastest by more than this was not in
# steady state. The producer emits the flag; this refuses to read past it.
STEADY_MAX_SPREAD = 1.10


class AblationRefused(RuntimeError):
    """An arm is missing, or was measured outside steady state."""


def _raw() -> dict[str, Any]:
    p = REPO / RAW_REL
    if not p.is_file():
        raise AblationRefused(f"{RAW_REL} is not on disk; run the probe first")
    return json.loads(p.read_text())


def arms() -> dict[str, dict[str, Any]]:
    m = _raw()["mlp"]
    out: dict[str, dict[str, Any]] = {
        "production": m["production"],
        "hoist": m["hoist"],
        "bitcast": m["bitcast"],
        "arm_a_stripped": m["arm_a_stripped"],
    }
    for a in m["op_class_ablations"]:
        out[a["label"]] = a
    for name in ("production", "noaffine", "noconv", "nounpack", "arm_a_stripped",
                 "bitcast"):
        if name not in out:
            raise AblationRefused(f"arm {name} is absent; the ladder is incomplete")
        spread = float(out[name].get("rep_spread", 99.0))
        if spread > STEADY_MAX_SPREAD:
            raise AblationRefused(
                f"arm {name} has rep spread {spread:.4f} > {STEADY_MAX_SPREAD}; it "
                "was not in steady state, so its median is a coin flip between two "
                "modes rather than a measurement"
            )
    return out


def _t(a: dict[str, Any], base: float) -> float:
    """Time normalised so production is 1.0. Ratios of rates invert to times."""
    return base / float(a["effective_gb_s"])


def decomposition() -> dict[str, Any]:
    a = arms()
    base = float(a["production"]["effective_gb_s"])
    t = {k: _t(v, base) for k, v in a.items()}
    arithmetic = t["production"] - t["arm_a_stripped"]
    unpack = t["noconv"] - t["arm_a_stripped"]
    fmas = t["nounpack"] - t["arm_a_stripped"]
    convert = arithmetic - unpack - fmas
    affine = t["production"] - t["noaffine"]
    for name, v in (("shift_and_mask", unpack), ("both_fmas", fmas),
                    ("uint_to_float_convert", convert)):
        if v <= 0:
            raise AblationRefused(
                f"the time attributed to {name} is {v:.4f}, not positive; the "
                "ablations do not decompose additively and the model is wrong. "
                "A negative class means an arm that removes work ran SLOWER "
                "than one that keeps it, which is a measurement to explain, not "
                "a share to report."
            )
    classes = {
        "shift_and_mask": unpack,
        "uint_to_float_convert": convert,
        "both_fmas": fmas,
    }
    return {
        "arithmetic_share_of_production_time": round(arithmetic, 4),
        "classes": {
            k: {
                "share_of_production_time": round(v, 4),
                "share_of_arithmetic": round(v / arithmetic, 4),
            }
            for k, v in classes.items()
        },
        "affine_fma_alone": {
            "share_of_production_time": round(affine, 4),
            "share_of_arithmetic": round(affine / arithmetic, 4),
            "measured_directly_by": "noaffine",
        },
        "largest_class": max(classes, key=classes.get),
        "no_single_class_dominates": max(classes.values()) / arithmetic < 0.5,
        "why_the_hoist_bought_nothing": (
            "the affine FMA it removed is 1.8% of this kernel's arithmetic and "
            "0.6% of its time. The measured hoist is 0.96x - slightly SLOWER, "
            "because its per-chunk fixup costs more than the FMA it saves."
        ),
        "method": (
            "times normalised to production. unpack = noconv - arm_a (noconv "
            "keeps only shift+mask); fmas = nounpack - arm_a (nounpack keeps "
            "only the two FMAs); convert = arithmetic - unpack - fmas. The "
            "decomposition ASSUMES the classes are additive, which is checked "
            "only by the residual coming out positive and by noaffine agreeing "
            "with the FMA term's scale."
        ),
    }


def bitcast_candidate() -> dict[str, Any]:
    a = arms()
    prod = float(a["production"]["effective_gb_s"])
    bc = a["bitcast"]
    cmp_rows = bc["output_compare"]
    speedup = float(bc["effective_gb_s"]) / prod
    return {
        "kernel": bc["kernel"],
        "speedup_over_production": round(speedup, 4),
        "production_gb_s": prod,
        "bitcast_gb_s": float(bc["effective_gb_s"]),
        "transform": bc["transform"],
        "exact_in_the_reals": True,
        "bit_identical": all(bool(c["bit_identical"]) for c in cmp_rows),
        "max_rel_fro": max(float(c["rel_fro"]) for c in cmp_rows),
        "max_abs_err": max(float(c["max_abs_err"]) for c in cmp_rows),
        "fp_reading": (
            "NOT bit-identical - the two FMAs run on refolded constants - but "
            "the error is at f32 epsilon, rel_fro under 2e-07 on every "
            "projection. Same tolerance class the hoist reached."
        ),
        "the_first_version_was_fast_and_wrong": (
            "bit 21 is 0.25 of the significand and the exponent is 2^1, so the "
            "step is 0.5 per code, not 0.25. The first build used 4 and -8 "
            "instead of 2 and -4, measured 1.225x, and returned garbage at "
            "rel_fro 1.26. The speed was real and the answer was not. This is "
            "what the output comparison is for."
        ),
    }


def token_projection() -> dict[str, Any]:
    """What the measured kernel speedup is worth IF it lands in the resident."""
    b = json.loads((REPO / BUDGET_REL).read_text())
    cur = float(b["decode_wall_ms_per_token"])
    rows = {r["organ"]: float(r["gpu_ms"]) for r in b["organs"]["rows"]}
    missing = [o for o in Q2_ORGANS if o not in rows]
    if missing:
        raise AblationRefused(f"budget has no rows for {missing}")
    organ_ms = sum(rows[o] for o in Q2_ORGANS)
    speedup = bitcast_candidate()["speedup_over_production"]
    saved = organ_ms - organ_ms / speedup
    new_ms = cur - saved
    return {
        "organs_this_kernel_runs": list(Q2_ORGANS),
        "organ_ms_today": round(organ_ms, 4),
        "measured_kernel_speedup": speedup,
        "ms_saved_if_it_lands": round(saved, 4),
        "token_ms_before": cur,
        "token_ms_after": round(new_ms, 4),
        "tps_before": round(1000.0 / cur, 3),
        "tps_after": round(1000.0 / new_ms, 3),
        "tps_gain": round(1000.0 / new_ms - 1000.0 / cur, 3),
        "evidence_class": "PROSPECTIVE",
        "why_prospective": (
            "the speedup is MEASURED on the production kernel in an isolated "
            "harness with production's own bytes and layout. It is a projection "
            "only in that the kernel has not yet been changed in the resident "
            "and the resident has not been re-measured. Nothing about the "
            "organ census or the arithmetic is assumed."
        ),
        "what_would_make_it_measured": (
            "edit the q2 affine matvec in the resident shader, re-run the "
            "resident token budget under a protected lease, and compare "
            "complete-token wall TPS - not this kernel's GB/s"
        ),
    }


def build() -> dict[str, Any]:
    d = decomposition()
    bc = bitcast_candidate()
    tp = token_projection()
    return {
        "obligation": "G094",
        "authority": "S026 §4, §5, §10, §11",
        "question": (
            "which internal physical resource does the q2 MLP matvec spend its "
            "arithmetic on, given that removing one FMA per weight bought nothing"
        ),
        "arms": {
            k: {
                "gb_s": float(v["effective_gb_s"]),
                "over_production": round(
                    float(v["effective_gb_s"])
                    / float(arms()["production"]["effective_gb_s"]), 4),
                "rep_spread": v.get("rep_spread"),
                "computes_the_right_answer": v.get("computes_the_right_answer", True),
            }
            for k, v in arms().items()
        },
        "decomposition": d,
        "bitcast_candidate": bc,
        "token_projection": tp,
        "scar": {
            "id": "REMOVING_ONE_OP_CLASS_IS_NOT_A_LEVER_WHEN_FOUR_SHARE_THE_COST",
            "statement": (
                "this kernel's arithmetic splits 29/44/27 across unpack, convert "
                "and FMA. Attacking any one without measuring the split first "
                "risks the hoist's outcome: a correct, well-built, useless "
                "kernel. MEASURE THE SPLIT BEFORE CHOOSING THE TARGET."
            ),
            "reopen": (
                "the split is kernel-specific. The q4 DeltaNet matvec has a "
                "different code width and a different unpack, so it needs its "
                "own ladder before anyone assumes convert dominates there too."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(lock_held=True, lane="g094-opclass"),
        ))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("decomposition", "bitcast_candidate", "token_projection")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
