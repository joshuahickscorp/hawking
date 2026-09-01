#!/usr/bin/env python3
"""G092 REFUTED: the hoist is 4% SLOWER, and both earlier readings were the harness.

This obligation was recorded wrong twice. Both errors are kept here because the
sequence is the finding.

  READING 1 - "1.6338x, a win."       Measured at the harness default warmup 5.
  READING 2 - "0.9995x, the INPUT."   One run where production happened to read
                                      317.4 instead of its usual 195.6, which I
                                      attributed to real activations vs the
                                      synthetic fill.
  READING 3 - "0.960x, SLOWER."       Warmup 60, three runs, both inputs.

READING 2's MECHANISM WAS WRONG AND ITS CONCLUSION WAS RIGHT BY ACCIDENT. The
input does not matter: at warmup 60 production reads 331.1 GB/s on the synthetic
fill and 330.5 / 331.9 on real activations. What moved was WARMUP.

Production is the FIRST arm measured, and at warmup 5 it was still faulting its
buffers in during its own measured reps. Its per-rep timings go BIMODAL - 252k ns
or 428k ns in the same run, and in one run the first two reps ran fast and the
remaining nine ran slow. The median then lands on whichever mode won that run.
At warmup 60 every rep is 251k-257k and the spread is 1.01.

    warmup 60, three runs, two inputs:
        production   330.5   331.9   331.1(synthetic)
        hoist        320.1   319.2   318.0
        hoist/prod   0.969   0.962   0.960

THE HOIST IS SLOWER THAN THE KERNEL IT REPLACES. Its per-chunk fixup costs more
than the one FMA per weight it removes - and G094 then measured why: that FMA is
1.8% of this kernel's arithmetic and 0.6% of its time.

The FP answer stands and was never in doubt: not bit-identical, error at f32
epsilon, rel_fro under 1e-07. The candidate is rejected on speed.

THE PRODUCER IS FIXED, NOT JUST THE RECEIPT. The harness default warmup is now
60, every arm records rep_spread and a steady_state flag, and this module refuses
to read an arm that was not in steady state.

    python3 tools/future/dequant_hoist_ab.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402

RECORDED_BY = "tools/future/dequant_hoist_ab.py"
RECEIPT_NAME = "DEQUANT_HOIST_AB.json"

REAL_REL = "receipts/future/_G092_HOIST_REALX_raw.json"
SYNTH_REL = "receipts/future/_G092_HOIST_SYNTH_W60_raw.json"

ARMS = ("production", "hoist", "arm_a_stripped")
STEADY_MAX_SPREAD = 1.10
MIN_WARMUP = 60


class AbRefused(RuntimeError):
    """An arm is missing, or was measured outside steady state."""


def _raw(rel: str) -> dict[str, Any]:
    p = REPO / rel
    if not p.is_file():
        raise AbRefused(f"{rel} is not on disk; run the probe first")
    return json.loads(p.read_text())


def _run(rel: str, label: str) -> dict[str, Any]:
    d = _raw(rel)
    warmup = int(d.get("warmup", 0))
    if warmup < MIN_WARMUP:
        raise AbRefused(
            f"{rel} was taken at warmup {warmup} < {MIN_WARMUP}; the "
            "first-measured arm is not in steady state at that warmup and its "
            "median is a coin flip between two modes"
        )
    m = d["mlp"]
    for a in ARMS:
        if a not in m:
            raise AbRefused(f"{rel}: arm {a} is absent; not a matched comparison")
        spread = float(m[a].get("rep_spread", 99.0))
        if spread > STEADY_MAX_SPREAD:
            raise AbRefused(
                f"{rel}: arm {a} has rep spread {spread:.4f} > {STEADY_MAX_SPREAD}"
            )
    gb = {a: float(m[a]["effective_gb_s"]) for a in ARMS}
    return {
        "input": label,
        "raw": rel,
        "warmup": warmup,
        "production_gb_s": gb["production"],
        "hoist_gb_s": gb["hoist"],
        "arm_a_stripped_gb_s": gb["arm_a_stripped"],
        "hoist_over_production": round(gb["hoist"] / gb["production"], 4),
        "arm_a_over_production": round(gb["arm_a_stripped"] / gb["production"], 4),
        "production_rep_spread": float(m["production"]["rep_spread"]),
        "loadavg": d.get("concurrent_load", {}).get("loadavg"),
    }


def measured() -> dict[str, Any]:
    real = _run(REAL_REL, "real_captured_activation")
    synth = _run(SYNTH_REL, "synthetic_dyadic_fill")
    prod_delta = abs(real["production_gb_s"] - synth["production_gb_s"])
    prod_rel = prod_delta / synth["production_gb_s"]
    return {
        "verdict": "SLOWER_THAN_PRODUCTION",
        "headline_hoist_over_production": real["hoist_over_production"],
        "real_activation_run": real,
        "synthetic_control_run": synth,
        "the_input_does_not_matter": {
            "production_real_gb_s": real["production_gb_s"],
            "production_synthetic_gb_s": synth["production_gb_s"],
            "relative_difference": round(prod_rel, 4),
            "reading": (
                "at adequate warmup the production kernel reads the same rate on "
                "both inputs to within "
                f"{prod_rel:.2%}. The earlier claim that the synthetic fill "
                "depressed production is REFUTED by this control."
            ),
        },
        "what_actually_moved": {
            "cause": "INSUFFICIENT_WARMUP_ON_THE_FIRST_MEASURED_ARM",
            "signature": (
                "at warmup 5 production's per-rep timings are bimodal, 252k ns or "
                "428k ns within one run, and one run showed 2 fast reps followed "
                "by 9 slow ones. At warmup 60 the spread is 1.01."
            ),
            "why_production_specifically": (
                "production is the first arm timed, so it pays first-touch "
                "residency for buffers every later arm finds already warm. The "
                "later arms were never bimodal in any run."
            ),
        },
    }


def bit_identity() -> dict[str, Any]:
    rows = _raw(REAL_REL)["mlp"]["hoist"]["output_compare"]
    total = sum(int(c["n_compared"]) for c in rows)
    exact = sum(int(c["n_bit_exact"]) for c in rows)
    return {
        "measured_on": "real_captured_activation",
        "n_compared": total,
        "n_bit_exact": exact,
        "all_exact": exact == total,
        "max_rel_fro": max(float(c["rel_fro"]) for c in rows),
        "max_abs_err": max(float(c["max_abs_err"]) for c in rows),
        "reading": (
            "NOT bit-identical, which is correct for a transform that "
            "reassociates, with the error at f32 epsilon. This was never the "
            "reason to reject the candidate."
        ),
        "the_dyadic_bit_exact_result_was_an_artifact": (
            "the harness fill is (i % 17) * 0.125 - 1.0 - dyadic values with four "
            "mantissa bits - and the codes are 2-bit, so both orderings landed on "
            "exactly representable intermediates. Flagged as an artifact when it "
            "was first recorded, and confirmed by the real-activation run."
        ),
    }


def scar() -> dict[str, Any]:
    return {
        "id": "WARMUP_5_LEAVES_THE_FIRST_MEASURED_ARM_OUTSIDE_STEADY_STATE",
        "statement": (
            "the harness timed arms in blocks at warmup 5. The first block - "
            "always production - was still faulting buffers in during its "
            "measured reps and went bimodal between 252k and 428k ns, a 1.70 "
            "spread. Every ratio taken against that arm inherits which mode won."
        ),
        "who_it_hit": (
            "receipts/future/MLP_ALU_ROOFLINE.json carries the signature: its MLP "
            "production arm has min 252500 and max 428333, spread 1.696, while "
            "its DeltaNet production arm is 1.010. Its median landed on the fast "
            "mode so its 329.6 GB/s is close to the warmup-60 value of 331 - it "
            "was right by luck, not by construction."
        ),
        "producer_fix": (
            "the harness default warmup is 60, arm_json emits gpu_ns_min, "
            "gpu_ns_max, rep_spread and a steady_state flag against a 1.10 bound, "
            "and the consumers refuse an arm outside it. RECEIPT-ONLY FIXES ARE "
            "FORBIDDEN: the producer changed, then the receipt was regenerated."
        ),
        "reopen": (
            "any receipt whose first-measured arm has rep_spread above 1.10 is "
            "not a measurement regardless of how clean its median looks"
        ),
    }


def build() -> dict[str, Any]:
    m = measured()
    return {
        "obligation": "G092",
        "candidate": "affine_q2_geo_tpr64_tg128_hoist",
        "kernel": "alu_roofline_affine_q2_geo_tpr64_tg128_hoist",
        "transform": (
            "apply the per-group affine once per 8-weight chunk instead of once "
            "per weight, using a per-chunk sum of x that amortises over all rows"
        ),
        "verdict": m["verdict"],
        "measured": m,
        "bit_identity": bit_identity(),
        "scar": scar(),
        "correction_history": [
            {"reading": 1, "claim": "1.6338x, a win",
             "cause": "warmup 5; production's median landed on the slow mode"},
            {"reading": 2, "claim": "0.9995x, the synthetic input slowed production",
             "cause": "warmup 5; production's median landed on the FAST mode in "
                      "that one run, and I attributed the difference to the input"},
            {"reading": 3, "claim": "0.960x, the hoist is slower",
             "cause": "warmup 60; three runs, two inputs, all ratios within 0.01"},
        ],
        "why_it_is_slower": (
            "G094 measured the class the hoist attacked: the affine FMA is 1.8% "
            "of this kernel's arithmetic and 0.6% of its time. The hoist's "
            "per-chunk fixup costs more than that."
        ),
        "ms_per_token_saved": 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(lock_held=True, lane="g092-hoist"),
        ))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("verdict", "measured", "correction_history")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
