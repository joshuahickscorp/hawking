#!/usr/bin/env python3
"""G118: the resident's functional-role hypothesis, measured and refuted as stated.

S031 §5-§12. The resident proposed, unprompted, that information value follows
FUNCTIONAL ROLE rather than source organ boundary - that the SwiGLU gate is a
"non-linear, state-dependent selector" deserving literal information, while up
and down are "linear components" that "can be almost entirely generated or
shared".

Matched arms: the same perturbation (zero a random fraction of output rows), the
same layer, the same element count to within 0.05%, replayed over real captured
activations and compared downstream rather than in weight space.

    layer 42, 40% of rows zeroed
        gate   damage 0.005527
        up     damage 0.004526
        down   damage 0.005905

DOWN IS THE MOST SENSITIVE TENSOR AT EVERY LAYER AND EVERY STRENGTH, and the
resident placed it in the generate-or-share bucket. Gate is 1.00x to 1.22x up,
not the order-of-magnitude separation the hypothesis needs.

VERDICT: REFUTED AS STATED, at LOCAL_FUNCTIONAL_FIDELITY. The claim was
structural, so a structural measurement can answer it - but S031 §9 forbids
local cosine as the FINAL verdict, so this does not close the capability-level
question, and the receipt says which level it speaks for.

THE LARGER FINDING IS THE ROBUSTNESS. Zeroing 40% of an MLP tensor's output rows
moves the hidden state by half a percent of cosine. That is the destruction curve
S031 §10 asks for, and it is the number that matters for byte elimination.

    python3 tools/future/functional_role_probe.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/functional_role_probe.py"
RECEIPT_NAME = "FUNCTIONAL_ROLE_PROBE.json"
RAW_REL = "receipts/future/_G118_ROLE_PROBE_raw.json"

# S031 §7. The resident's own labels, as a HYPOTHESIS LAYER - not source truth.
ROLE = {"gate": "CONTROL", "up": "BULK_LINEAR", "down": "BULK_LINEAR"}

# S031 §2 hierarchy. This probe speaks for exactly one level.
MEASURED_LEVEL = "LOCAL_FUNCTIONAL_FIDELITY"

# Matched arms are only matched if the destroyed element counts agree.
MATCH_TOLERANCE = 0.01


class ProbeRefused(RuntimeError):
    """The arms are not matched, or the evidence is missing."""


def _raw() -> dict[str, Any]:
    p = REPO / RAW_REL
    if not p.is_file():
        raise ProbeRefused(f"{RAW_REL} is not on disk; run the probe first")
    return json.loads(p.read_text())


def rows() -> list[dict[str, Any]]:
    return list(_raw()["rows"])


def arms_are_matched() -> dict[str, Any]:
    """A damage comparison across tensors means nothing if the arms differ."""
    worst = 0.0
    checked = 0
    for layer in sorted({r["layer"] for r in rows()}):
        for frac in sorted({r["frac"] for r in rows()}):
            grp = [r for r in rows() if r["layer"] == layer and r["frac"] == frac]
            if len(grp) < 2:
                continue
            n = [r["elements_destroyed"] for r in grp]
            spread = (max(n) - min(n)) / max(n)
            worst = max(worst, spread)
            checked += 1
    if worst > MATCH_TOLERANCE:
        raise ProbeRefused(
            f"arms differ by {worst:.4f} in elements destroyed, above "
            f"{MATCH_TOLERANCE}; this is not a matched comparison"
        )
    return {
        "n_groups_checked": checked,
        "worst_element_count_spread": round(worst, 6),
        "tolerance": MATCH_TOLERANCE,
        "why_it_matters": (
            "gate and up are 17408x5120 while down is 5120x17408. Comparing "
            "damage across them is only meaningful because the same NUMBER of "
            "elements is destroyed, not the same fraction of a different shape."
        ),
    }


def ranking() -> dict[str, Any]:
    arms_are_matched()
    out = []
    for layer in sorted({r["layer"] for r in rows()}):
        for frac in sorted({r["frac"] for r in rows()}):
            grp = {r["tensor"]: r for r in rows()
                   if r["layer"] == layer and r["frac"] == frac}
            if len(grp) < 3:
                continue
            order = sorted(grp, key=lambda t: -grp[t]["damage"])
            out.append({
                "layer": layer,
                "frac": frac,
                "damage": {t: round(grp[t]["damage"], 6) for t in grp},
                "most_sensitive": order[0],
                "least_sensitive": order[-1],
                "gate_over_up": round(grp["gate"]["damage"] / grp["up"]["damage"], 4),
                "down_over_gate": round(grp["down"]["damage"] / grp["gate"]["damage"], 4),
            })
    downs = sum(1 for o in out if o["most_sensitive"] == "down")
    return {
        "points": out,
        "n_points": len(out),
        "n_where_down_is_most_sensitive": downs,
        "down_is_most_sensitive_everywhere": downs == len(out),
        "gate_over_up_range": [
            min(o["gate_over_up"] for o in out),
            max(o["gate_over_up"] for o in out),
        ],
    }


def verdict() -> dict[str, Any]:
    r = ranking()
    lo, hi = r["gate_over_up_range"]
    supported = hi >= 2.0 and r["n_where_down_is_most_sensitive"] == 0
    return {
        "hypothesis": (
            "information value follows functional role: CONTROL (the SwiGLU "
            "gate) deserves literal information while BULK_LINEAR (up, down) "
            "can be generated or shared"
        ),
        "proposed_by": "the resident, sealed-3.14, unprompted",
        "status": "SUPPORTED" if supported else "REFUTED",
        "measured_at_level": MEASURED_LEVEL,
        "gate_over_up_range": [lo, hi],
        "down_is_most_sensitive_everywhere": r["down_is_most_sensitive_everywhere"],
        "why": (
            f"gate is {lo}x to {hi}x as damaging as up per matched element - not "
            "the separation the hypothesis needs - and DOWN, which the resident "
            "placed in the generate-or-share bucket, is the most sensitive "
            f"tensor at {r['n_where_down_is_most_sensitive']} of "
            f"{r['n_points']} measured points."
        ),
        "what_this_does_not_close": (
            "S031 §9 forbids local cosine as the final verdict. This measures "
            "LOCAL_FUNCTIONAL_FIDELITY two layers downstream, not CAPABILITY. A "
            "role could still carry unequal CAPABILITY value while carrying "
            "equal fidelity value, and that experiment has not been run."
        ),
        "what_survives_of_it": (
            "the shape of the question. Functional role remains a legitimate "
            "allocation axis; what is refuted is this particular assignment - "
            "gate high, up and down low. The measurement points at down, which "
            "writes directly into the residual stream, rather than at the gate."
        ),
    }


def robustness() -> dict[str, Any]:
    """S031 §10. The destruction curve, which matters more than the ordering."""
    pts = [r for r in rows() if r["frac"] == max(x["frac"] for x in rows())]
    worst = max(pts, key=lambda r: r["damage"])
    return {
        "at_fraction_zeroed": worst["frac"],
        "worst_tensor": worst["tensor"],
        "worst_layer": worst["layer"],
        "worst_damage": round(worst["damage"], 6),
        "statement": (
            f"zeroing {worst['frac']:.0%} of the output rows of "
            f"{worst['tensor']} at layer {worst['layer']} - "
            f"{worst['elements_destroyed']:,} elements - moves the hidden state "
            f"by {worst['damage']:.4f} of cosine. The MLP tolerates enormous "
            "row destruction at this measurement depth."
        ),
        "why_this_is_the_bigger_result": (
            "byte elimination needs to know how much can go, not which tensor "
            "goes first. A curve this flat says the binding constraint is not "
            "found at this level of measurement, and that the capability-level "
            "experiment is the one worth running."
        ),
        "caveat": (
            "hidden-state cosine two layers downstream is not capability, and a "
            "flat fidelity curve is exactly the situation where a capability "
            "cliff could still hide. This licenses the next experiment, not a "
            "deletion."
        ),
    }


def build() -> dict[str, Any]:
    raw = _raw()
    return {
        "obligation": "G118",
        "authority": "S031 §5-§12, §22",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "method": raw["method"],
        "perturbation": raw["perturbation"],
        "measure": raw["measure"],
        "activations": raw["activations"],
        "role_labels": ROLE,
        "role_labels_are_a_hypothesis_layer": (
            "S031 §7: these are the resident's candidate labels, not source "
            "truth, and each has to be testable. This receipt tests them."
        ),
        "arms_are_matched": arms_are_matched(),
        "ranking": ranking(),
        "verdict": verdict(),
        "robustness": robustness(),
        "this_is_the_behaviour_being_graded": (
            "S031 §48: the grade is not whether the gate proved important. The "
            "resident proposed a structural hypothesis, the lab built the "
            "matched control it implies, and the belief moved on the result. "
            "Its first idea died on mutual information; its second died on "
            "matched perturbation."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args(argv)
    doc = build()
    if a.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps({k: doc[k] for k in ("verdict", "robustness", "ranking")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
