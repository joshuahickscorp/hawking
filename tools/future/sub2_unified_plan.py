#!/usr/bin/env python3
"""G107: three planners, one plan, and the places they contradicted each other.

S029 asked for three independent plans for reaching 2.0 complete BPW - Claude's,
Grok's, and for the first time in this campaign the RESIDENT'S OWN - then a
unification citing which part came from whom and where they disagreed.

They disagreed three times and the disagreements are the useful part.

    CLAUDE said MLP 2.5 -> 2.25 bpw is worth 0.293 ms/token.
    GROK said 0.000. GROK WAS RIGHT: 2.5 bpw is 2.000 code bits plus 32/64 aux
    bits for the f16 scale and bias, so going to 2.25 halves the AUX and never
    touches a code. Aux bills 0.000 ms/GB, measured.

    THE RESIDENT said the SwiGLU gate is control deserving literal storage while
    up and down are bulk that can be generated or shared.
    THE MEASUREMENT said gate never exceeds 1.31x up per matched element, and
    DOWN - which it placed in the discard bucket - is most sensitive at 9 of 12
    points.

    CLAUDE said 2.0 BPW plus the landed levers reaches 62.75 TPS if decode stays
    byte-bound. GROK said no conventional encoding reaches 2.0 at all: the floor
    is 2.5081 even granting every untested move. BOTH HOLD - the first is a
    conditional about what 2.0 would buy, the second is about whether the
    current representation language can express it.

    python3 tools/future/sub2_unified_plan.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/sub2_unified_plan.py"
RECEIPT_NAME = "SUB2_UNIFIED_PLAN.json"

PLANNERS = ("claude", "grok", "resident")

SOURCES = {
    "claude": "receipts/future/GAP_LEDGER_60.json",
    "grok": "receipts/future/REPRESENTATION_FLOOR.json",
    "resident": "receipts/future/FUNCTIONAL_ROLE_PROBE.json",
}


class PlanRefused(RuntimeError):
    """A planner's artifact is missing; a unification of two is not three."""


def _receipt(rel: str) -> dict[str, Any]:
    p = REPO / rel
    if not p.is_file():
        raise PlanRefused(
            f"{rel} is not on disk. S029 requires THREE plans; unifying fewer "
            "and calling it three would be the fake completion this campaign "
            "forbids."
        )
    return json.loads(p.read_text())


def planners_present() -> dict[str, Any]:
    out = {}
    for who, rel in SOURCES.items():
        out[who] = {"artifact": rel, "present": (REPO / rel).is_file()}
    missing = [w for w, v in out.items() if not v["present"]]
    if missing:
        # Same reason as _receipt. Two refusal paths with different wording is
        # how a caller ends up handling one and not the other.
        raise PlanRefused(
            f"missing plan artifacts for {missing}. S029 requires THREE plans; "
            "unifying fewer and calling it three would be the fake completion "
            "this campaign forbids."
        )
    return out


def disagreements() -> list[dict[str, Any]]:
    """Where the planners contradicted each other, and who the evidence backed."""
    floor = _receipt(SOURCES["grok"])
    probe = _receipt(SOURCES["resident"])
    return [
        {
            "id": "D1.mlp_2p25_is_aux_not_codes",
            "claude_said": "MLP 2.5 -> 2.25 bpw is worth 0.293 ms/token",
            "grok_said": "0.000 ms/token",
            "resolved_by": "arithmetic on the stored layout",
            "winner": "grok",
            "why": (
                "2.5 bpw stored is 2.000 code bits plus 32/64 = 0.500 aux bits "
                "for the f16 scale and bias per 64 weights. Going to 2.25 "
                "halves the AUX and never touches a code, and aux bills "
                "0.000 ms/GB by measurement."
            ),
            "consequence": (
                "Claude's non-refuted total fell from 1.114 to 0.821 ms/token - "
                "further below the 1 ms materiality bar, not closer to it"
            ),
        },
        {
            "id": "D2.functional_role_allocation",
            "resident_said": (
                "the SwiGLU gate is control and deserves literal storage; up "
                "and down are linear bulk that can be generated or shared"
            ),
            "measurement_said": (
                f"gate never exceeds {probe['verdict']['gate_over_up_range'][1]}x "
                f"up per matched element, and down is most sensitive at "
                f"{probe['ranking']['n_where_down_is_most_sensitive']} of "
                f"{probe['ranking']['n_points']} points"
            ),
            "resolved_by": "matched-element perturbation across three layers",
            "winner": "measurement",
            "why": (
                "the resident put down in the discard bucket and down is the "
                "most sensitive tensor tested. The claim was structural and a "
                "structural measurement answered it."
            ),
            "consequence": (
                "functional role survives as an allocation AXIS; this "
                "particular assignment does not. Scars prune methods, not goals."
            ),
        },
        {
            "id": "D3.what_2p0_would_buy_versus_whether_2p0_is_expressible",
            "claude_said": (
                "2.0 BPW plus the landed bitcast levers reaches 62.75 TPS if "
                "decode stays byte-bound"
            ),
            "grok_said": (
                "no conventional encoding reaches 2.0 at all; the floor is "
                f"{floor['floor']['if_every_untested_move_worked_bpw']} BPW even "
                "granting every untested move"
            ),
            "resolved_by": "they answer different questions",
            "winner": "both",
            "why": (
                "Claude's is a conditional about the PAYOFF of 2.0. Grok's is "
                "about whether the current representation LANGUAGE can express "
                "it. Both are true, and together they are the case for leaving "
                "the language rather than lowering the target."
            ),
            "consequence": (
                "the unified plan's top route is not a better code. It is a "
                "capability-allocated representation."
            ),
        },
    ]


def routes() -> list[dict[str, Any]]:
    """Every surviving route, with evidence status and a cheapest falsifier."""
    return [
        {
            "id": "R1.capability_allocated_heterogeneous",
            "from": "claude + resident",
            "route": (
                "allocate bits by measured capability value rather than "
                "uniformly: some regions literal, some 2-bit, some 1-bit, some "
                "generated, some omitted"
            ),
            "evidence_status": "UNTESTED",
            "why_live": (
                "every refutation blocking it was measured at a FIDELITY level "
                "(receipts/future/FIDELITY_HIERARCHY.json), and the body "
                "tolerates 40% row destruction at 0.0059 cosine"
            ),
            "cheapest_falsifier": (
                "one capability probe at a destruction level local fidelity "
                "would reject. If capability breaks there too, heterogeneous "
                "allocation buys nothing and the fidelity bars were right."
            ),
            "max_payoff": "unbounded within the 1.139 BPW between 3.139 and 2.0",
        },
        {
            "id": "R2.conventional_coding",
            "from": "grok",
            "route": "entropy coding, larger groups, lower bitwidths",
            "evidence_status": "MEASURED_INSUFFICIENT",
            "why_live": "it is not - it bottoms out at 2.5081 BPW",
            "cheapest_falsifier": "already run; the floor is measured",
            "max_payoff": "0.821 ms/token, below the 1 ms materiality bar",
        },
        {
            "id": "R3.functional_role_allocation",
            "from": "resident",
            "route": "allocate by functional role rather than organ boundary",
            "evidence_status": "REFUTED_AS_STATED",
            "why_live": (
                "the AXIS survives; the gate-high assignment does not. A "
                "different role labelling has not been tested."
            ),
            "cheapest_falsifier": (
                "matched-element damage across a new role labelling, the same "
                "control that killed the first one"
            ),
            "max_payoff": "unknown until a labelling is proposed",
        },
        {
            "id": "R4.dense_becomes_conditionally_sparse",
            "from": "resident",
            "route": (
                "the resident observed that in an MoE the information that must "
                "be present is the IDENTITY of the active experts, not the "
                "continuous weights. A dense source need not have a dense "
                "executable."
            ),
            "evidence_status": "UNTESTED",
            "why_live": (
                "the MoE precedent is real - Qwen80 reached 1.4444 complete BPW "
                "with experts at 1.2349 - and nothing has tested whether a "
                "dense body admits an analogous structure"
            ),
            "cheapest_falsifier": (
                "measure whether any activation-conditional structure exists in "
                "the dense MLP at all. If every weight participates in every "
                "token equally, this route is dead."
            ),
            "max_payoff": "the MoE precedent reached 2.17x smaller than this body",
        },
    ]


def build() -> dict[str, Any]:
    d = disagreements()
    r = routes()
    return {
        "obligation": "G107",
        "authority": "S029",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "planners": planners_present(),
        "n_disagreements": len(d),
        "disagreements": d,
        "routes": r,
        "n_routes_live": sum(1 for x in r if x["evidence_status"] == "UNTESTED"),
        "top_route": "R1.capability_allocated_heterogeneous",
        "why_that_route": (
            "the three plans agree on the negative - conventional coding cannot "
            "express 2.0 - and the only refutations standing against a "
            "capability-allocated representation were measured at fidelity "
            "levels that cannot speak to a capability claim."
        ),
        "who_was_wrong_about_what": (
            "Claude was wrong about the ms value of an aux cut. The resident was "
            "wrong about which tensor carries information. Grok was wrong about "
            "nothing measured here, and its floor is the load-bearing number."
        ),
        "the_resident_is_not_scored_on_hit_rate": (
            "it produced three falsifiable structural hypotheses and all three "
            "died cheaply - mutual information in two minutes, matched "
            "perturbation in forty seconds a layer, the bias claim on "
            "inspection. Each removed a live possibility. That is the loop this "
            "campaign is building, and the hit rate is not the metric."
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
    print(json.dumps({k: doc[k] for k in
                      ("n_disagreements", "top_route", "who_was_wrong_about_what",
                       "routes")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
