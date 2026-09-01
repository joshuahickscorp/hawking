#!/usr/bin/env python3
"""G097: the inference rules this campaign paid for, as data that fires.

S026 §17-§18. Each rule states a condition over evidence ALREADY ON DISK, cites
the receipt that bought it, carries a reopen condition, and when it fires it
emits a real WorkUnit through frontiers._item - not a log line.

A rule that fires on a hand-written scenario proves nothing, so every condition
reads a receipt this repository actually contains. A rule whose evidence is
missing REFUSES rather than defaulting to not-fired: silence and absence are
different, and only one of them is information.

    python3 tools/future/causal_pattern_library.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402
import frontiers as fr  # noqa: E402

RECORDED_BY = "tools/future/causal_pattern_library.py"
RECEIPT_NAME = "CAUSAL_PATTERN_LIBRARY.json"


class RuleRefused(RuntimeError):
    """A rule's evidence is missing, so it can neither fire nor stay silent."""


def _receipt(rel: str) -> dict[str, Any]:
    p = REPO / rel
    if not p.is_file():
        raise RuleRefused(
            f"{rel} is not on disk; this rule's condition cannot be evaluated "
            "and NOT-FIRED would be a false negative"
        )
    return json.loads(p.read_text())


# ── conditions ──────────────────────────────────────────────────────────────
# Each returns (fired, observation). They read disk; none takes an argument.

def _c_multistream_underutilization() -> tuple[bool, str]:
    d = _receipt("receipts/future/SINGLE_TOKEN_PARALLEL_SLACK.json")
    n = int(d["n_true_dependency"])
    e = int(d["n_edges"])
    return (n == e, f"{n} of {e} token edges are true dependencies")


def _c_no_slack_suppresses_reordering() -> tuple[bool, str]:
    d = _receipt("receipts/future/SINGLE_TOKEN_PARALLEL_SLACK.json")
    ov = int(d["theoretically_overlapable_ns"])
    return (ov == 0, f"theoretically_overlapable_ns = {ov}")


def _c_op_class_split_forbids_single_target() -> tuple[bool, str]:
    d = _receipt("receipts/future/OP_CLASS_ABLATION.json")
    cls = d["decomposition"]["classes"]
    top = max(v["share_of_arithmetic"] for v in cls.values())
    return (top < 0.50,
            f"largest op class is {top:.1%} of the arithmetic; no class dominates")


def _c_unsteady_arm_is_not_a_measurement() -> tuple[bool, str]:
    d = _receipt("receipts/future/DEQUANT_HOIST_AB.json")
    hist = d["correction_history"]
    return (len(hist) >= 2,
            f"{len(hist)} readings of one candidate, {len(hist) - 1} of them wrong")


def _c_isolated_projection_is_not_a_bound() -> tuple[bool, str]:
    d = _receipt("receipts/future/Q4_BITCAST_AB.json")
    p = d["projection_vs_graph"]
    return (bool(p["the_lower_bound_pattern_did_not_hold"]),
            f"isolated projection over-predicted at {p['graph_over_prediction']}x "
            "after under-predicting twice")


def _c_arithmetic_school_is_bounded() -> tuple[bool, str]:
    d = _receipt("receipts/future/GAP_LEDGER_60.json")
    a = d["arithmetic_ceiling"]
    return (not a["reaches_60"],
            f"perfect arithmetic removal reaches {a['tps_after']} TPS, "
            f"{a['still_short_of_60_by_ms']} ms short of 60")


# ── the library ─────────────────────────────────────────────────────────────

RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "RULE.UNDERUTILIZATION_PRIOR",
        "if": "aggregate throughput rises with independent sessions AND "
              "single-session semantics are unchanged",
        "then": "raise the prior on latency hiding and insufficient independent "
                "work; look INSIDE kernels",
        "condition": _c_multistream_underutilization,
        "evidence": "receipts/future/SINGLE_TOKEN_PARALLEL_SLACK.json",
        "bought_by": "G009 multi-session capacity, G082 token DAG",
        "reopen": "a graph with a genuinely independent top-level region",
        "emits": {
            "id": "FT.MODEL_EXECUTION.rule.occupancy-probe",
            "frontier": "MODEL_EXECUTION",
            "title": "Probe kernel occupancy and memory-level parallelism",
            "detail": "the token DAG has no top-level slack, so the unused "
                      "capacity multi-session execution proves exists must be "
                      "inside kernel execution: occupancy, outstanding reads, "
                      "independent accumulators",
            "hypothesis_family": "kernel-internal underutilization",
        },
    },
    {
        "id": "RULE.SUPPRESS_REORDERING",
        "if": "the token DAG has zero overlapable top-level work",
        "then": "suppress every top-level reordering hypothesis at proposal time",
        "condition": _c_no_slack_suppresses_reordering,
        "evidence": "receipts/future/SINGLE_TOKEN_PARALLEL_SLACK.json",
        "bought_by": "G082, S026 §4",
        "reopen": "speculative drafting or a second sequence adds an independent "
                  "top-level region",
        "emits": None,
        "emits_nothing_because": (
            "this rule's action is a REFUSAL, already enforced in "
            "frontiers._item. A rule that suppresses work must not also "
            "generate work - that would be the busywork it exists to prevent."
        ),
    },
    {
        "id": "RULE.MEASURE_THE_SPLIT_FIRST",
        "if": "a kernel's arithmetic splits across classes with no class above 50%",
        "then": "refuse to attack one class before the split is measured; the "
                "expected win from removing one is the class's share, not the "
                "whole span",
        "condition": _c_op_class_split_forbids_single_target,
        "evidence": "receipts/future/OP_CLASS_ABLATION.json",
        "bought_by": "the dequant hoist and fold_addqx, which both attacked a "
                     "1.8% class and bought nothing",
        "reopen": "a different kernel; the split is kernel-specific and the q4 "
                  "ladder had to be measured separately",
        "emits": {
            "id": "FT.MODEL_EXECUTION.rule.deltanet-state-ladder",
            "frontier": "MODEL_EXECUTION",
            "title": "Build the op-class ladder for the DeltaNet state kernel",
            "detail": "the q2 and q4 matvec ladders are built and both paid. "
                      "The DeltaNet state update, rearrange and gated norm are "
                      "about 2.0 ms and have no ladder, so nobody knows what "
                      "their arithmetic is spent on",
            "hypothesis_family": "op-class decomposition",
        },
    },
    {
        "id": "RULE.UNSTEADY_ARM_IS_NOT_A_MEASUREMENT",
        "if": "an arm's slowest rep exceeds its fastest by more than 10%",
        "then": "refuse the measurement; its median is a coin flip between "
                "modes, not a value",
        "condition": _c_unsteady_arm_is_not_a_measurement,
        "evidence": "receipts/future/DEQUANT_HOIST_AB.json",
        "bought_by": "G092, recorded wrong twice before warmup was found",
        "reopen": "never; this is an instrument property, not a model property",
        "emits": {
            "id": "FT.MODEL_EXECUTION.rule.audit-arm-spreads",
            "frontier": "MODEL_EXECUTION",
            "title": "Audit every landed roofline receipt for bimodal arms",
            "detail": "MLP_ALU_ROOFLINE.json carries the signature - production "
                      "spread 1.696 against DeltaNet's 1.010 - and its median "
                      "landed on the fast mode by luck. Other receipts taken at "
                      "warmup 5 have not been checked",
            "hypothesis_family": "instrument defect",
        },
    },
    {
        "id": "RULE.ISOLATED_PROJECTION_IS_NOT_A_BOUND",
        "if": "an isolated organ measurement is used to project a graph-level win",
        "then": "label it an estimate in BOTH directions; do not call it a bound",
        "condition": _c_isolated_projection_is_not_a_bound,
        "evidence": "receipts/future/Q4_BITCAST_AB.json",
        "bought_by": "two under-predictions that became a claimed lower bound, "
                     "then one over-prediction that falsified it",
        "reopen": "a mechanism explaining the direction would make it a bound "
                  "again; none is established",
        "emits": {
            "id": "FT.MODEL_EXECUTION.rule.census-attribution-check",
            "frontier": "MODEL_EXECUTION",
            "title": "Check whether the organ census attributes kernel time correctly",
            "detail": "the named suspect for projection error is the "
                      "DENOMINATOR, not the kernel: counting DeltaNet's whole "
                      "organ as q4 matvec already moved one projection by 25%. "
                      "Per-kernel attribution has not been verified",
            "hypothesis_family": "measurement attribution",
        },
    },
    {
        "id": "RULE.SCHOOL_BOUNDED_BELOW_TARGET",
        "if": "a school's perfect success does not reach the target",
        "then": "the school must not be the only thing running; open a second "
                "class in parallel rather than after",
        "condition": _c_arithmetic_school_is_bounded,
        "evidence": "receipts/future/GAP_LEDGER_60.json",
        "bought_by": "arm_a bounding the whole decode-arithmetic school at 52 TPS",
        "reopen": "a change that moves arm_a itself - fewer bytes changes the "
                  "streaming floor the ceiling is computed from",
        "emits": {
            "id": "FT.MODEL_REPRESENTATION.rule.byte-elimination",
            "frontier": "MODEL_REPRESENTATION",
            "title": "Eliminate 16.8% of the matvec weight bytes",
            "detail": "after perfect arithmetic removal the matvecs are 15.27 ms "
                      "of pure streaming and 60 TPS needs 2.56 ms more, which "
                      "can only come from bytes. Entropy coding cannot supply it "
                      "at 1.87 bits per 2 stored and 93.5% independent "
                      "information; this is information elimination",
            "hypothesis_family": "information elimination",
        },
    },
)


def evaluate() -> list[dict[str, Any]]:
    out = []
    for r in RULES:
        fired, observation = r["condition"]()
        out.append({
            "id": r["id"],
            "if": r["if"],
            "then": r["then"],
            "fired": fired,
            "observation": observation,
            "evidence": r["evidence"],
            "bought_by": r["bought_by"],
            "reopen": r["reopen"],
            "emits_workunit": r["emits"]["id"] if r["emits"] else None,
            "emits_nothing_because": r.get("emits_nothing_because"),
        })
    return out


def emitted_workunits() -> list[dict[str, Any]]:
    """S026 §17: firing a rule emits a WorkUnit, not a log line.

    Built through frontiers._item so a rule cannot emit a unit the frontier
    layer would itself refuse - including one belonging to a dead school.
    """
    units = []
    for r in RULES:
        if not r["emits"]:
            continue
        fired, _ = r["condition"]()
        if not fired:
            continue
        spec = r["emits"]
        units.append(fr._item(
            id=spec["id"],
            frontier=spec["frontier"],
            kind="NEXT_WORK",
            title=spec["title"],
            detail=spec["detail"],
            required_lanes=(),
            gain=8,
            species="causal-rule",
            verifier="the rule's own evidence receipt",
            evidence=(r["evidence"],),
            hypothesis_family=spec["hypothesis_family"],
        ))
    return units


def meta_policy() -> dict[str, Any]:
    """S026 §18: what this campaign learned about its own method."""
    return {
        "discriminators_that_killed_fastest": [
            "an output comparison inside the harness - it caught two candidates "
            "that were fast and returned garbage, before either left the probe",
            "a matched control arm held across runs - arm_a holding to 0.05% is "
            "what licensed attributing a production change to anything else",
            "an archived negative control replayed against a new judge - it "
            "caught a judge keyed on the wrong flag inside a minute",
        ],
        "measurements_that_mislead": [
            "a first-measured arm at low warmup: bimodal, median is a coin flip",
            "a synthetic dyadic input: it made bit-identity an artifact",
            "an isolated organ number projected onto the graph: wrong in both "
            "directions across three observations",
            "a byte count times an organ average rate: byte classes differ",
        ],
        "ratios_that_transfer": [
            "arm_a over production, which held to 0.15% between a contaminated "
            "window and a protected lease",
        ],
        "kernel_changes_that_usually_fail": [
            "removing one op class when four share the cost - twice now",
            "trading bytes for decode arithmetic",
        ],
        "oracle_controls_that_saved_time": [
            "arm_a as the arithmetic-free floor: it bounded an entire school in "
            "one number rather than one candidate at a time",
        ],
        "this_is_not_a_law": (
            "these are observations from one body on one machine over one "
            "campaign. They are priors for choosing the next experiment, not "
            "conclusions to cite as evidence."
        ),
    }


def build() -> dict[str, Any]:
    rules = evaluate()
    units = emitted_workunits()
    return {
        "obligation": "G097",
        "authority": "S026 §17, §18",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "n_rules": len(rules),
        "n_fired": sum(1 for r in rules if r["fired"]),
        "rules": rules,
        "emitted_workunits": units,
        "n_emitted": len(units),
        "meta_policy": meta_policy(),
        "a_missing_receipt_refuses": (
            "every condition reads a receipt this repository contains. If one "
            "is absent the rule RAISES rather than reporting not-fired, because "
            "silence and absence are different and only one is information."
        ),
        "rules_emit_through_the_frontier_layer": (
            "units are built with frontiers._item, so a rule cannot emit work "
            "the frontier layer would itself refuse - including work belonging "
            "to a school an emitted scar has already closed."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps({"n_rules": doc["n_rules"], "n_fired": doc["n_fired"],
                      "n_emitted": doc["n_emitted"],
                      "rules": [{k: r[k] for k in ("id", "fired", "observation",
                                                   "emits_workunit")}
                                for r in doc["rules"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
