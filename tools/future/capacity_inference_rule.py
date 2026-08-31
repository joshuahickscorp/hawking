#!/usr/bin/env python3
"""G090: the cross-stream capacity rule, executable.

S025 §35: the user should not have needed to interpret G009. When HCLI sees

    one stream ~361 GB/s
    three streams ~580 GB/s

it should itself infer that hardware capacity exceeds single-stream exposure and
generate the intra-token concurrency, dependency and occupancy questions. That
inference should not wait for a steer next time.

S025 §36 states the general law:

    when N independent workloads materially increase aggregate physical
    throughput while individual workload semantics stay comparable, infer
    SINGLE-WORKLOAD UNDERUTILIZATION - and then generate discriminators for
    dependency, occupancy, latency hiding, memory-level parallelism and
    scheduler topology.

Scope carefully. Do NOT automatically assume one cause: the rule generates
COMPETING explanations, and naming the winner before measuring is the error the
whole tree exists to prevent.

§37 adds the experiment-policy lesson from how this campaign actually got here:
bandwidth looked low, contiguity was killed, dispatch count was killed,
bytes-per-dispatch was killed, and only then did concurrency expose capacity.
Next time, ask whether N independent copies expose more capacity BEFORE spending
hours polishing static streaming.

    python3 tools/future/capacity_inference_rule.py --fire
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/capacity_inference_rule.py"
RECEIPT_NAME = "CAPACITY_INFERENCE_RULE.json"

CONC_REL = "receipts/future/RESIDENT_CONCURRENCY_MEASURED.json"

UNDERUTILIZATION = "SINGLE_WORKLOAD_UNDERUTILIZATION"
NO_INFERENCE = "NO_INFERENCE"

# The same bar the concurrency ladder uses. A gain inside noise is not capacity.
MATERIAL_RATIO = 1.05

GENERATED_QUESTIONS = (
    ("dependency", "is the workload a chain? build the single-token dependency "
                   "DAG and count edges that are not true data dependencies"),
    ("occupancy", "does the kernel launch enough parallel work? sweep threadgroup "
                  "and grid at fixed bytes and fixed arithmetic"),
    ("latency_hiding", "is latency exposed rather than hidden? this is the PARENT "
                       "of the others and must be split, never answered"),
    ("memory_level_parallelism", "how many independent loads is each thread "
                                 "holding? deepen per-thread load pipelining at "
                                 "identical bytes"),
    ("scheduler_topology", "does submission serialise? same work across two "
                           "command queues versus one, output bit-identical"),
)


class RuleRefused(RuntimeError):
    """An inference from a comparison that was not comparable."""


def fires_on(levels: list[Mapping[str, Any]], *, semantics_comparable: bool) -> dict[str, Any]:
    """Does the rule fire? Semantics must be comparable or it refuses to look."""
    if not semantics_comparable:
        raise RuleRefused(
            "the workloads did not do comparable work, so more aggregate "
            "throughput is not evidence of underutilisation - it may just be "
            "different work"
        )
    by_n = {int(r["concurrency"]): r for r in levels}
    if 1 not in by_n:
        raise RuleRefused("no n=1 baseline; there is nothing to be under-utilised against")
    base = by_n[1].get("aggregate_decode_tps")
    if not base:
        raise RuleRefused("the n=1 baseline produced nothing measurable")
    ratios = {
        n: (r.get("aggregate_decode_tps") or 0) / base
        for n, r in by_n.items()
    }
    best = max(ratios, key=lambda k: ratios[k])
    fired = best > 1 and ratios[best] > MATERIAL_RATIO
    return {
        "fired": fired,
        "inference": UNDERUTILIZATION if fired else NO_INFERENCE,
        "ratios": {str(k): round(v, 4) for k, v in ratios.items()},
        "bar": MATERIAL_RATIO,
        "why": (
            f"n={best} reaches {ratios[best]:.2f}x the n=1 aggregate, above the "
            f"{MATERIAL_RATIO} bar, at comparable semantics"
            if fired else
            "no level materially exceeds n=1, so a single workload is not "
            "demonstrably under-utilising the machine"
        ),
    }


def questions() -> list[dict[str, Any]]:
    return [
        {"class": name, "discriminator": disc, "status": "GENERATED"}
        for name, disc in GENERATED_QUESTIONS
    ]


def experiment_policy_lesson() -> dict[str, Any]:
    """§37. How this campaign actually got here, as reusable policy."""
    return {
        "sequence_taken": [
            "bandwidth looked low",
            "contiguity killed (MLP_REGION_FALSIFIER, 331.6 -> 332.2 GB/s)",
            "dispatch count killed (DISPATCH_CEREMONY)",
            "bytes-per-dispatch killed (DISPATCH_SIZE_SWEEP)",
            "concurrency exposed capacity (RESIDENT_CONCURRENCY_MEASURED)",
        ],
        "policy": (
            "before spending hours polishing static streaming, ask whether N "
            "independent copies expose more capacity. The cross-stream probe is "
            "cheap, it is a control rather than a candidate, and it would have "
            "reordered this entire sequence."
        ),
        "cost_of_not_having_it": (
            "three schools were killed one at a time before the control that "
            "reframes all of them was run"
        ),
    }


def fire() -> dict[str, Any]:
    path = REPO / CONC_REL
    if not path.is_file():
        return {"fired": False, "inference": NO_INFERENCE,
                "why": f"{CONC_REL} is not on disk; nothing to infer from"}
    doc = json.loads(path.read_text())
    out = fires_on(doc["levels"], semantics_comparable=True)
    out["generated_questions"] = questions() if out["fired"] else []
    out["does_not_name_a_cause"] = (
        "the rule produces COMPETING explanations. Naming the winner before "
        "measuring is the error the capacity tree exists to prevent, and "
        "'concurrency helped so it must be serial dependency' is that error."
    )
    return out


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.capacity_inference_rule.v1",
        "version": 1,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "law": (
            "When N independent workloads materially increase aggregate physical "
            "throughput while individual workload semantics stay comparable, "
            "infer SINGLE_WORKLOAD_UNDERUTILIZATION and generate discriminators "
            "for dependency, occupancy, latency hiding, memory-level parallelism "
            "and scheduler topology."
        ),
        "fired_on_g009": fire(),
        "experiment_policy": experiment_policy_lesson(),
        "claim_boundary": (
            "Static sidecar artifact. The rule fires from a landed receipt and "
            "generates QUESTIONS; it measures nothing and answers nothing. It "
            "refuses to fire when the workloads were not doing comparable work, "
            "because more aggregate throughput from different work is not "
            "underutilisation."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--fire", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps(doc["fired_on_g009"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
