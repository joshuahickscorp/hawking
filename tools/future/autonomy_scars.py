"""AUTONOMY SCARS — defects in the resident's own scheduling, kept as science.

Negative science covers representation hypotheses. It covered nothing about the
ORCHESTRATOR, and every defect in this campaign's autonomy loop was of that
second kind: the machinery existed and the evidence model around it was wrong.
Those are exactly the failures that survive a rewrite, because nothing records
them and the next scheduler schema reinvents the same mistake.

Each entry is a defect that actually fired, with the symptom that hid it. The
symptom matters more than the fix: all four looked healthy from outside.

    python3 tools/future/autonomy_scars.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from pathlib import Path
from typing import Any

from tools.future._common import REPO, write_receipt

RECEIPT = "AUTONOMY_SCARS.json"
SCHEMA = "hawking.future.autonomy_scars.v1"

SCARS: tuple[dict[str, Any], ...] = (
    {
        "id": "STATIC_LANE_TAXONOMY_DIVERGED_FROM_FRONTIER",
        "family": "scheduler_taxonomy",
        "verdict": "BURIED",
        "claim_refuted": (
            "that a scheduler may keep its own list of resource lanes alongside the "
            "frontier's"
        ),
        "what_happened": (
            "autonomy_run declared AVAILABLE_LANES = CPU_ANALYSIS, CPU_VERIFY, "
            "CPU_REPRESENTATION, DISK_IO. The frontier matches required_lanes <= "
            "available against CPU, ANALYSIS, REPRESENTATION, SIMULATION, ODYSSEY, "
            "TOOLING. No item required an invented name, so the subset test was false "
            "for all 31 NEXT_WORK items and next_work() and refill() returned an empty "
            "list on every call, in every run, from the day the driver was written."
        ),
        "why_it_hid": (
            "the loop was never short of work -- it queued capabilities and specimens "
            "directly -- so it looked busy and healthy, and the trial that scored it "
            "had already passed. Nothing reads as broken when a filter silently "
            "matches nothing."
        ),
        "cost": "the frontier's own work never ran once, while the driver documented "
                "itself as deriving work from the frontier",
        "law": "a scheduler derives its lane vocabulary from the authority; it never restates it",
        "reopen_condition": "never; a second lane list is the defect itself",
        "regression_test": "tools/future/test_autonomy_run.py::"
                           "test_the_driver_speaks_the_lane_vocabulary_the_frontier_actually_uses",
    },
    {
        "id": "SCAR_LOOKUP_ON_IMPLEMENTATION_NAMES",
        "family": "negative_science_keying",
        "verdict": "BURIED",
        "claim_refuted": "that a module name can be asked whether it is a dead hypothesis family",
        "what_happened": (
            "the loop consulted refuse_if_dead with hypothesis_family set to a python "
            "module name. The index keys on hypothesis, representation, organ, model, "
            "machine and mechanism, so the question was a category error and the only "
            "possible answer was no. 71 consultations, 0 refusals."
        ),
        "why_it_hid": (
            "consultations were counted and reported, so the loop appeared to be using "
            "negative science. A counter of questions asked is not a measure of "
            "questions answered."
        ),
        "cost": "the resident could not reject anything on evidence, and did not know it",
        "law": "negative science is keyed by hypothesis semantics, never by implementation identity",
        "reopen_condition": "never",
        "regression_test": "tools/future/test_autonomy_run.py::"
                           "test_proposal_space_is_the_fixed_taxonomy_not_the_set_of_dead_ideas",
    },
    {
        "id": "DECLARED_CAPABILITY_READ_AS_EXECUTED_CAPABILITY",
        "family": "self_certification",
        "verdict": "BURIED",
        "claim_refuted": "that naming a tool in source is evidence the tool was driven",
        "what_happened": (
            "the Odyssey gate's resident-schedulability probe accepted any AST Assign "
            "containing a tool's path as proof a module drives it. odyssey_launch names "
            "Doctor's scripts in an `owned = [...]` literal, so the gate certified "
            "ITSELF as Doctor's resident driver and scored schedule and frontier true."
        ),
        "why_it_hid": (
            "it made a criterion look closer to met, which is the direction nobody "
            "audits"
        ),
        "cost": "a launch criterion would have passed on the gate's own declaration",
        "law": "DECLARED CAPABILITY != EXECUTED CAPABILITY. Require invocation, a "
               "resulting receipt, and the link between them.",
        "reopen_condition": "never",
        "regression_test": "tools/future/test_odyssey_launch.py::"
                           "test_negative_control_the_gate_cannot_certify_itself_as_the_driver",
    },
    {
        "id": "STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM",
        "family": "evidence_compression",
        "verdict": "BURIED",
        "claim_refuted": (
            "that a failure status naming a subsystem is evidence that subsystem failed"
        ),
        "what_happened": (
            "flash_meta_teacher_trace writes status BLOCKED_NO_METAL_GPU for ANY "
            "dense_source_bf16_prefix_initialization error, and the claim_boundary text "
            "asserts the host has no Metal-capable GPU. That sentence was then carried "
            "across the campaign as a hardware fact, including into this sidecar's own "
            "autonomy driver. The host is an M3 Ultra; the device enumerates from "
            "Swift, from the exact metal crate the runtime uses, and shaders compile "
            "from source."
        ),
        "why_it_hid": (
            "the status was specific, plausible, and repeated. Specificity reads as "
            "evidence."
        ),
        "cost": (
            "gate 2 of the Flash meta funnel -- and every family behind it -- was "
            "classified as waiting on hardware that was present the whole time"
        ),
        "law": "STATUS LABELS ARE HYPOTHESES UNTIL THEIR CAUSAL CLAIM IS VERIFIED. "
               "A failure receipt records the exact underlying error and the probes "
               "that would separate the candidate causes.",
        "reopen_condition": (
            "the specific process context of the original failure is still "
            "unidentified; that diagnosis is open work, not a closed scar"
        ),
        "regression_test": "tools/future/test_metal_reachability.py::"
                           "test_the_hardcoded_boundary_status_is_recorded_as_a_negative_finding",
    },
)

SISTER_SYMPTOMS: tuple[str, ...] = (
    "\"model missing\" may be a stale hardcoded path -- three Odyssey tools pointed at "
    "a directory that had moved, while the 52GB parent sat on another volume",
    "\"no work\" may be a scheduler taxonomy mismatch",
    "\"Doctor driven\" may be self-certification",
    "\"no GPU\" may be error laundering",
    "\"retired specimen\" may mean historically retired, not scientifically unusable",
)


def scars() -> list[dict[str, Any]]:
    return [dict(s) for s in SCARS]


def missing_regression_tests() -> list[dict[str, str]]:
    """A scar whose regression test does not exist is a story, not a guard."""
    out = []
    for scar in SCARS:
        rel, _, name = str(scar["regression_test"]).partition("::")
        path = REPO / rel
        present = path.is_file() and (not name or f"def {name}(" in path.read_text(errors="replace"))
        if not present:
            out.append({"id": scar["id"], "regression_test": scar["regression_test"]})
    return out


def build() -> Path:
    missing = missing_regression_tests()
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Defects in the resident's own scheduling and evidence model, kept as "
            "negative science so the next scheduler schema does not reinvent them."
        ),
        "evidence_class": "STATIC_ONLY",
        "why_this_exists": (
            "negative science covered representation hypotheses and nothing about the "
            "orchestrator. Every autonomy defect this campaign found was of the second "
            "kind: the machinery existed and the evidence model around it was wrong."
        ),
        "n_scars": len(SCARS),
        "scars": scars(),
        "general_law": (
            "STATUS LABELS ARE HYPOTHESES UNTIL THEIR CAUSAL CLAIM IS VERIFIED."
        ),
        "sister_symptoms": list(SISTER_SYMPTOMS),
        "scars_without_a_regression_test": missing,
        "recovered_implementation": [
            "tools/future/negative_index.py keys scars by hypothesis semantics; these "
            "are orchestration defects and do not belong in that keyspace",
        ],
        "gaps_closed": ["autonomy defects were fixed but never recorded as science"],
        "negative_findings": [
            "all four defects looked healthy from outside; none produced an error",
        ],
        "resident_callable": {
            "entry_point": "tools.future.autonomy_scars.scars()",
            "workunit": "one CPU_ANALYSIS unit; consulted before a scheduler change",
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.HCLI_SELF.emit-workunits",
            "fails_closed": "a scar whose regression test is absent is reported, not hidden",
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/autonomy_scars.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    doc = json.loads(out.read_text())
    print(out)
    print(json.dumps({"n_scars": doc["n_scars"],
                      "without_regression_test": doc["scars_without_a_regression_test"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
