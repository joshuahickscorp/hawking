"""FLASH_ORGAN_WORKGRAPHS — one funnel WorkGraph per Gravity school.

flash_schools.py already schedules a single school alone and ranks its
candidates. meta_funnel.py already refuses to skip a gate. workgraph.py
already schedules units with hardware-only SLEEPING. None of those is the
dependency structure that lets fourteen organ schools advance in parallel
over the real funnel, with Gate 2 (real teacher fit) honestly asleep until
a Flash teacher corpus exists, and every node past a sleeper UNREACHABLE
rather than pending.

This module extends those three. It does not fork the school catalog, the
nine meta gates, or the fourteen resource lanes. It adds the two funnel
stages they do not have (tiny synthetic sanity, state stability), the
per-school graph, and the sleep/unreachable distinction for missing
receipts — which workgraph.make_unit cannot express, because its SLEEPING
is reserved for unqualified hardware.

    python3 tools/future/flash_organ_workgraphs.py --build
    python3 -m pytest tools/future/test_flash_organ_workgraphs.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.future._common import REPO, RECEIPTS, git, load_json, write_receipt
from tools.future import flash_schools as fs
from tools.future import meta_funnel as mf
from tools.future import teacher_corpus as tc
from tools.future import workgraph as wg
from tools.future import workunit_species as wus


RECEIPT = "FLASH_ORGAN_WORKGRAPHS.json"
SCHEMA = "hawking.future.flash_organ_workgraphs.v1"
VERSION = 1
RECORDED_BY = "tools/future/flash_organ_workgraphs.py"

# Status vocabulary. "pending" is refused for anything past a sleeper —
# that is the whole point of this graph.
ST_COMPLETE = "COMPLETE"
ST_READY = "READY"
ST_AWAITING = "AWAITING_PREDECESSOR"
ST_SLEEPING = "SLEEPING"
ST_UNREACHABLE = "UNREACHABLE"

GRAPH_STATUSES = frozenset({ST_COMPLETE, ST_READY, ST_AWAITING, ST_SLEEPING, ST_UNREACHABLE})

# Schools whose census family tuple is empty by declaration. They are
# function organs, not missing weight families. They still get a graph.
FUNCTION_ORGANS: frozenset[str] = frozenset(
    sid for sid, fams in fs.SCHOOL_CENSUS_FAMILIES.items() if not fams
)

# Paths that would be a real Flash teacher corpus if they existed on disk.
# TEACHER_CORPUS_CONTRACT.json is a validator, not a capture.
TEACHER_CORPUS_CANDIDATES: tuple[str, ...] = (
    "receipts/future/FLASH_TEACHER_CORPUS.json",
    "receipts/headless/FLASH_TEACHER_CORPUS.json",
    "receipts/headless/FLASH_TEACHER_FORCED_CAPTURE.json",
    "receipts/FLASH_TEACHER_CORPUS.json",
    "workspace/campaign/governance/odyssey/resources/teacher_traces/FLASH_TEACHER_TRACE_MANIFEST.json",
)

# Named receipts a later stage would consume. Presence on disk is necessary
# for COMPLETE, never sufficient once an ancestor sleeps.
STAGE_OUTPUT_RECEIPTS: dict[str, tuple[str, ...]] = {
    "analytic_structure": ("receipts/future/FLASH_ORGAN_SCHOOLS.json",),
    "tiny_synthetic_sanity": (),
    "real_teacher_fit": TEACHER_CORPUS_CANDIDATES,
    "held_out_numerical": (
        "receipts/future/FLASH_HELDOUT_NUMERICAL.json",
        "receipts/headless/FLASH_HELDOUT_NUMERICAL.json",
    ),
    "route_stability": (
        "receipts/headless/FLASH_NOETIC_ROUTER_SELECTION.json",
        "receipts/future/evidence/FLASH_NOETIC_ROUTER_SELECTION.json",
    ),
    "state_stability": (
        "receipts/future/FLASH_STATE_STABILITY.json",
        "receipts/headless/FLASH_STATE_STABILITY.json",
        "receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_ACCEPTED.json",
    ),
    "logit_token": (
        "receipts/future/FLASH_LOGIT_TOKEN.json",
        "receipts/headless/FLASH_COMPLETE_TOKEN_NATIVE_ATTEMPT.json",
    ),
    "capability_subset": (
        "receipts/future/FLASH_BOUNDED_CAPABILITY.json",
        "receipts/headless/FLASH_BOUNDED_CAPABILITY.json",
    ),
    "physical_nr_lowering": (
        "receipts/future/FLASH_NR_COMPLETE.json",
        "receipts/headless/FLASH_COMPLETE_V2.nr.json",
    ),
    "native_nx": (
        "receipts/future/FLASH_NX_COMPLETENESS_AUDIT.json",
        "receipts/headless/FLASH_COMPLETE_V0.nx.json",
        "receipts/headless/FLASH_NEXT_NOETIC_EXECUTABLE.json",
    ),
    "complete_ebpw": (
        "receipts/future/EBPW_CATEGORY_VALIDATOR.json",
        "receipts/headless/FLASH_EBPW_BUDGET.json",
    ),
}

CLAIM_BOUNDARY = (
    "Static sidecar artifact. One WorkGraph per Flash Gravity school over the "
    "real funnel. No hardware measurement. Gate 2 (real teacher fit) SLEEPS "
    "until a Flash teacher corpus exists on disk; nodes past a sleeper are "
    "UNREACHABLE, not pending. COMPLETE is refused without a receipt on disk."
)


class OrganWorkGraphError(ValueError):
    """Base error for the organ-school funnel graphs."""


class CompleteWithoutReceipt(OrganWorkGraphError):
    """COMPLETE is refused unless the named receipt is a real file."""


class NodeAsleep(OrganWorkGraphError):
    """A SLEEPING node was asked to run. It does not run."""

    def __init__(self, node_id: str, wake_condition: str) -> None:
        self.node_id = node_id
        self.wake_condition = wake_condition
        super().__init__(
            f"{node_id} is SLEEPING; wake_condition={wake_condition!r}; "
            "refusing to run"
        )


class NodeUnreachable(OrganWorkGraphError):
    """A node past a sleeper was asked to run. It does not run."""

    def __init__(self, node_id: str, because: str) -> None:
        self.node_id = node_id
        self.because = because
        super().__init__(f"{node_id} is UNREACHABLE because {because}")


class EmptyCensusFamily(OrganWorkGraphError):
    """A weight organ whose census family is empty is not a complete graph."""


# ---------------------------------------------------------------------------
# Funnel stages. Eleven nodes. Nine recovered from meta_funnel; two are the gap.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunnelStage:
    index: int
    name: str
    title: str
    cost_class: str
    required_input: str
    verifier: str
    sleeps_when: str
    wake_condition: str
    passing_proves: str
    passing_does_not_prove: str
    meta_funnel_gate: int | None
    in_process_input: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "title": self.title,
            "cost_class": self.cost_class,
            "required_input": self.required_input,
            "verifier": self.verifier,
            "sleeps_when": self.sleeps_when,
            "wake_condition": self.wake_condition,
            "passing_proves": self.passing_proves,
            "passing_does_not_prove": self.passing_does_not_prove,
            "meta_funnel_gate": self.meta_funnel_gate,
            "in_process_input": self.in_process_input,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }


FUNNEL_STAGES: tuple[FunnelStage, ...] = (
    FunnelStage(
        1,
        "analytic_structure",
        "analytic structure",
        "CHEAP_ANALYTICAL",
        "allocation_plan",
        "tools.future.meta_funnel._eval_analytical + flash_schools.schedule_school",
        "the school cannot emit an allocation plan (unknown school, or empty census family on a weight organ)",
        "a Gravity school with a well-formed allocation_plan (FLASH_ORGAN_SCHOOLS or in-process schedule_school)",
        "The hypothesis is a well-formed allocation of TOTAL EXECUTABLE INFORMATION.",
        "Teacher fit, synthetic numerical sanity, routes, state, tokens, capability, NR, NX, or EBPW.",
        1,
        True,
    ),
    FunnelStage(
        2,
        "tiny_synthetic_sanity",
        "tiny synthetic sanity",
        "CHEAP_SYNTHETIC_CPU",
        "synthetic_probe",
        "tools.future.flash_organ_workgraphs.run_tiny_synthetic_sanity",
        "analytic_structure cannot produce live candidates to sanity-check",
        "analytic_structure READY or COMPLETE for this school (in-process schedule_school is enough)",
        "Candidates are well-formed Gravity programs and the allocation plan survives the analytical screen. No teacher X.",
        "Real teacher fit, held-out fidelity, or anything measured on captured activations.",
        None,
        True,
    ),
    FunnelStage(
        3,
        "real_teacher_fit",
        "real teacher fit",
        "REAL_TEACHER_CPU",
        "teacher_corpus",
        "tools.future.teacher_corpus.validate_corpus + meta_funnel._eval_teacher",
        "no Flash teacher-forced / captured-activation corpus exists on disk (NOT_BUILT)",
        "a Flash teacher corpus receipt on disk, bound to Qwen3.8-Flash-Next, accepted by teacher_corpus.validate_corpus",
        "The candidate reconstructs or predicts on the teacher corpus it was fit to.",
        "Held-out numerical validity, route identity, state identity, tokens, capability, or anything physical.",
        2,
        False,
    ),
    FunnelStage(
        4,
        "held_out_numerical",
        "held-out numerical",
        "HELDOUT_NUMERICAL_CPU",
        "held_out_numerical",
        "tools.future.meta_funnel._eval_heldout",
        "held-out activations (disjoint from the teacher corpus) are NOT_MEASURED / absent",
        "held-out numerical receipt on disk for this school, disjoint from the teacher corpus",
        "Local numerical fidelity on unseen X for the stated organ.",
        "Route identity, state identity, complete-token argmax, capability, NR, NX, or EBPW.",
        3,
        False,
    ),
    FunnelStage(
        5,
        "route_stability",
        "route stability",
        "ROUTE_TRACE_CPU",
        "route_traces",
        "tools.future.meta_funnel._eval_routes + router_science",
        "student/teacher route traces are NOT_MEASURED, or teacher fit has not passed",
        "route-trace receipt on disk with teacher top-k identity after a passed teacher fit",
        "On the measured traces, the student selects the same experts as the teacher.",
        "Recurrent/KV state identity, logit/token identity, capability, NR, NX, or EBPW.",
        4,
        False,
    ),
    FunnelStage(
        6,
        "state_stability",
        "state stability",
        "STATE_TRACE_CPU",
        "state_traces",
        "tools.future.flash_organ_workgraphs._eval_state_stability",
        "recurrent / KV / HC state traces are NOT_MEASURED, or route stability has not passed",
        "state-trace receipt on disk showing source-approved recurrent/KV/HC semantics after route stability",
        "Organ-local recurrent or KV state agrees with the teacher on the stated probe.",
        "Logit/token identity, capability, physical NR, NX, or EBPW. State agreement is not generation.",
        None,
        False,
    ),
    FunnelStage(
        7,
        "logit_token",
        "logit/token",
        "LOGIT_TOKEN_CPU",
        "logit_token",
        "tools.future.meta_funnel._eval_logits",
        "logit/token probe is NOT_MEASURED, or state stability has not passed",
        "logit/token receipt on disk with argmax agreement after state stability",
        "Token-level identity on the stated probe (argmax / short decode).",
        "Bounded capability, a physical NR, a complete NX, or EBPW. 16 greedy tokens are not capability.",
        5,
        False,
    ),
    FunnelStage(
        8,
        "capability_subset",
        "capability subset",
        "BOUNDED_CAPABILITY_CPU",
        "bounded_capability",
        "tools.future.meta_funnel._eval_capability",
        "bounded capability suite is NOT_RUN, or logit/token has not passed",
        "bounded-capability receipt on disk matching the incumbent on every substantive axis",
        "The candidate matched the incumbent on every substantive capability axis of the stated suite.",
        "Physical NR lowering, a complete NX, protected token_ns, or EBPW.",
        6,
        False,
    ),
    FunnelStage(
        9,
        "physical_nr_lowering",
        "physical NR lowering",
        "PHYSICAL_NR_STATIC",
        "physical_nr",
        "tools.future.meta_funnel._eval_nr + flash_nr_complete",
        "no Physical NR artifact, or PLAN_ONLY / NOT_IMPLEMENTED / COMPILE_TIME_SCIENCE_ONLY",
        "a Physical NR artifact on disk that lowered without a disclosed failure (STATIC_ONLY here; no kernel)",
        "A Physical NR artifact exists and lowered without a disclosed failure.",
        "A complete NX, protected complete-token timing, or EBPW. PLAN_ONLY is a refusal, not a pass.",
        7,
        False,
    ),
    FunnelStage(
        10,
        "native_nx",
        "native NX",
        "COMPLETE_NX_STATIC",
        "complete_nx",
        "tools.future.meta_funnel._eval_nx + flash_nx_audit",
        "NX is absent, SCAFFOLD_ONLY, or SEALED_METADATA_ONLY_NOT_FOR_PROMOTION",
        "a source-independent complete NX artifact on disk as more than sealed metadata",
        "A source-independent complete NX artifact exists as more than sealed metadata.",
        "EBPW, protected TPS, or promotion. SCAFFOLD_ONLY is a refusal, not a pass.",
        8,
        False,
    ),
    FunnelStage(
        11,
        "complete_ebpw",
        "complete EBPW",
        "EBPW_ACCOUNTING_STATIC",
        "ebpw_ledger",
        "tools.future.meta_funnel._eval_ebpw + ebpw_categories",
        "complete-system executable bytes are NOT_MEASURED, null, or omit a required field",
        "a complete-system EBPW ledger on disk with every required field counted",
        "A complete-system EBPW ledger exists with every required field counted.",
        "Protected token_ns, joules, or promotion. null complete_system_bytes is a refusal, not a number.",
        9,
        False,
    ),
)
STAGES_BY_NAME = {s.name: s for s in FUNNEL_STAGES}
TEACHER_FIT_STAGE = "real_teacher_fit"
PREFIX_STAGES = frozenset({"analytic_structure", "tiny_synthetic_sanity"})


def node_id(school: str, stage: str) -> str:
    return f"{school}:{stage}"


def school_of(nid: str) -> str:
    return nid.split(":", 1)[0] if ":" in nid else nid


def stage_of(nid: str) -> str:
    return nid.split(":", 1)[1] if ":" in nid else nid


# ---------------------------------------------------------------------------
# Disk / HEAD probes. On-disk is the only COMPLETE authority.
# ---------------------------------------------------------------------------


def path_on_disk(rel: str) -> bool:
    return (REPO / rel).is_file()


@lru_cache(maxsize=None)
def path_in_head(rel: str) -> bool:
    if path_on_disk(rel):
        return True
    listed = git("ls-tree", "--name-only", "HEAD", rel)
    return any(line == rel for line in listed.splitlines())


def probe_receipt(rel: str) -> dict[str, Any]:
    return {
        "path": rel,
        "on_disk": path_on_disk(rel),
        "in_head": path_in_head(rel),
        "complete_authority": "on_disk",
    }


def first_on_disk(rels: Sequence[str]) -> str | None:
    for rel in rels:
        if path_on_disk(rel):
            return rel
    return None


def flash_teacher_corpus_state() -> dict[str, Any]:
    """A real Flash teacher corpus, not the validator contract, not a fixture.

    TEACHER_CORPUS_CONTRACT.json is a STATIC_ONLY validator. Fixtures in
    teacher_corpus.py are tagged authority=STATIC_ONLY and must not wake Gate 2.
    GLM teacher-forced findings are the wrong specimen.
    """
    hits = [probe_receipt(rel) for rel in TEACHER_CORPUS_CANDIDATES]
    on_disk = [h for h in hits if h["on_disk"]]
    in_head = [h for h in hits if h["in_head"] and not h["on_disk"]]
    contract = probe_receipt(f"receipts/future/{tc.RECEIPT}")
    glm = probe_receipt("GLM_TEACHER_FORCED_PARALLELISM_FINDINGS.json")
    present = bool(on_disk)
    return {
        "state": "PRESENT" if present else "NOT_BUILT",
        "on_disk": on_disk,
        "in_head_not_on_disk": in_head,
        "contract_receipt": contract,
        "contract_is_not_a_corpus": True,
        "glm_teacher_forced_is_wrong_specimen": True,
        "glm_probe": glm,
        "reason": (
            "Flash teacher-forced / captured-activation corpus is NOT_BUILT on disk. "
            "teacher_corpus.py is a validator; its fixtures are not a Flash corpus. "
            "GLM teacher-forced captures are a different specimen."
            if not present
            else "at least one Flash teacher corpus path is on disk; validate_corpus still has to accept it"
        ),
    }


def _eval_state_stability(_candidate: Mapping[str, Any], value: Any) -> tuple[str, str]:
    """State-stability evaluator. Same fail-closed shape as meta_funnel gates.

    Recovers the kill idea from decode_civilization recurrent-state / KV
    contracts and meta_funnel's status tokens. Does not invent a pass.
    """
    if isinstance(value, dict) and value.get("state_identity") is False:
        return "KILLED", value.get("mechanism") or "recurrent/KV state diverges from the teacher"
    return mf._eval_from_status(
        value,
        "organ-local state agrees with the teacher on the stated probe; not generation, not capability",
        "state stability failed the stated null",
    )


# ---------------------------------------------------------------------------
# Census family status. Empty is reported, never a fake-complete graph.
# ---------------------------------------------------------------------------


def census_family_status(
    school: str,
    inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if school not in fs.SCHOOL_CATALOG:
        raise fs.UnknownSchoolError(school)
    inv = dict(inventory) if inventory is not None else fs.organ_inventory()
    declared = list(fs.SCHOOL_CENSUS_FAMILIES.get(school) or ())
    source = str(inv.get("census_source") or "unavailable")
    by_family = inv.get("by_family") or {}
    present = [f for f in declared if f in by_family]
    present_with_bytes = [
        f
        for f in present
        if isinstance((by_family.get(f) or {}).get("bytes"), int)
        and int((by_family.get(f) or {}).get("bytes") or 0) > 0
    ]

    if not declared:
        return {
            "school": school,
            "kind": "function_organ",
            "empty_census_family": False,
            "graph_emitted": True,
            "looks_complete": False,
            "census_source": source,
            "declared_families": [],
            "present_families": [],
            "reason": (
                f"{school} declares no census family (function organ: "
                f"{fs.SCHOOL_ORGAN_SLUG[school]}); a graph is still emitted. "
                "Empty declaration is not a missing weight family."
            ),
        }

    if source == "unavailable":
        return {
            "school": school,
            "kind": "census_unreachable",
            "empty_census_family": False,
            "graph_emitted": True,
            "looks_complete": False,
            "census_source": source,
            "declared_families": declared,
            "present_families": [],
            "reason": (
                "FLASH_ORGAN_CENSUS is unreachable in this checkout; "
                "unreachability is not an empty family. Graph still emitted."
            ),
        }

    if not present:
        return {
            "school": school,
            "kind": "empty_census_family",
            "empty_census_family": True,
            "graph_emitted": False,
            "looks_complete": False,
            "census_source": source,
            "declared_families": declared,
            "present_families": [],
            "reason": (
                f"{school} declares census families {declared} but none are present "
                f"in a reachable census (source={source}). Refusing a complete-looking "
                "empty graph."
            ),
        }

    return {
        "school": school,
        "kind": "weight_organ",
        "empty_census_family": False,
        "graph_emitted": True,
        "looks_complete": False,
        "census_source": source,
        "declared_families": declared,
        "present_families": present,
        "present_with_bytes": present_with_bytes,
        "reason": f"{school} census families present: {present}",
    }


def allocation_plan_for_school(
    school: str,
    scheduled: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Heterogeneous plan from the school's live candidates, not a measurement."""
    organ = fs.SCHOOL_ORGAN_SLUG[school]
    scheduled = scheduled if scheduled is not None else fs.schedule_school(school)
    regions: list[dict[str, Any]] = []
    for cand in scheduled.get("candidates") or []:
        family = str(cand.get("program_family") or "")
        if not family:
            continue
        kind = "predictable_bulk"
        bits = "uniform"
        if family in {"GeneratedProgram", "SharedBasisProgram"}:
            kind, bits = "shared_generator", "shared"
        elif family in {"SparseResidualProgram"}:
            kind, bits = "sparse_residual", "sparse"
        elif family in {"LiteralTensor"} and school in {"ROUTER", "LM_HEAD", "NORMALIZATION", "DELTANET_RECURRENT_STATE", "KV_STATE"}:
            kind, bits = "capability_island", "literal"
        elif school == "ROUTER":
            kind, bits = "routing_sensitive", "premium"
        regions.append(
            {
                "kind": kind,
                "bits_class": bits,
                "family": family,
                "organ": organ,
                "candidate_id": cand.get("id"),
            }
        )
    if not regions:
        return mf._uniform_plan(school.lower(), organ)
    kinds = {r["kind"] for r in regions}
    return {
        "unit": "TOTAL_EXECUTABLE_INFORMATION",
        "forces_uniform_bpw": len(kinds) == 1,
        "regions": regions,
        "claims_complete_system": False,
        "accounting_complete": False,
        "derived_from": "flash_schools.schedule_school candidates",
        "school": school,
    }


def run_analytic_structure(school: str, inventory: Mapping[str, Any] | None = None) -> dict[str, Any]:
    scheduled = fs.schedule_school(school, inventory=inventory)
    plan = allocation_plan_for_school(school, scheduled)
    verdict, reason = mf._eval_analytical({}, plan)
    return {
        "school": school,
        "stage": "analytic_structure",
        "ran": True,
        "verdict": verdict,
        "reason": reason,
        "n_candidates": scheduled["n_candidates"],
        "n_regions": len(plan.get("regions") or []),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "not_teacher_fit": True,
    }


def run_tiny_synthetic_sanity(school: str, inventory: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Cheap CPU sanity: live candidates + analytical screen. No teacher X."""
    scheduled = fs.schedule_school(school, inventory=inventory)
    live = list(scheduled.get("candidates") or [])
    if not live:
        return {
            "school": school,
            "stage": "tiny_synthetic_sanity",
            "ran": True,
            "verdict": "REFUSED",
            "reason": "school emitted no live candidates; sanity has nothing to check",
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }
    admitted = 0
    for row in live:
        fs.admit_candidate(row)
        admitted += 1
    plan = allocation_plan_for_school(school, scheduled)
    analytic, analytic_reason = mf._eval_analytical({}, plan)
    trap = None
    literal = next((c for c in live if c.get("program_family") == "LiteralTensor"), None)
    # prove_flop_trap needs a positive control so the half-bytes trap sorts first.
    if literal is not None and int(literal.get("storage_bytes") or 0) > 0:
        trap = fs.prove_flop_trap(literal)
    return {
        "school": school,
        "stage": "tiny_synthetic_sanity",
        "ran": True,
        "verdict": "PASSED" if analytic == "PASSED" and admitted == len(live) else analytic,
        "reason": (
            f"admitted {admitted}/{len(live)} candidates; analytical {analytic}: {analytic_reason}"
        ),
        "n_candidates": len(live),
        "admitted": admitted,
        "analytic_verdict": analytic,
        "flop_trap_fired": bool(trap and trap.get("trap_loses_on_joint_cost")),
        "used_teacher_corpus": False,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "not_teacher_fit": True,
    }


# ---------------------------------------------------------------------------
# Node evaluation. Fail closed. COMPLETE requires a file on disk.
# ---------------------------------------------------------------------------


ReceiptProbe = Callable[[str], bool]


def _default_probe(rel: str) -> bool:
    return path_on_disk(rel)


def evaluate_stage_inputs(
    stage: FunnelStage,
    school: str,
    *,
    inventory: Mapping[str, Any] | None = None,
    receipt_probe: ReceiptProbe | None = None,
    teacher_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    probe = receipt_probe or _default_probe
    teacher_state = teacher_state if teacher_state is not None else flash_teacher_corpus_state()

    if stage.name == "analytic_structure":
        census = census_family_status(school, inventory)
        if census["empty_census_family"]:
            return {
                "input_state": "ABSENT",
                "present": False,
                "sleeps": True,
                "wake_condition": (
                    f"census family {census['declared_families']} present in a reachable "
                    "FLASH_ORGAN_CENSUS family_summary"
                ),
                "reason": census["reason"],
            }
        return {
            "input_state": "PRESENT",
            "present": True,
            "sleeps": False,
            "wake_condition": None,
            "reason": "allocation_plan is produced in-process by flash_schools.schedule_school",
        }

    if stage.name == "tiny_synthetic_sanity":
        return {
            "input_state": "PRESENT",
            "present": True,
            "sleeps": False,
            "wake_condition": None,
            "reason": "synthetic probe is in-process over schedule_school candidates; no teacher X",
        }

    if stage.name == "real_teacher_fit":
        present = teacher_state.get("state") == "PRESENT"
        return {
            "input_state": "PRESENT" if present else "NOT_BUILT",
            "present": present,
            "sleeps": not present,
            "wake_condition": stage.wake_condition,
            "reason": teacher_state.get("reason"),
            "teacher_state": dict(teacher_state),
        }

    rels = STAGE_OUTPUT_RECEIPTS.get(stage.name) or ()
    on_disk = [rel for rel in rels if probe(rel)]
    if on_disk:
        return {
            "input_state": "PRESENT",
            "present": True,
            "sleeps": False,
            "wake_condition": None,
            "reason": f"input receipt on disk: {on_disk[0]}",
            "receipts_on_disk": on_disk,
        }
    in_head = [rel for rel in rels if path_in_head(rel)]
    return {
        "input_state": "NOT_BUILT" if not in_head else "NOT_ON_DISK",
        "present": False,
        "sleeps": True,
        "wake_condition": stage.wake_condition,
        "reason": (
            f"required input {stage.required_input!r} has no on-disk receipt "
            f"(in_head={in_head or None}); COMPLETE authority is on-disk"
        ),
        "receipts_in_head_not_on_disk": in_head,
    }


def _output_receipt_on_disk(stage: FunnelStage, probe: ReceiptProbe) -> str | None:
    for rel in STAGE_OUTPUT_RECEIPTS.get(stage.name) or ():
        if probe(rel):
            return rel
    return None


def evaluate_node(
    school: str,
    stage: FunnelStage,
    *,
    ancestor_status: str | None,
    ancestor_id: str | None,
    sleeping_ancestor: str | None,
    inventory: Mapping[str, Any] | None = None,
    receipt_probe: ReceiptProbe | None = None,
    teacher_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One node. COMPLETE requires a receipt on disk. Sleep never runs."""
    probe = receipt_probe or _default_probe
    nid = node_id(school, stage.name)
    inputs = evaluate_stage_inputs(
        stage,
        school,
        inventory=inventory,
        receipt_probe=probe,
        teacher_state=teacher_state,
    )
    deps = [node_id(school, FUNNEL_STAGES[stage.index - 2].name)] if stage.index > 1 else []

    if sleeping_ancestor:
        status = ST_UNREACHABLE
        blocked = (
            f"ancestor {sleeping_ancestor} is SLEEPING; this node is UNREACHABLE, not pending"
        )
    elif inputs["sleeps"]:
        status = ST_SLEEPING
        blocked = inputs["reason"]
    else:
        out_rel = _output_receipt_on_disk(stage, probe)
        if out_rel:
            status = ST_COMPLETE
            blocked = None
        elif (
            ancestor_status in {ST_READY, ST_AWAITING}
            and ancestor_id
            and not stage.in_process_input
        ):
            # Later stages with present inputs still wait on a predecessor
            # that has not COMPLETED. Prefix stages share an in-process
            # schedule_school, so they stay READY rather than freezing
            # behind a receipt that this sidecar cannot write.
            status = ST_AWAITING
            blocked = (
                f"predecessor {ancestor_id} is {ancestor_status}; "
                "not a sleeper, not pending-past-sleep"
            )
        else:
            status = ST_READY
            blocked = None

    # COMPLETE without a file is a bug in this function. Belt.
    if status == ST_COMPLETE:
        out_rel = _output_receipt_on_disk(stage, probe)
        if not out_rel:
            raise CompleteWithoutReceipt(
                f"{nid}: evaluator tried to mark COMPLETE with no on-disk receipt"
            )

    schedulable_now = status == ST_READY and not sleeping_ancestor
    return {
        "id": nid,
        "school": school,
        "stage": stage.name,
        "index": stage.index,
        "title": stage.title,
        "dependencies": deps,
        "inputs": [stage.required_input],
        "required_input": stage.required_input,
        "input_state": inputs["input_state"],
        "verifier": stage.verifier,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "cost_class": stage.cost_class,
        "meta_funnel_gate": stage.meta_funnel_gate,
        "sleeps_when": stage.sleeps_when,
        "wake_condition": inputs.get("wake_condition") or stage.wake_condition,
        "status": status,
        "schedulable_now": schedulable_now,
        "blocked_reason": blocked,
        "unreachable_because": sleeping_ancestor if status == ST_UNREACHABLE else None,
        "completion_receipt": _output_receipt_on_disk(stage, probe) if status == ST_COMPLETE else None,
        "input_detail": {k: v for k, v in inputs.items() if k != "teacher_state"},
        "passing_proves": stage.passing_proves,
        "passing_does_not_prove": stage.passing_does_not_prove,
        "resource_lane": "CPU_ANALYSIS",
        "organ_slug": fs.SCHOOL_ORGAN_SLUG[school],
    }


def mark_complete(node: Mapping[str, Any], receipt_path: str | Path) -> dict[str, Any]:
    """The only COMPLETE constructor. Raises if the file is not on disk."""
    path = Path(receipt_path)
    nid = str(node.get("id") or "<no-id>")
    if node.get("status") == ST_SLEEPING:
        raise CompleteWithoutReceipt(
            f"{nid}: refusing COMPLETE on a SLEEPING node even if a file is named"
        )
    if node.get("status") == ST_UNREACHABLE:
        raise CompleteWithoutReceipt(
            f"{nid}: refusing COMPLETE on an UNREACHABLE node"
        )
    if not path.is_file():
        raise CompleteWithoutReceipt(
            f"{nid}: refusing COMPLETE; {path} is not a file on disk"
        )
    out = dict(node)
    out["status"] = ST_COMPLETE
    out["completion_receipt"] = str(path)
    out["schedulable_now"] = False
    out["blocked_reason"] = None
    return out


def run_node(
    node: Mapping[str, Any],
    *,
    inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a READY node. SLEEPING / UNREACHABLE raise, never invent a pass."""
    status = node.get("status")
    nid = str(node.get("id") or "<no-id>")
    if status == ST_SLEEPING:
        raise NodeAsleep(nid, str(node.get("wake_condition") or ""))
    if status == ST_UNREACHABLE:
        raise NodeUnreachable(nid, str(node.get("unreachable_because") or "sleeping ancestor"))
    if status not in {ST_READY, ST_AWAITING, ST_COMPLETE}:
        raise OrganWorkGraphError(f"{nid}: cannot run status {status!r}")
    school = str(node["school"])
    stage = str(node["stage"])
    if stage == "analytic_structure":
        result = run_analytic_structure(school, inventory)
    elif stage == "tiny_synthetic_sanity":
        result = run_tiny_synthetic_sanity(school, inventory)
    else:
        raise OrganWorkGraphError(
            f"{nid}: stages past the live prefix are not run by this sidecar "
            f"(status={status}; Gate 2 is the last honest runnable boundary)"
        )
    result["node_id"] = nid
    result["node_status"] = status
    result["marked_complete"] = False
    result["complete_requires_receipt_on_disk"] = True
    return result


# ---------------------------------------------------------------------------
# Per-school graph. No cross-school edges, by construction.
# ---------------------------------------------------------------------------


def build_school_graph(
    school: str,
    *,
    inventory: Mapping[str, Any] | None = None,
    receipt_probe: ReceiptProbe | None = None,
    teacher_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if school not in fs.SCHOOL_CATALOG:
        raise fs.UnknownSchoolError(
            f"{school!r} is not a Gravity school; catalog={list(fs.SCHOOL_CATALOG)}"
        )
    inv = dict(inventory) if inventory is not None else fs.organ_inventory()
    teacher_state = teacher_state if teacher_state is not None else flash_teacher_corpus_state()
    census = census_family_status(school, inv)

    if census["empty_census_family"]:
        return {
            "school": school,
            "organ_slug": fs.SCHOOL_ORGAN_SLUG[school],
            "independent": True,
            "schedulable_alone": True,
            "empty_census_family": True,
            "graph_emitted": False,
            "looks_complete": False,
            "nodes": None,
            "edges": None,
            "n_nodes": 0,
            "n_edges": 0,
            "census": census,
            "reason": census["reason"],
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
            "teacher_fit": None,
        }

    nodes: list[dict[str, Any]] = []
    sleeping_ancestor: str | None = None
    ancestor_status: str | None = None
    ancestor_id: str | None = None
    for stage in FUNNEL_STAGES:
        node = evaluate_node(
            school,
            stage,
            ancestor_status=ancestor_status,
            ancestor_id=ancestor_id,
            sleeping_ancestor=sleeping_ancestor,
            inventory=inv,
            receipt_probe=receipt_probe,
            teacher_state=teacher_state,
        )
        if node["status"] == ST_SLEEPING and sleeping_ancestor is None:
            sleeping_ancestor = node["id"]
        nodes.append(node)
        ancestor_status = node["status"]
        ancestor_id = node["id"]

    edges = [
        {"from": nodes[i]["id"], "to": nodes[i + 1]["id"], "school": school}
        for i in range(len(nodes) - 1)
    ]
    by_status: dict[str, list[str]] = {s: [] for s in sorted(GRAPH_STATUSES)}
    for n in nodes:
        by_status[n["status"]].append(n["id"])

    teacher = next(n for n in nodes if n["stage"] == TEACHER_FIT_STAGE)
    prefix = [n for n in nodes if n["stage"] in PREFIX_STAGES]
    after = [n for n in nodes if n["index"] > teacher["index"]]

    return {
        "school": school,
        "organ_slug": fs.SCHOOL_ORGAN_SLUG[school],
        "independent": True,
        "schedulable_alone": True,
        "empty_census_family": False,
        "graph_emitted": True,
        "looks_complete": False,
        "function_organ": school in FUNCTION_ORGANS,
        "census": census,
        "nodes": nodes,
        "edges": edges,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "by_status": by_status,
        "teacher_fit": {
            "id": teacher["id"],
            "status": teacher["status"],
            "wake_condition": teacher["wake_condition"],
            "input_state": teacher["input_state"],
        },
        "live_prefix": [
            {"id": n["id"], "status": n["status"], "schedulable_now": n["schedulable_now"]}
            for n in prefix
        ],
        "unreachable_after_teacher": [n["id"] for n in after if n["status"] == ST_UNREACHABLE],
        "n_complete": len(by_status[ST_COMPLETE]),
        "n_ready": len(by_status[ST_READY]),
        "n_sleeping": len(by_status[ST_SLEEPING]),
        "n_unreachable": len(by_status[ST_UNREACHABLE]),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_all_graphs(
    *,
    inventory: Mapping[str, Any] | None = None,
    receipt_probe: ReceiptProbe | None = None,
    teacher_state: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    inv = dict(inventory) if inventory is not None else fs.organ_inventory()
    teacher_state = teacher_state if teacher_state is not None else flash_teacher_corpus_state()
    return [
        build_school_graph(
            sid,
            inventory=inv,
            receipt_probe=receipt_probe,
            teacher_state=teacher_state,
        )
        for sid in fs.SCHOOL_CATALOG
    ]


def union_edges(graphs: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for g in graphs:
        for e in g.get("edges") or []:
            out.append(dict(e))
    return out


def cross_school_edges(graphs: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    bad: list[dict[str, str]] = []
    for e in union_edges(graphs):
        src = school_of(e["from"])
        dst = school_of(e["to"])
        declared = e.get("school")
        if src != dst or (declared and (src != declared or dst != declared)):
            bad.append(e)
    return bad


def all_nodes(graphs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in graphs:
        for n in g.get("nodes") or []:
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# Ready-prefix WorkGraph (CPU_ANALYSIS). Sleeping organ nodes stay here.
# ---------------------------------------------------------------------------


def emit_ready_units(graphs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Admit READY organ nodes into a workgraph.WorkGraph as CPU_ANALYSIS.

    Organ-funnel SLEEPING is not hardware SLEEPING. Those nodes are not
    converted; converting them would either lie (pending) or trip
    workgraph.make_unit's hardware-only sleeping guard.
    """
    g = wg.WorkGraph(ncpu=max(8, len(fs.SCHOOL_CATALOG)))
    admitted: list[dict[str, Any]] = []
    for school_graph in graphs:
        if not school_graph.get("graph_emitted"):
            continue
        for node in school_graph.get("nodes") or []:
            if node["status"] != ST_READY:
                continue
            deps = [
                d
                for d in node["dependencies"]
                if any(
                    n["id"] == d and n["status"] == ST_READY
                    for n in (school_graph.get("nodes") or [])
                )
            ]
            unit = wg.make_unit(
                id=node["id"],
                role="flash_organ_funnel_stage",
                description=(
                    f"Funnel stage {node['stage']} for Gravity school {node['school']} "
                    f"({node['title']}). STATIC_ONLY. Not teacher fit."
                ),
                dependencies=deps,
                resource_lane="CPU_ANALYSIS",
                mutation_scope=[f"school:{node['school']}"],
                verifier=node["verifier"],
                expected_information_gain=wg.INFO_MEDIUM,
                cost_units=1,
                requires_hardware=False,
                species="learned_compiler_experiment",
                effect_class="READ_ONLY",
                extras={
                    "school": node["school"],
                    "stage": node["stage"],
                    "organ_funnel_status": node["status"],
                    "not_hardware_sleep": True,
                },
            )
            outcome = g.admit(unit)
            if outcome["kind"] == "inserted":
                admitted.append(unit)
    ready = g.compute_ready(mutate=False)
    return {
        "admitted_ids": [u["id"] for u in admitted],
        "n_admitted": len(admitted),
        "ready_ids": [u["id"] for u in ready],
        "n_ready": len(ready),
        "cross_school_ready_deps": [
            u["id"]
            for u in admitted
            if any(school_of(d) != school_of(u["id"]) for d in u["dependencies"])
        ],
        "sleeping_organ_nodes_converted": 0,
        "note": (
            "Only READY organ-funnel nodes are admitted. SLEEPING teacher-fit "
            "nodes stay in the organ graph; workgraph.make_unit reserves SLEEPING "
            "for unqualified hardware."
        ),
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "workgraph_executes": False,
    }


def emit_school_workunit(school: str) -> dict[str, Any]:
    if school not in fs.SCHOOL_CATALOG:
        raise fs.UnknownSchoolError(school)
    row = wus.emit_hcli_workunit(
        id=f"future.flash_organ_workgraphs.{school}",
        role="flash_organ_funnel_graph",
        description=(
            f"Independently evaluate the 11-stage funnel WorkGraph for Gravity "
            f"school {school}. Prefix (analytic, tiny synthetic) is CPU_ANALYSIS. "
            "Gate 2 sleeps until a Flash teacher corpus exists."
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier=f"future.flash_organ_workgraphs.{school}.no_cross_school_edges",
        provider="tools.future.flash_organ_workgraphs",
        preferred_backend=None,
        status="pending",
        classification="STATIC_ONLY",
        extras={
            "command": [
                "python3",
                "tools/future/flash_organ_workgraphs.py",
                "--school",
                school,
            ],
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "species": "learned_compiler_experiment",
            "school": school,
            "requires_quiescence": False,
            "blocked_reason": None,
        },
    )
    wus.validate_emitted_unit(row)
    return row


# ---------------------------------------------------------------------------
# Negative controls that must fire during build, not only in pytest.
# ---------------------------------------------------------------------------


def prove_negative_controls(
    graphs: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    # 1. Absent input receipt → SLEEP, never run.
    slept = False
    wake = None
    teacher_nodes = [
        n
        for g in graphs
        for n in (g.get("nodes") or [])
        if n["stage"] == TEACHER_FIT_STAGE
    ]
    ran_while_asleep = False
    if teacher_nodes:
        sample = teacher_nodes[0]
        slept = sample["status"] == ST_SLEEPING
        wake = sample.get("wake_condition")
        try:
            run_node(sample, inventory=inventory)
            ran_while_asleep = True
        except NodeAsleep:
            ran_while_asleep = False
    absent_input_sleeps = {
        "fired": slept and wake and not ran_while_asleep,
        "n_teacher_fit_sleeping": sum(1 for n in teacher_nodes if n["status"] == ST_SLEEPING),
        "n_teacher_fit_total": len(teacher_nodes),
        "wake_condition": wake,
        "run_while_asleep": ran_while_asleep,
    }
    if not absent_input_sleeps["fired"]:
        raise RuntimeError("negative control did not fire: teacher_fit did not SLEEP/refuse run")

    # 2. No cross-school edges.
    cross = cross_school_edges(graphs)
    no_cross = {
        "fired": cross == [],
        "n_edges": len(union_edges(graphs)),
        "n_cross_school": len(cross),
        "cross_school": cross[:8],
    }
    if not no_cross["fired"]:
        raise RuntimeError(f"negative control did not fire: cross-school edges {cross[:4]}")

    # 3. Empty census family is reported, not a complete-looking empty graph.
    empty_inv = {
        "census_source": "pinned_snapshot",
        "by_family": {},
        "budget_by_family": {},
        "families": [],
        "router_tensor_bytes": None,
    }
    empty_graph = build_school_graph("ROUTED_EXPERTS", inventory=empty_inv)
    empty_census = {
        "fired": (
            empty_graph["empty_census_family"] is True
            and empty_graph["graph_emitted"] is False
            and empty_graph["looks_complete"] is False
            and empty_graph["nodes"] is None
        ),
        "kind": (empty_graph.get("census") or {}).get("kind"),
        "reason": empty_graph.get("reason"),
        "nodes_is_none": empty_graph.get("nodes") is None,
        "looks_complete": empty_graph.get("looks_complete"),
    }
    if not empty_census["fired"]:
        raise RuntimeError("negative control did not fire: empty census looked complete")

    # 4. COMPLETE without a receipt on disk is refused.
    raised = False
    reason = None
    ready = next(
        (n for g in graphs for n in (g.get("nodes") or []) if n["status"] == ST_READY),
        None,
    )
    if ready is None:
        raise RuntimeError("negative control could not find a READY node to refuse COMPLETE on")
    try:
        mark_complete(ready, REPO / "receipts" / "future" / "DOES_NOT_EXIST_ORGAN_WG.json")
    except CompleteWithoutReceipt as exc:
        raised = True
        reason = str(exc)
    complete_refused = {
        "fired": raised,
        "node_id": ready["id"],
        "reason": reason,
    }
    if not complete_refused["fired"]:
        raise RuntimeError("negative control did not fire: COMPLETE succeeded without a receipt")

    # Sleeping node cannot be marked COMPLETE even with a real file (this module).
    sleep_raised = False
    try:
        mark_complete(teacher_nodes[0], Path(__file__))
    except CompleteWithoutReceipt:
        sleep_raised = True
    if not sleep_raised:
        raise RuntimeError("negative control did not fire: SLEEPING node accepted COMPLETE")

    every_teacher_sleeps = bool(teacher_nodes) and all(
        n["status"] == ST_SLEEPING for n in teacher_nodes
    )
    after_unreachable = all(
        n["status"] == ST_UNREACHABLE
        for g in graphs
        for n in (g.get("nodes") or [])
        if n["index"] > 3
    )
    if not every_teacher_sleeps:
        raise RuntimeError("not every school's real_teacher_fit is SLEEPING")
    if not after_unreachable:
        raise RuntimeError("a node past teacher_fit was not UNREACHABLE")

    return {
        "absent_input_sleeps": absent_input_sleeps,
        "no_cross_school_edges": no_cross,
        "empty_census_family_reported": empty_census,
        "complete_without_receipt_refused": complete_refused,
        "sleeping_cannot_be_marked_complete": {"fired": sleep_raised},
        "every_teacher_fit_sleeps": every_teacher_sleeps,
        "every_node_past_teacher_unreachable": after_unreachable,
        "all_fired": True,
    }


# ---------------------------------------------------------------------------
# Recovery / gaps / resident callability
# ---------------------------------------------------------------------------


def recovered_implementation() -> dict[str, Any]:
    return {
        "note": (
            "flash_schools already schedules one school alone. meta_funnel already "
            "refuses to skip a gate. workgraph already schedules units, but its "
            "SLEEPING is hardware-only. This module extends all three; it does not "
            "fork the school catalog, the nine gates, or the fourteen lanes."
        ),
        "landed_siblings_extended": [
            {
                "path": "tools/future/flash_schools.py",
                "used": [
                    "SCHOOL_CATALOG",
                    "SCHOOL_ORGAN_SLUG",
                    "SCHOOL_CENSUS_FAMILIES",
                    "schedule_school",
                    "admit_candidate",
                    "organ_inventory",
                    "prove_flop_trap",
                ],
                "adequate_as_funnel_workgraph": False,
                "gap": "schedules a school; does not wire the 11-stage funnel as a graph",
            },
            {
                "path": "tools/future/meta_funnel.py",
                "used": [
                    "GATES (9)",
                    "_eval_analytical",
                    "_eval_from_status",
                    "_uniform_plan",
                    "ABSENT_TOKENS / input_state (consumed as law)",
                ],
                "adequate_as_funnel_workgraph": False,
                "gap": (
                    "nine gates, not eleven: missing tiny_synthetic_sanity and "
                    "state_stability. advance() is per-candidate, not per-school DAG"
                ),
            },
            {
                "path": "tools/future/workgraph.py",
                "used": ["WorkGraph", "make_unit", "graph_identity (via admit)", "CPU_ANALYSIS lane"],
                "adequate_as_funnel_workgraph": False,
                "gap": (
                    "SLEEPING reserved for hardware-gated work "
                    "(make_unit rejects non-hardware sleeping). Organ-funnel sleep "
                    "is missing-receipt sleep and lives here."
                ),
            },
            {
                "path": "tools/future/teacher_corpus.py",
                "used": ["validate_corpus (named as Gate 2 verifier)", "RECEIPT (contract, not a corpus)"],
                "adequate_as_funnel_workgraph": False,
                "gap": "validator + fixtures; no Flash teacher-forced corpus on disk",
            },
            {
                "path": "tools/future/expert_bank_school.py",
                "used": "ROUTED_EXPERTS candidates via flash_schools._wrap_expert_bank",
                "adequate_as_funnel_workgraph": False,
            },
            {
                "path": "tools/future/ngram_school.py",
                "used": "NGRAM candidates via flash_schools._wrap_ngram",
                "adequate_as_funnel_workgraph": False,
            },
            {
                "path": "tools/future/moe_physical_school.py",
                "used": "cited by SHARED_EXPERTS FusedPhysicalProgram; not forked",
                "adequate_as_funnel_workgraph": False,
            },
            {
                "path": "tools/future/router_science.py",
                "used": "cited by ROUTER school + route_stability verifier name",
                "adequate_as_funnel_workgraph": False,
            },
        ],
        "not_duplicating": [
            "SCHOOL_CATALOG / SCHOOL_ORGAN_SLUG / SCHOOL_CENSUS_FAMILIES",
            "meta_funnel.GATES 1-9 (mapped, not copied as a second kill table)",
            "workgraph fourteen resource lanes",
            "teacher_corpus anti-fabrication guards",
        ],
    }


def gaps_closed() -> list[str]:
    return [
        "one 11-node WorkGraph per Gravity school over the real funnel, independently schedulable",
        "tiny_synthetic_sanity and state_stability nodes that meta_funnel's nine gates do not have",
        "Gate 2 (real_teacher_fit) SLEEPS with a wake condition on every school until a Flash teacher corpus exists on disk",
        "nodes past a sleeper are UNREACHABLE, not pending",
        "no cross-school edges (asserted over the union of all fourteen graphs)",
        "empty census family on a weight organ is reported and no complete-looking empty graph is emitted",
        "COMPLETE refused without a receipt on disk, including on a SLEEPING node given a real file",
        "READY prefix admitted to workgraph.WorkGraph as CPU_ANALYSIS; organ-sleep is not converted into hardware-sleep",
    ]


def negative_findings(graphs: Sequence[Mapping[str, Any]], teacher_state: Mapping[str, Any]) -> list[str]:
    n_emitted = sum(1 for g in graphs if g.get("graph_emitted"))
    n_complete = sum(g.get("n_complete") or 0 for g in graphs)
    return [
        (
            f"Flash teacher corpus is {teacher_state.get('state')}: "
            f"{teacher_state.get('reason')}"
        ),
        f"{n_emitted} school graphs emitted; {n_complete} COMPLETE nodes (on-disk receipt required).",
        "Did not fit any candidate to the 350GB Flash specimen.",
        "Did not take a hardware measurement. evidence_class STATIC_ONLY, gpu_authority false.",
        "Did not mark any node past real_teacher_fit COMPLETE or READY; they are UNREACHABLE.",
        "orchestration.py BINDINGS does not name this module; this lane cannot write that file.",
        "workgraph.make_unit still refuses non-hardware SLEEPING; organ-funnel sleep is this graph's status, not a fork of the fourteen lanes.",
        "FLASH_NX / FLASH_NR / FLASH_EBPW receipts, even when present in HEAD, cannot complete later stages while Gate 2 sleeps.",
        "receipts/future may be unmaterialized in this sparse checkout; in_head is not COMPLETE authority.",
    ]


def resident_callable_block(school_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "entry_point": "tools.future.flash_organ_workgraphs.build_all_graphs()",
        "workunit": (
            "one CPU_ANALYSIS unit per school; fourteen independent 11-node funnel "
            "graphs; Gate 2 sleeps; later stages unreachable"
        ),
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier": "FT.MODEL_REPRESENTATION.meta-gates-3-9",
        "fails_closed": (
            "absent input receipt → SLEEP with wake_condition, never run; "
            "COMPLETE without an on-disk receipt raises CompleteWithoutReceipt; "
            "empty census family → graph_emitted false, nodes is None, looks_complete false; "
            "nodes past a sleeper are UNREACHABLE, not pending"
        ),
        "discoverable": True,
        "module": "tools.future.flash_organ_workgraphs",
        "callables": [
            "build",
            "build_all_graphs",
            "build_school_graph",
            "run_node",
            "mark_complete",
            "cross_school_edges",
        ],
        "workunit_id_pattern": "future.flash_organ_workgraphs.<SCHOOL_ID>",
        "workunits_emitted": [f"future.flash_organ_workgraphs.{s}" for s in school_ids],
        "hcli_can_invoke": True,
        "cannot": [
            "acquire a GPU lease",
            "wake Gate 2 without a Flash teacher corpus on disk",
            "mark COMPLETE without a receipt on disk",
            "treat UNREACHABLE as pending",
            "block one school with another school's edge",
        ],
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(school_id: str | None = None) -> Path:
    inventory = fs.organ_inventory()
    teacher_state = flash_teacher_corpus_state()
    wanted = [school_id] if school_id else list(fs.SCHOOL_CATALOG)
    if school_id and school_id not in fs.SCHOOL_CATALOG:
        raise fs.UnknownSchoolError(school_id)
    graphs = [
        build_school_graph(sid, inventory=inventory, teacher_state=teacher_state)
        for sid in wanted
    ]
    # Proofs always run over the full catalog so a --school build cannot
    # silently drop the independence / empty-census guards.
    all_graphs = (
        graphs
        if school_id is None
        else build_all_graphs(inventory=inventory, teacher_state=teacher_state)
    )
    proofs = prove_negative_controls(all_graphs, inventory)
    ready_wg = emit_ready_units(all_graphs)
    workunits = [emit_school_workunit(g["school"]) for g in graphs if g.get("graph_emitted")]

    n_nodes = sum(g.get("n_nodes") or 0 for g in all_graphs)
    n_edges = len(union_edges(all_graphs))
    n_sleep = sum(g.get("n_sleeping") or 0 for g in all_graphs)
    n_unreach = sum(g.get("n_unreachable") or 0 for g in all_graphs)
    n_ready = sum(g.get("n_ready") or 0 for g in all_graphs)
    n_complete = sum(g.get("n_complete") or 0 for g in all_graphs)
    n_emitted = sum(1 for g in all_graphs if g.get("graph_emitted"))

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "One WorkGraph per Flash Gravity organ school over the real funnel, "
            "independently schedulable, honestly asleep at Gate 2 until a Flash "
            "teacher corpus exists, and UNREACHABLE after that sleeper."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "funnel_stages": [s.as_dict() for s in FUNNEL_STAGES],
        "n_funnel_stages": len(FUNNEL_STAGES),
        "school_catalog": list(fs.SCHOOL_CATALOG),
        "function_organs": sorted(FUNCTION_ORGANS),
        "teacher_corpus": teacher_state,
        "graphs": graphs if school_id else all_graphs,
        "counts": {
            "schools_in_catalog": len(fs.SCHOOL_CATALOG),
            "graphs_emitted": n_emitted,
            "graphs_empty_census": sum(1 for g in all_graphs if g.get("empty_census_family")),
            "nodes": n_nodes,
            "edges": n_edges,
            "complete": n_complete,
            "ready": n_ready,
            "sleeping": n_sleep,
            "unreachable": n_unreach,
            "cross_school_edges": 0,
            "funnel_stages": len(FUNNEL_STAGES),
        },
        "independence": {
            "cross_school_edges": cross_school_edges(all_graphs),
            "schools_block_each_other": False,
            "rule": "no edge may leave its school; asserted over the union graph",
        },
        "gate2": {
            "stage": TEACHER_FIT_STAGE,
            "meta_funnel_gate": 2,
            "status_on_every_emitted_school": ST_SLEEPING,
            "wake_condition": STAGES_BY_NAME[TEACHER_FIT_STAGE].wake_condition,
            "nodes_past_it": ST_UNREACHABLE,
        },
        "ready_prefix_workgraph": ready_wg,
        "workunits": workunits,
        "negative_controls": proofs,
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(all_graphs, teacher_state),
        "resident_callable": resident_callable_block([g["school"] for g in all_graphs]),
        "era_vocabulary": {
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_is_not_its_own_civilization": True,
        },
        "correct_answer": (
            f"{n_emitted} of {len(fs.SCHOOL_CATALOG)} schools emitted an 11-node graph. "
            f"Gate 2 (real_teacher_fit) is SLEEPING on every emitted graph because the "
            f"Flash teacher corpus is {teacher_state.get('state')}. "
            f"{n_unreach} nodes are UNREACHABLE (not pending) past that sleeper. "
            f"{n_ready} nodes are READY in the live prefix (analytic structure, "
            f"tiny synthetic sanity). {n_complete} nodes are COMPLETE. "
            "That is the correct answer."
        ),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--school", metavar="SCHOOL_ID")
    a = ap.parse_args()
    if a.school and not a.build and not a.selftest:
        g = build_school_graph(a.school)
        print(
            json.dumps(
                {
                    "school": g["school"],
                    "graph_emitted": g["graph_emitted"],
                    "empty_census_family": g["empty_census_family"],
                    "n_nodes": g.get("n_nodes"),
                    "teacher_fit": g.get("teacher_fit"),
                    "live_prefix": g.get("live_prefix"),
                    "n_unreachable": g.get("n_unreachable"),
                    "evidence_class": g["evidence_class"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    out = build(school_id=a.school if a.build and a.school else None)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
