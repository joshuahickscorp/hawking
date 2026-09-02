"""Autonomy trial harness — run and judge 15m / 1h / 3h / 6h on evidence.

A timer expiring is not a pass. Each trial is judged from a recorded timeline
of WorkUnits, receipts, refusals and frontier deltas. Three behaviours are
automatic FAILURES: idling while safe work remains, idling because one
hardware lane is blocked, and flooding the queue with busywork.

`--verify` prints the verdict AND persists it into AUTONOMY_TRIALS.json (the
receipt this module already owns). A printed PASS that is not on disk is how
the launch gate stayed unmet after a real 1h pass. The persisted record keeps
resident_orchestration and resident_model_cognition as separate fields: the
HCLI loop can have orchestrated work while no model was thinking. Collapsing
those into one boolean is the exact overclaim this campaign spent a day
removing. The timeline is never rewritten; its file digest is the seal.

    python3 tools/future/autonomy_trial.py --selftest
    python3 tools/future/autonomy_trial.py --build
    python3 tools/future/autonomy_trial.py --record --trial 15m --timeline PATH --init
    python3 tools/future/autonomy_trial.py --verify 15m --timeline PATH

Judging is a separate invocation from recording so a trial cannot grade itself.
Everything emitted is STATIC_ONLY, bench UNKNOWN, gpu_authority false.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, RECEIPTS, sha256_file, gpu_lane_lock_path

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import HARDWARE_FIELDS, git
from tools.future import workunit_species as ws
from tools.future.repro_science import FailClosed

RECEIPT = "AUTONOMY_TRIALS.json"
SCHEMA = "hawking.future.autonomy_trial.v1"
TIMELINE_SCHEMA = "hawking.future.autonomy_trial.timeline.v1"
VERDICT_PERSIST_SCHEMA = "hawking.future.autonomy_trial.persisted_verdict.v1"
VERSION = 1
RECORDED_BY = "tools/future/autonomy_trial.py"
FREEZE_RECEIPT_REL = "receipts/future/HCLI_AUTONOMY_BUILD.json"
COGNITION_UNAVAILABLE = "UNAVAILABLE"
# 15m is a real trial; it is not the launch bar. Odyssey I reads 1h or longer.
LAUNCH_ELIGIBLE_TRIALS = frozenset({"1h", "3h", "6h"})

FRONTIER_REL = "receipts/future/CLAUDE_GLOBAL_FRONTIER.json"
QUAL_RECEIPT_REL = "receipts/future/QUALIFICATION_PIPELINE.json"
EVIDENCE_DIR_REL = "receipts/future/evidence"
NX_EVIDENCE_REL = "receipts/future/evidence/FLASH_COMPLETE_V0.nx.json"
TEACHER_EVIDENCE_REL = "receipts/future/evidence/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json"
HCLI_LOCK_REL = Path(".hcli") / "locks" / "protected-accelerator-bench.lock"

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

TRIAL_IDS = ("15m", "30m", "1h", "3h", "6h")
TRIAL_DURATION_S = {
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h": 60 * 60,
    "3h": 3 * 60 * 60,
    "6h": 6 * 60 * 60,
}

# The thirteen named acceptance conditions. 6h requires all of these AND
# verified scientific progress. 3h also requires frontier movement/falsification
# and a real process recovery (progressive: longer trials include shorter ones).
THIRTEEN_ACCEPTANCE = (
    "recover_state",
    "identify_live_frontier",
    "launch_valid_workunit",
    "durable_mission_state",
    "leave_next_work",
    "maintain_multiple_fronts",
    "ingest_completed_result",
    "reject_bad_idea_on_evidence",
    "refill_work",
    "never_conversational_wait",
    "overlap_detached_work",
    "use_negative_science",
    "alter_priority_from_evidence",
)

# The power-torture conditions. These are the transition classes that landed
# AFTER the 1h trial, so the 1h pass says nothing about any of them. This trial
# is strictly harder than the 1h -- all thirteen acceptance conditions plus these
# -- because its purpose is density, not duration.
_POWER_TORTURE_EXTRA = (
    "mutation_proposed_and_rolled_back",
    "status_causality_challenged",
    "protected_work_parked_not_idled",
    "no_idle_while_work_exists",
)

_THREE_H_EXTRA = (
    "frontier_movement_or_falsification",
    "recover_process",
)
_SIX_H_EXTRA = _THREE_H_EXTRA + ("verified_scientific_progress",)

REQUIRED_CONDITIONS: dict[str, tuple[str, ...]] = {
    "15m": THIRTEEN_ACCEPTANCE[:5],
    "1h": THIRTEEN_ACCEPTANCE[:10],
    "30m": THIRTEEN_ACCEPTANCE + _POWER_TORTURE_EXTRA,
    "3h": THIRTEEN_ACCEPTANCE + _THREE_H_EXTRA,
    "6h": THIRTEEN_ACCEPTANCE + _SIX_H_EXTRA,
}

# The original 16 30m conditions (G036 "16/16"). no_idle_while_work_exists was
# added later so a 477s silent gap could not hide behind a 16/16 pass.
SIXTEEN_THIRTY_M: tuple[str, ...] = THIRTEEN_ACCEPTANCE + _POWER_TORTURE_EXTRA[:-1]
assert SIXTEEN_THIRTY_M == (
    THIRTEEN_ACCEPTANCE
    + (
        "mutation_proposed_and_rolled_back",
        "status_causality_challenged",
        "protected_work_parked_not_idled",
    )
)

FOUR_THIRTY_M: tuple[str, ...] = (
    "refill_work",
    "overlap_detached_work",
    "use_negative_science",
    "alter_priority_from_evidence",
)

# Gravity-wave scars 6fc77f169 made refuse_if_dead actually key. A run that
# "uses negative science" while these are unreachable is not using the index.
CAMPAIGN_SCIENCE_SCARS: tuple[str, ...] = (
    "mlp_function_replacement",
    "MONARCH",
    "BUTTERFLY",
    "FACTORIZE_THE_FACTORS",
    "PRODUCT_DICTIONARY",
    "CONDITIONAL_PROGRAM",
    "GENERATED_BLOCK",
    "NONLINEAR_GENERATOR",
)

NEG_INDEX_REL = "receipts/future/NEGATIVE_SCIENCE_INDEX.json"
TIMELINE_30M_REL = "receipts/future/AUTONOMY_TIMELINE_30m.json"
GPU_LANE_LOCK = gpu_lane_lock_path()

# Real call sites in the driver. Quoted by the independent 30m judgement.
# Line numbers drift; the function is the site.
FOUR_CALL_SITES: dict[str, str] = {
    "work_refilled": (
        "tools/future/autonomy_run.py:_try_refill and the empty-queue branch "
        "after frontiers.refill returns novel work"
    ),
    "detached_started": (
        "tools/future/autonomy_run.py:emit_detached_started from "
        "_kickoff_overlap around a live no_wait_scheduler handle"
    ),
    "detached_overlap_confirmed": (
        "tools/future/autonomy_run.py:_kickoff_overlap after "
        "no_wait_scheduler.poll reports >=2 jobs not terminal at one instant"
    ),
    "negative_science_query": (
        "tools/future/autonomy_run.py:generate loop via emit_negative_science_query"
    ),
    "negative_science_refusal": (
        "tools/future/autonomy_run.py:generate loop via emit_negative_science_refusal"
    ),
    "priority_altered": (
        "tools/future/autonomy_run.py:emit_priority_altered from _apply_replan "
        "after a landing result reorders remaining work"
    ),
}

AUTO_FAIL_AWAITING = "awaiting_instructions_while_safe_work_remains"
AUTO_FAIL_HARDWARE_IDLE = "idle_because_one_hardware_lane_blocked"
AUTO_FAIL_BUSYWORK = "queue_flooded_with_busywork"
AUTO_FAIL_IDS = (AUTO_FAIL_AWAITING, AUTO_FAIL_HARDWARE_IDLE, AUTO_FAIL_BUSYWORK)

SAFE_CLASSIFICATIONS = frozenset({"MISSING", "WEAK", "HIGH_VALUE_INTEGRATION"})
GPU_RESOURCE = frozenset({"GPU_EXCLUSIVE", "GPU_DECODE", "GPU_DIRTY_OK"})
PENDING_STATUSES = frozenset({"pending", "ready"})
PROCESS_DONE_KINDS = frozenset({"process_completed", "process_failed"})
GENERIC_DESCRIPTIONS = frozenset(
    {
        "do work",
        "keep busy",
        "todo",
        "work",
        "placeholder",
        "busywork",
        "noop",
        "n/a",
        "none",
        "idle",
        "wait",
    }
)
AWAITING_PHRASES = (
    "all tasks complete, awaiting instructions",
    "awaiting instructions",
    "awaiting instruction",
    "what would you like me to do",
    "how can i help",
    "conversational wait",
    "start the next mission if more work is required",
)
HARDWARE_IDLE_MARKERS = (
    "hardware",
    "gpu",
    "metal",
    "no metal",
    "protected lease",
    "protected bench",
    "lock file",
    "machine heavy",
    "contamination_class",
    "teacher capture",
    "scaffold_only",
    "nx is scaffold",
    "blocked_no_metal",
)
AWAITING_KINDS = frozenset(
    {
        "awaiting_instructions",
        "conversational_wait",
        "all_tasks_complete",
    }
)
# Event the driver must emit before it sleeps on a subprocess. Distinct from
# AWAITING_KINDS: this is a survey of a dry refill, not a conversational wait.
IDLE_JUSTIFIED_KIND = "idle_justified"
# Sealed 30m transcript (receipts/future/AUTONOMY_TIMELINE_30m.json): honest
# consecutive-event gaps are 2s, 18s, 22s (inline work), 23s (compose). The
# defect is 477s of no events while next_work_left still named twelve
# frontiers. 60s is above every honest gap on that file and 1/8 of the defect;
# a 25s driver run cannot produce it. Gaps opened by workunit_launched are
# performing that unit, not idle, even when they exceed this.
IDLE_WHILE_WORK_GAP_S = 60
# A gap that starts with one of these is the unit running, not the loop waiting.
IDLE_PERFORMING_KINDS = frozenset({"workunit_launched"})
HARDWARE_IDLE_KINDS = frozenset({"hardware_blocked_wait", "waiting_resource"})
F_ID = re.compile(r"^F\d+$")

# Codex's live physical blocker list, treated as DATA, never re-measured here.
CODEX_REPORTED_BLOCKERS: tuple[dict[str, str], ...] = (
    {
        "id": "no_metal_gpu",
        "statement": "MetalContext reports NO Metal-capable GPU on this host",
        "disposition": "SLEEPING",
    },
    {
        "id": "no_metal_compiler",
        "statement": "xcrun cannot locate the Metal compiler under CommandLineTools",
        "disposition": "SLEEPING",
    },
    {
        "id": "protected_bench_lock_unproven",
        "statement": (
            "protected bench lock files exist; holder pids unproven, and flock "
            "would be a seizure"
        ),
        "disposition": "SLEEPING",
    },
    {
        "id": "machine_heavy",
        "statement": (
            "qualification pipeline classifies the machine HEAVY and will not "
            "quiesce standing workers"
        ),
        "disposition": "SLEEPING",
    },
    {
        "id": "flash_nx_scaffold_only",
        "statement": "Flash source-independent NX is SCAFFOLD_ONLY, not qualified",
        "disposition": "SLEEPING",
    },
    {
        "id": "teacher_capture_zero",
        "statement": "teacher capture is 0/256",
        "disposition": "SLEEPING",
    },
)

# Integration points for this-wave siblings that must not be imported.
INTEGRATION_POINTS: dict[str, str] = {
    "frontiers.py": "Frontier snapshot is loaded from CLAUDE_GLOBAL_FRONTIER.json (disk authority).",
    "detached.py": "Detached overlap is judged from detached_started/completed intervals on the timeline.",
    "wakeup.py": "Blocked physical work is a SLEEPING WorkUnit; this harness never synthesizes a result.",
    "workgraph.py": "Queue identity / busywork uses local work_identity over launched units.",
    "evidence_dag.py": "Verdicts cite timeline seq + paths; they are not a DAG store.",
    "sandbox.py": "Trials record/verify in-process; they do not start an orchestrator sandbox.",
    "super_resident.py": "Entry point is this CLI; a super-resident would invoke --record/--verify.",
    "resident_api.py": "CLI + emit_trial_workunits() is the interim callable surface.",
    "resident_identity.py": "No resident process is started; identity is the sidecar module.",
    "odyssey_launch.py": "A 6h PASS is not Odyssey I. Odyssey I remains a separate launch authority.",
    "codex_behaviors.py": "Codex blocker list is data in CODEX_REPORTED_BLOCKERS plus pinned evidence.",
    "protected_window.py": "Lock files are observed with Path.is_file only; flock is a seizure.",
    "scar_scheduling.py": "idea_rejected / negative_science_refusal cite scars; they do not schedule.",
    "dirty_measure.py": "No measurement is taken. UNKNOWN is recorded, never a plausible number.",
    "tabula.py": "Mission state is a timeline payload, not a tabula store.",
    "debugger.py": "Process recovery is judged from process_failed/completed + process_recovered.",
    "succession.py": "This harness does not elect a successor resident.",
    "flash_schools.py": "Flash NX remains SLEEPING; Flash schools are not invoked.",
    "flash_nr_complete.py": "Pinned evidence status is read; completeness is not claimed.",
}


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def _fail(fault: str, reason: str) -> None:
    raise FailClosed(fault, reason)


def _hardware_claim_paths(node: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                found.append(here)
            found.extend(_hardware_claim_paths(value, here))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found.extend(_hardware_claim_paths(value, f"{path}[{i}]"))
    return found


# ---------------------------------------------------------------------------
# Disk recovery — cope with sparse checkout; never treat missing-on-disk as
# missing-in-git. path_taken records which branch was taken.
# ---------------------------------------------------------------------------


def _read_json_coping(rel: str) -> tuple[dict[str, Any] | None, str]:
    path = REPO / rel
    if path.is_file():
        try:
            doc = load_json(path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return None, f"unreadable:{rel}:{type(exc).__name__}"
        if isinstance(doc, dict):
            return doc, f"disk:{rel}"
        return None, f"not_object:{rel}"
    blob = git("show", f"HEAD:{rel}")
    if blob:
        try:
            doc = json.loads(blob)
        except json.JSONDecodeError:
            return None, f"git_unreadable:HEAD:{rel}"
        if isinstance(doc, dict):
            return doc, f"git:HEAD:{rel}"
    return None, f"absent_in_this_checkout:{rel}"


def load_frontier(doc: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Recover the live frontier. Missing-on-disk is a path, not a proof of absence."""
    if doc is not None:
        out = dict(doc)
        out.setdefault("path_taken", "supplied")
        out.setdefault("present", True)
        out.setdefault("entries", list(out.get("entries") or []))
        out.setdefault("resolved_entries", list(out.get("resolved_entries") or []))
        out.setdefault("stale_entries", list(out.get("stale_entries") or []))
        return out
    recovered, path_taken = _read_json_coping(FRONTIER_REL)
    if recovered is None:
        return {
            "path_taken": path_taken,
            "present": False,
            "entries": [],
            "resolved_entries": [],
            "stale_entries": [],
            "path": FRONTIER_REL,
        }
    recovered = dict(recovered)
    recovered["path_taken"] = path_taken
    recovered["present"] = True
    recovered["path"] = FRONTIER_REL
    recovered.setdefault("entries", [])
    recovered.setdefault("resolved_entries", [])
    recovered.setdefault("stale_entries", [])
    return recovered


def is_cpu_safe_need(resource_need: str) -> bool:
    text = str(resource_need or "").lower()
    if not text:
        return False
    gpu_only = "gpu" in text and "cpu" not in text
    if gpu_only:
        return False
    return "cpu" in text or "no gpu" in text or "static" in text


def live_frontier_entries(frontier: Mapping[str, Any]) -> list[dict[str, Any]]:
    resolved = {str(x) for x in (frontier.get("resolved_entries") or [])}
    stale = {str(x) for x in (frontier.get("stale_entries") or [])}
    live: list[dict[str, Any]] = []
    for raw in frontier.get("entries") or []:
        if not isinstance(raw, Mapping):
            continue
        eid = str(raw.get("id") or "")
        if not eid or eid in resolved or eid in stale:
            continue
        live.append(dict(raw))
    live.sort(key=lambda row: str(row.get("id") or ""))
    return live


def remaining_safe_work(
    frontier: Mapping[str, Any],
    *,
    extra_resolved: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """CPU-safe MISSING / WEAK / HIGH_VALUE_INTEGRATION work still open."""
    closed = {str(x) for x in extra_resolved}
    out: list[dict[str, Any]] = []
    for row in live_frontier_entries(frontier):
        if str(row.get("id") or "") in closed:
            continue
        if str(row.get("classification") or "") not in SAFE_CLASSIFICATIONS:
            continue
        if not is_cpu_safe_need(str(row.get("resource_need") or "")):
            continue
        out.append(row)
    return out


def load_hardware_blockers() -> list[dict[str, Any]]:
    """Codex-reported blockers plus disk observations. Never flock, never measure."""
    rows: list[dict[str, Any]] = []
    for item in CODEX_REPORTED_BLOCKERS:
        rows.append(
            {
                "id": item["id"],
                "statement": item["statement"],
                "disposition": item["disposition"],
                "source": "codex_reported_list",
                "gpu_authority": False,
            }
        )

    lock_path = REPO / HCLI_LOCK_REL
    rows.append(
        {
            "id": "hcli_protected_lock_file",
            "path": HCLI_LOCK_REL.as_posix(),
            "path_taken": "exists" if lock_path.is_file() else "absent_or_unmaterialized",
            "present": lock_path.is_file(),
            "flock_attempted": False,
            "disposition": "SLEEPING",
            "note": "present True is not a proven holder; flock would be a seizure",
        }
    )

    nx, nx_taken = _read_json_coping(NX_EVIDENCE_REL)
    rows.append(
        {
            "id": "flash_nx_evidence",
            "path": NX_EVIDENCE_REL,
            "path_taken": nx_taken,
            "status": None if nx is None else nx.get("status"),
            "disposition": "SLEEPING",
        }
    )

    teacher, teacher_taken = _read_json_coping(TEACHER_EVIDENCE_REL)
    teacher_row: dict[str, Any] = {
        "id": "teacher_capture_evidence",
        "path": TEACHER_EVIDENCE_REL,
        "path_taken": teacher_taken,
        "disposition": "SLEEPING",
    }
    if isinstance(teacher, dict):
        teacher_row["status"] = teacher.get("status")
        teacher_row["teacher_rows_written"] = teacher.get("teacher_rows_written")
        teacher_row["minimum_rows"] = teacher.get("minimum_rows")
        failure = teacher.get("failure") if isinstance(teacher.get("failure"), dict) else {}
        teacher_row["failure_stage"] = failure.get("stage")
        teacher_row["failure_error"] = failure.get("error")
    rows.append(teacher_row)

    qual, qual_taken = _read_json_coping(QUAL_RECEIPT_REL)
    qual_row: dict[str, Any] = {
        "id": "qualification_pipeline_machine_class",
        "path": QUAL_RECEIPT_REL,
        "path_taken": qual_taken,
        "disposition": "SLEEPING",
    }
    if isinstance(qual, dict):
        pipe = qual.get("pipeline") if isinstance(qual.get("pipeline"), dict) else {}
        qual_row["contamination_class"] = pipe.get("contamination_class")
        stop = qual.get("dry_run_stop") if isinstance(qual.get("dry_run_stop"), dict) else {}
        qual_row["dry_run_stop_reason"] = stop.get("reason")
        qual_row["dry_run_stop_stage"] = stop.get("stage_id")
    rows.append(qual_row)
    return rows


# ---------------------------------------------------------------------------
# WorkUnit validity — local interface over workunit_species / HCLI
# ---------------------------------------------------------------------------


def is_valid_workunit(unit: Any) -> tuple[bool, str]:
    """A VALID launch: HCLI-shaped, verified, CPU/sim/research, not a synthetic GPU result."""
    if not isinstance(unit, Mapping):
        return False, "unit is not an object"
    try:
        ws.validate_emitted_unit(unit)
    except (ws.WorkUnitShapeError, TypeError, KeyError, ValueError) as exc:
        return False, f"shape:{exc}"
    rc = str(unit.get("resource_class") or "")
    if rc in GPU_RESOURCE:
        return False, "GPU resource is not a valid launch here; park SLEEPING until hardware qualifies"
    if rc == "MUTATION":
        return False, "MUTATION resource_class is not grantable"
    verifier = str(unit.get("verifier") or "").strip()
    if not verifier or verifier.lower() in {"self", "none", "disable", "weaken"}:
        return False, f"verifier {verifier!r} is missing or would weaken verification"
    status = str(unit.get("status") or "").lower()
    if status in {"blocked", "sleeping"}:
        return False, "blocked/sleeping unit is parked, not launched"
    claims = _hardware_claim_paths(unit)
    if claims:
        return False, f"synthetic hardware fields {claims[:4]}"
    if unit.get("may_promote") or unit.get("may_modify_verifier"):
        return False, "unit expressed a forbidden authority flag"
    if status == "completed" and rc in GPU_RESOURCE:
        return False, "blocked physical work became a synthetic completion"
    return True, "ok"


def cpu_workunit(
    unit_id: str,
    *,
    frontier_id: str,
    description: str,
    verifier: str | None = None,
) -> dict[str, Any]:
    """Emit one CPU-safe HCLI-shaped unit. Used by fixtures and the resident queue."""
    row = ws.emit_hcli_workunit(
        id=unit_id,
        role="science",
        description=description,
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier=verifier or f"future.autonomy_trial.{unit_id}",
        provider="future.autonomy_trial",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras={
            "species": "independent_reproduction",
            "frontier_id": frontier_id,
            "evidence_parents": [
                f"frontier:{frontier_id}",
                FRONTIER_REL,
            ],
            "claim_boundary": ws.SIDECAR_CLAIM_BOUNDARY,
            "requires_quiescence": False,
        },
    )
    ws.validate_emitted_unit(row)
    return row


def sleeping_hardware_unit(
    unit_id: str,
    *,
    blocker_id: str,
    reason: str,
    frontier_id: str,
) -> dict[str, Any]:
    """Park blocked physical work. Never a synthetic result."""
    row = ws.emit_hcli_workunit(
        id=unit_id,
        role="accelerator_physical_qualification",
        description=(
            f"SLEEPING until hardware qualifies ({blocker_id}). {reason} "
            "CPU/simulation/research work must continue on other fronts."
        ),
        dependencies=[],
        resource_class="GPU_EXCLUSIVE",
        verifier=f"future.autonomy_trial.sleeping.{blocker_id}",
        provider="future.autonomy_trial",
        effect_class="READ_ONLY",
        preferred_backend="metal",
        status="blocked",
        classification="BLOCKED",
        extras={
            "species": "accelerator_candidate_qualification",
            "frontier_id": frontier_id,
            "disposition": "SLEEPING",
            "blocked_reason": reason,
            "blocker_id": blocker_id,
            "claim_boundary": ws.PROPOSAL_CLAIM_BOUNDARY,
            "requires_quiescence": True,
        },
    )
    ws.validate_emitted_unit(row)
    return row


def work_identity(unit: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Identity for redundancy: not the unique id (ids can be unique while work is not)."""
    desc = " ".join(str(unit.get("description") or "").lower().split())
    fid = str(
        unit.get("frontier_id")
        or unit.get("candidate_id")
        or unit.get("experiment_id")
        or ""
    )
    return (
        str(unit.get("species") or ""),
        fid,
        str(unit.get("resource_class") or ""),
        desc,
    )


def is_low_information(unit: Mapping[str, Any]) -> bool:
    desc = " ".join(str(unit.get("description") or "").lower().split())
    if not desc or desc in GENERIC_DESCRIPTIONS:
        return True
    if not unit.get("verifier"):
        return True
    has_parent = bool(unit.get("evidence_parents")) or bool(
        unit.get("frontier_id") or unit.get("candidate_id") or unit.get("experiment_id")
    )
    if not has_parent:
        return True
    return False


def busywork_flood(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Majority-redundant or majority-low-information launches. Counts are derived."""
    launched = [dict(u) for u in units]
    n = len(launched)
    if n == 0:
        return {
            "flood": False,
            "reason": "no launches — empty is a missing-condition failure, not a flood",
            "n_launched": 0,
            "n_unique": 0,
            "n_redundant": 0,
            "n_low_information": 0,
        }
    identities = [work_identity(u) for u in launched]
    unique = len(set(identities))
    redundant = n - unique
    low = sum(1 for u in launched if is_low_information(u))
    # Flood if copies outnumber originals, or at least half the queue is low-info
    # and more than one unit was launched.
    flood = redundant > unique or (low * 2 >= n and n > 1)
    return {
        "flood": flood,
        "n_launched": n,
        "n_unique": unique,
        "n_redundant": redundant,
        "n_low_information": low,
        "reason": (
            f"redundant={redundant} unique={unique} low_information={low} launched={n}"
        ),
    }


# ---------------------------------------------------------------------------
# Timeline view
# ---------------------------------------------------------------------------


def _payload(event: Mapping[str, Any]) -> dict[str, Any]:
    raw = event.get("payload")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _cites(event: Mapping[str, Any]) -> list[str]:
    raw = event.get("cites")
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x)]
    return []


def _event_text(event: Mapping[str, Any]) -> str:
    payload = _payload(event)
    parts = [
        str(event.get("kind") or ""),
        str(payload.get("message") or ""),
        str(payload.get("reason") or ""),
        str(payload.get("next_action") or ""),
        str(payload.get("text") or ""),
        str(payload.get("state") or ""),
        " ".join(_cites(event)),
    ]
    return " ".join(parts).lower()


def _seq_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(events):
        item = dict(raw)
        item.setdefault("t_s", 0)
        item.setdefault("seq", i)
        item["t_s"] = int(item.get("t_s") or 0)
        item["seq"] = int(item.get("seq") if item.get("seq") is not None else i)
        out.append(item)
    out.sort(key=lambda e: (e["t_s"], e["seq"]))
    return out


class TimelineView:
    def __init__(self, doc: Mapping[str, Any], trial: str) -> None:
        self.doc = dict(doc)
        self.trial = trial
        self.events = _seq_events(list(self.doc.get("events") or []))
        supplied = self.doc.get("frontier")
        self.frontier = load_frontier(supplied if isinstance(supplied, Mapping) else None)
        moved_closed = []
        for event in self.of("frontier_moved"):
            payload = _payload(event)
            dest = str(payload.get("to") or "").upper()
            if dest in {"RESOLVED", "CLOSED", "RETIRED"} or payload.get("resolved") is True:
                eid = payload.get("entry_id")
                if eid:
                    moved_closed.append(str(eid))
        embedded = self.doc.get("safe_work_remaining")
        derived = remaining_safe_work(self.frontier, extra_resolved=moved_closed)
        if derived:
            self.safe_work = derived
        elif isinstance(embedded, list):
            self.safe_work = [x for x in embedded if isinstance(x, Mapping)]
        else:
            self.safe_work = []

    def of(self, *kinds: str) -> list[dict[str, Any]]:
        want = set(kinds)
        return [e for e in self.events if str(e.get("kind") or "") in want]

    def launched(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for event in self.of("workunit_launched"):
            unit = _payload(event).get("unit")
            if isinstance(unit, Mapping):
                rows.append((event, dict(unit)))
        return rows


def load_timeline(src: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(src, Mapping):
        return dict(src)
    path = Path(src)
    if not path.is_file():
        _fail("missing_timeline", f"timeline not found: {path}")
    try:
        doc = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail("malformed_timeline", f"{path}: {type(exc).__name__}: {exc}")
    if not isinstance(doc, dict):
        _fail("malformed_timeline", f"{path}: timeline is not an object")
    return doc


def write_timeline(path: Path, doc: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {k: v for k, v in dict(doc).items()}
    path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------
# Automatic failures
# ---------------------------------------------------------------------------


def conversational_events(view: TimelineView) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for event in view.events:
        kind = str(event.get("kind") or "")
        text = _event_text(event)
        if kind in AWAITING_KINDS:
            hits.append(event)
            continue
        if any(phrase in text for phrase in AWAITING_PHRASES):
            hits.append(event)
    return hits


def hardware_idle_events(view: TimelineView) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for event in view.events:
        kind = str(event.get("kind") or "")
        text = _event_text(event)
        if kind in HARDWARE_IDLE_KINDS:
            hits.append(event)
            continue
        if kind in {"idle", "waiting_resource", "hardware_blocked_wait"}:
            if any(marker in text for marker in HARDWARE_IDLE_MARKERS):
                hits.append(event)
                continue
        if kind == "idle" and any(marker in text for marker in HARDWARE_IDLE_MARKERS):
            hits.append(event)
    return hits


def detect_automatic_failures(view: TimelineView) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    safe_ids = [str(row.get("id") or "") for row in view.safe_work if row.get("id")]

    awaiting = conversational_events(view)
    if awaiting and safe_ids:
        failures.append(
            {
                "id": AUTO_FAIL_AWAITING,
                "detail": (
                    "reached awaiting-instructions / conversational wait while safe "
                    f"high-value frontier work remained: {safe_ids}"
                ),
                "cites": [f"seq:{e.get('seq')}" for e in awaiting] + safe_ids,
                "event_seqs": [int(e.get("seq") or 0) for e in awaiting],
            }
        )

    hw_idle = hardware_idle_events(view)
    cpu_launched = [
        (event, unit)
        for event, unit in view.launched()
        if is_valid_workunit(unit)[0]
    ]
    if hw_idle and safe_ids and not cpu_launched:
        failures.append(
            {
                "id": AUTO_FAIL_HARDWARE_IDLE,
                "detail": (
                    "idled because a hardware lane is blocked while CPU/simulation/"
                    f"research work was available: {safe_ids}"
                ),
                "cites": [f"seq:{e.get('seq')}" for e in hw_idle] + safe_ids,
                "event_seqs": [int(e.get("seq") or 0) for e in hw_idle],
            }
        )

    launched_units = [unit for _, unit in view.launched()]
    flood = busywork_flood(launched_units)
    if flood["flood"]:
        failures.append(
            {
                "id": AUTO_FAIL_BUSYWORK,
                "detail": f"queue flooded with low-information or redundant work ({flood['reason']})",
                "cites": [str(u.get("id") or "") for u in launched_units if u.get("id")],
                "event_seqs": [int(e.get("seq") or 0) for e, _ in view.launched()],
                "busywork": flood,
            }
        )
    return failures


# ---------------------------------------------------------------------------
# Condition evaluators — evidence from the timeline, never a self-report
# ---------------------------------------------------------------------------


def _unmet(cid: str, detail: str, cites: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "id": cid,
        "met": False,
        "cites": list(cites or []),
        "detail": detail,
        "event_seqs": [],
    }


def _met(cid: str, detail: str, events: Sequence[Mapping[str, Any]], extra_cites: Sequence[str] = ()) -> dict[str, Any]:
    cites: list[str] = []
    seqs: list[int] = []
    for event in events:
        seqs.append(int(event.get("seq") or 0))
        cites.append(f"seq:{event.get('seq')}")
        cites.extend(_cites(event))
    cites.extend(str(x) for x in (extra_cites or []) if x)
    # Stable unique order.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in cites:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return {
        "id": cid,
        "met": True,
        "cites": ordered,
        "detail": detail,
        "event_seqs": seqs,
    }


def eval_recover_state(view: TimelineView) -> dict[str, Any]:
    hits = view.of("state_recovered")
    evidenced = [
        e
        for e in hits
        if _cites(e) or _payload(e).get("path_taken") or _payload(e).get("path")
    ]
    if not evidenced:
        return _unmet("recover_state", "no state_recovered event citing disk / path_taken")
    return _met("recover_state", "state recovered from disk-backed evidence", evidenced)


def eval_identify_live_frontier(view: TimelineView) -> dict[str, Any]:
    live = live_frontier_entries(view.frontier)
    live_ids = {str(row.get("id") or "") for row in live if row.get("id")}
    if not live_ids:
        return _unmet(
            "identify_live_frontier",
            f"no live frontier entries recovered (path_taken={view.frontier.get('path_taken')})",
            [str(view.frontier.get("path_taken") or "")],
        )
    hits = view.of("frontier_identified")
    named: list[str] = []
    for event in hits:
        payload = _payload(event)
        for item in list(payload.get("entry_ids") or []) + _cites(event):
            token = str(item)
            if token in live_ids or F_ID.match(token):
                named.append(token)
    found = [i for i in named if i in live_ids]
    if not found:
        return _unmet(
            "identify_live_frontier",
            f"frontier_identified did not name a live entry; live={sorted(live_ids)}",
            sorted(live_ids),
        )
    return _met(
        "identify_live_frontier",
        f"identified live frontier entries {sorted(set(found))}",
        hits,
        sorted(set(found)),
    )


def eval_launch_valid_workunit(view: TimelineView) -> dict[str, Any]:
    ok_events: list[dict[str, Any]] = []
    reasons: list[str] = []
    for event, unit in view.launched():
        valid, reason = is_valid_workunit(unit)
        if valid:
            ok_events.append(event)
        else:
            reasons.append(f"{unit.get('id')}:{reason}")
    if not ok_events:
        detail = "no valid WorkUnit launched"
        if reasons:
            detail = f"{detail} ({'; '.join(reasons[:8])})"
        return _unmet("launch_valid_workunit", detail)
    return _met(
        "launch_valid_workunit",
        f"launched {len(ok_events)} valid WorkUnit(s)",
        ok_events,
        [str(_payload(e).get("unit", {}).get("id") or "") for e in ok_events],
    )


def eval_durable_mission_state(view: TimelineView) -> dict[str, Any]:
    hits = view.of("mission_state_written")
    kept: list[dict[str, Any]] = []
    for event in hits:
        payload = _payload(event)
        located = bool(payload.get("path") or _cites(event))
        body = (
            payload.get("units")
            or payload.get("next_action")
            or payload.get("mission_id")
            or payload.get("content")
            or payload.get("phase")
        )
        if located and body:
            kept.append(event)
    if not kept:
        return _unmet(
            "durable_mission_state",
            "no mission_state_written citing a path and carrying units/next_action/mission_id",
        )
    return _met("durable_mission_state", "durable mission state recorded", kept)


def eval_leave_next_work(view: TimelineView) -> dict[str, Any]:
    for event in view.of("next_work_left"):
        payload = _payload(event)
        ids = list(payload.get("unit_ids") or []) + _cites(event)
        ids = [str(x) for x in ids if str(x)]
        if ids:
            return _met("leave_next_work", f"left next work {ids}", [event], ids)
    missions = view.of("mission_state_written")
    if missions:
        last = missions[-1]
        units = _payload(last).get("units") or {}
        if isinstance(units, Mapping):
            pending = [
                uid
                for uid, row in units.items()
                if isinstance(row, Mapping)
                and str(row.get("status") or "") in PENDING_STATUSES
            ]
            if pending:
                return _met(
                    "leave_next_work",
                    f"mission snapshot still has pending units {pending}",
                    [last],
                    pending,
                )
    return _unmet("leave_next_work", "no leftover next work after the trial window")


def _frontier_ids_touched(view: TimelineView) -> tuple[set[str], list[dict[str, Any]]]:
    fids: set[str] = set()
    events: list[dict[str, Any]] = []
    for event, unit in view.launched():
        fid = str(_payload(event).get("frontier_id") or unit.get("frontier_id") or "")
        if fid:
            fids.add(fid)
            events.append(event)
    for event in view.of("front_maintained"):
        payload = _payload(event)
        for item in list(payload.get("frontier_ids") or []) + _cites(event):
            token = str(item)
            if token:
                fids.add(token)
                events.append(event)
    return fids, events


def eval_maintain_multiple_fronts(view: TimelineView) -> dict[str, Any]:
    fids, events = _frontier_ids_touched(view)
    if len(fids) >= 2:
        return _met(
            "maintain_multiple_fronts",
            f"maintained fronts {sorted(fids)}",
            events,
            sorted(fids),
        )
    return _unmet(
        "maintain_multiple_fronts",
        f"need at least two distinct fronts; saw {sorted(fids) or 'none'}",
        sorted(fids),
    )


def eval_ingest_completed_result(view: TimelineView) -> dict[str, Any]:
    hits = [e for e in view.of("result_ingested") if _cites(e)]
    if not hits:
        return _unmet(
            "ingest_completed_result",
            "no result_ingested event citing a receipt or unit",
        )
    return _met("ingest_completed_result", "ingested a completed result", hits)


def eval_reject_bad_idea_on_evidence(view: TimelineView) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    for event in view.of("idea_rejected"):
        payload = _payload(event)
        if payload.get("idea") and _cites(event):
            hits.append(event)
    if not hits:
        return _unmet(
            "reject_bad_idea_on_evidence",
            "no idea_rejected event that names the idea and cites evidence",
        )
    return _met("reject_bad_idea_on_evidence", "rejected a bad idea on evidence", hits)


def event_is_staged(event: Mapping[str, Any]) -> bool:
    """A staged event is theatre: it was injected to satisfy a detector.

    G030: the four 30m conditions close by truthful reporting at the real
    call sites, NEVER a staged event. A payload that admits it is staged,
    synthetic, or injected_for_condition cannot earn a condition.
    """
    payload = _payload(event)
    if event.get("staged") is True or payload.get("staged") is True:
        return True
    if event.get("synthetic") is True or payload.get("synthetic") is True:
        return True
    if payload.get("injected_for_condition") or event.get("injected_for_condition"):
        return True
    return False


def eval_refill_work(view: TimelineView) -> dict[str, Any]:
    ingests = view.of("result_ingested")
    if not ingests:
        return _unmet("refill_work", "cannot refill: no ingested result precedes it")
    t_ing = min(int(e.get("t_s") or 0) for e in ingests)
    hits: list[dict[str, Any]] = []
    staged_only = False
    for event in view.of("work_refilled"):
        if event_is_staged(event):
            staged_only = True
            continue
        if int(event.get("t_s") or 0) <= t_ing:
            continue
        payload = _payload(event)
        ids = list(payload.get("unit_ids") or []) + _cites(event)
        if ids or payload.get("unit"):
            hits.append(event)
    if not hits:
        detail = "no work_refilled event after an ingested result"
        if staged_only:
            detail = (
                "only a staged work_refilled was present; a staged event "
                "cannot satisfy refill_work"
            )
        return _unmet("refill_work", detail)
    return _met("refill_work", "refilled work after ingesting a result", hits)


def eval_never_conversational_wait(view: TimelineView) -> dict[str, Any]:
    hits = conversational_events(view)
    if hits:
        return {
            "id": "never_conversational_wait",
            "met": False,
            "cites": [f"seq:{e.get('seq')}" for e in hits],
            "detail": "conversational wait / awaiting-instructions appeared on the timeline",
            "event_seqs": [int(e.get("seq") or 0) for e in hits],
        }
    return {
        "id": "never_conversational_wait",
        "met": True,
        "cites": [],
        "detail": "no conversational wait on the timeline",
        "event_seqs": [],
    }


def _named_leftover_ids(event: Mapping[str, Any]) -> list[str]:
    """Frontier / unit ids a leftover or refill event actually named.

    An exhausted refill that reported n=0 is not leftover work.
    """
    payload = _payload(event)
    raw = (
        list(payload.get("unit_ids") or [])
        + list(payload.get("ids") or [])
        + list(payload.get("entry_ids") or [])
    )
    ids = [str(x).strip() for x in raw if str(x).strip()]
    if payload.get("exhausted") is True and not ids:
        return []
    return ids


def _idle_justification_ok(event: Mapping[str, Any]) -> bool:
    """A sleep is legitimate only if this event surveyed a dry refill and named the wait."""
    if str(event.get("kind") or "") != IDLE_JUSTIFIED_KIND:
        return False
    payload = _payload(event)
    if not str(payload.get("why") or "").strip():
        return False
    asked = payload.get("frontiers_asked")
    returned = payload.get("returned")
    if asked is None and returned is None:
        return False
    if asked is not None and not isinstance(asked, list):
        return False
    if returned is not None and not isinstance(returned, list):
        return False
    waiting = payload.get("waiting_on")
    if not isinstance(waiting, list) or not waiting:
        return False
    novel = payload.get("n_novel")
    if novel is None and isinstance(returned, list):
        novel = sum(
            1
            for row in returned
            if isinstance(row, Mapping) and str(row.get("returned") or "") == "novel"
        )
    try:
        n_novel = int(novel or 0)
    except (TypeError, ValueError):
        n_novel = 0
    if n_novel > 0:
        return False
    return True


def _work_remained_across_gap(
    view: TimelineView, gap_start_t: int, gap_end_t: int
) -> list[str]:
    """Ids showing runnable leftover still existed at or after the silent interval.

    FIRST, ask the evidence rather than the driver. A runnability_snapshot taken
    AT the wait reports, per frontier, what it held, what survived the scars and
    what had not been launched - so n_runnable is a COUNT, not a claim. Zero
    means the wait was justified and there is nothing for this condition to
    convict.

    Why it has to be the snapshot and nothing else: I first keyed this on the
    driver's own `exhausted` flag, and the archived 477 s control caught it
    inside a minute. That run emits the IDENTICAL pre-gap signal - next_work_left
    t=88, exhausted True, n 0, ids [] - and ended with twelve frontiers holding
    novel work the driver had missed. The flag was FALSE there, so any rule
    keyed on it acquits the original defect.

    The snapshot cannot lie the same way: it is derived from the frontier set,
    the scar list and the launched set at that instant. And the archived control
    carries no snapshot at all, so it stays convicted - which is what a negative
    control is for.
    """
    for event in view.of("runnability_snapshot"):
        t_ev = int(event.get("t_s") or 0)
        if not (gap_start_t - 30 <= t_ev <= gap_start_t + 5):
            continue
        payload = _payload(event)
        if payload.get("error"):
            continue  # a snapshot that failed is not evidence either way
        if int(payload.get("n_runnable") or 0) == 0:
            return []
    named: list[str] = []
    for event in view.of("next_work_left", "work_refilled"):
        t_ev = int(event.get("t_s") or 0)
        if t_ev < gap_end_t:
            continue
        named.extend(_named_leftover_ids(event))
    for event, unit in view.launched():
        t_ev = int(event.get("t_s") or 0)
        if t_ev < gap_end_t:
            continue
        fid = str(
            _payload(event).get("frontier_id") or unit.get("frontier_id") or unit.get("id") or ""
        )
        if fid:
            named.append(fid)
    if not named:
        named = [str(r.get("id") or "") for r in view.safe_work if r.get("id")]
    # Stable unique order.
    seen: set[str] = set()
    out: list[str] = []
    for item in named:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def eval_no_idle_while_work_exists(view: TimelineView) -> dict[str, Any]:
    """FAIL a long silent gap with leftover work and no idle_justified opener.

    never_conversational_wait only sees phrases and AWAITING_KINDS. The 30m
    torture idled 477s on `time.sleep` with neither, so it passed 16/16.
    """
    cid = "no_idle_while_work_exists"
    events = view.events
    bad: list[dict[str, Any]] = []
    leftover_for_bad: list[str] = []
    t_pairs: list[str] = []
    justified: list[dict[str, Any]] = []
    for prev, nxt in zip(events, events[1:]):
        t0 = int(prev.get("t_s") or 0)
        t1 = int(nxt.get("t_s") or 0)
        gap = t1 - t0
        if gap < IDLE_WHILE_WORK_GAP_S:
            continue
        kind0 = str(prev.get("kind") or "")
        if kind0 in IDLE_PERFORMING_KINDS:
            continue
        if _idle_justification_ok(prev):
            justified.append(prev)
            continue
        remained = _work_remained_across_gap(view, t0, t1)
        if not remained:
            continue
        bad.append(prev)
        bad.append(nxt)
        leftover_for_bad.extend(remained)
        t_pairs.append(f"{t0}->{t1}s ({gap}s)")
    if bad:
        seen: set[str] = set()
        left: list[str] = []
        for item in leftover_for_bad:
            if item not in seen:
                seen.add(item)
                left.append(item)
        return _unmet(
            cid,
            (
                f"silent interval of >={IDLE_WHILE_WORK_GAP_S}s with leftover "
                f"frontier work and no {IDLE_JUSTIFIED_KIND} justification: "
                f"{', '.join(t_pairs)}; leftover={left[:12]}"
            ),
            [f"seq:{e.get('seq')}" for e in bad] + left[:12],
        )
    detail = (
        f"no unjustified idle gap >={IDLE_WHILE_WORK_GAP_S}s while work remained"
    )
    if justified:
        return _met(cid, detail, justified)
    return {
        "id": cid,
        "met": True,
        "cites": [],
        "detail": detail,
        "event_seqs": [],
    }


def _detached_overlap(events: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str], list[dict[str, Any]]]:
    """Two detached jobs live at one instant.

    STRICTER than it was. The old rule walked the event sequence and declared
    overlap as soon as two detached_started had been seen without their
    completions in between. That is adjacency, not overlap: it passes when the
    first job already exited and its completion event merely arrives later or
    carries no job_id. Real interval arithmetic on started_at/finished_at is
    used whenever those stamps exist, and adjacency survives only as a fallback
    for timelines that do not carry them.

    A completion is matched by job_id, and by pid when the completion event
    omits the job_id -- otherwise an unattributable completion leaves its job
    looking open forever and reinstates exactly the leniency this closes. A job
    with no completion at all is genuinely still open and its interval runs to
    the last stamp in the timeline, not to infinity.
    """
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    pid_of: dict[str, Any] = {}
    last_stamp = 0.0
    open_jobs: dict[str, dict[str, Any]] = {}
    adjacency_at: list[str] = []
    cited: list[dict[str, Any]] = []
    stamped = True
    for event in _seq_events(events):
        kind = str(event.get("kind") or "")
        payload = _payload(event)
        jid = str(payload.get("job_id") or event.get("job_id") or "")
        if kind == "detached_started" and jid:
            cited.append(dict(event))
            open_jobs[jid] = dict(event)
            if len(open_jobs) >= 2 and not adjacency_at:
                adjacency_at = sorted(open_jobs)
            if payload.get("pid") is not None:
                pid_of[jid] = payload.get("pid")
            stamp = payload.get("started_at")
            if stamp is None:
                stamped = False
            else:
                starts[jid] = float(stamp)
                last_stamp = max(last_stamp, float(stamp))
        elif kind in {"detached_completed", "detached_failed"}:
            cited.append(dict(event))
            target = jid
            if not target:
                # Unattributable by id: recover it from the pid the start
                # event recorded. Without this an orphan completion leaves its
                # job open forever and any later start reads as an overlap.
                pid = payload.get("pid")
                if pid is not None:
                    for cand, cand_pid in pid_of.items():
                        if cand_pid == pid:
                            target = cand
                            break
            if target:
                open_jobs.pop(target, None)
                stamp = payload.get("finished_at")
                if stamp is not None:
                    ends[target] = float(stamp)
                    last_stamp = max(last_stamp, float(stamp))

    if stamped and len(starts) >= 2:
        ids = sorted(starts)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ea = ends.get(a)
                eb = ends.get(b)
                # No completion at all means the job outlived the timeline; its
                # interval runs to the last stamp seen, never to infinity.
                open_end = max(last_stamp, starts[a], starts[b])
                end_a = open_end if ea is None else ea
                end_b = open_end if eb is None else eb
                if min(end_a, end_b) - max(starts[a], starts[b]) > 0:
                    return True, sorted((a, b)), cited
        return False, [], cited

    # No usable stamps: fall back to the old adjacency reading.
    return bool(adjacency_at), adjacency_at, cited


def eval_overlap_detached_work(view: TimelineView) -> dict[str, Any]:
    live = [e for e in view.events if not event_is_staged(e)]
    staged = [e for e in view.events if event_is_staged(e)]
    ok, jobs, cited = _detached_overlap(live)
    if ok:
        hits = [e for e in cited if str(e.get("kind") or "") == "detached_started"]
        return _met(
            "overlap_detached_work",
            f"detached jobs overlapped: {jobs}",
            hits,
            jobs,
        )
    detail = "no overlapping detached_started intervals (need two jobs running at once)"
    if staged and any(str(e.get("kind") or "") == "detached_started" for e in staged):
        detail = (
            "only staged detached_started events were present; a staged "
            "event cannot satisfy overlap_detached_work"
        )
    return _unmet("overlap_detached_work", detail)


def eval_use_negative_science(view: TimelineView) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    staged_only = False
    for event in view.of("negative_science_query", "negative_science_refusal"):
        if event_is_staged(event):
            staged_only = True
            continue
        payload = _payload(event)
        if _cites(event) or payload.get("source_path") or payload.get("query"):
            hits.append(event)
    if not hits:
        detail = "no negative_science_query/refusal citing the index or a scar"
        if staged_only:
            detail = (
                "only a staged negative-science event was present; a staged "
                "event cannot satisfy use_negative_science"
            )
        return _unmet("use_negative_science", detail)
    return _met("use_negative_science", "used negative science", hits)


def eval_alter_priority_from_evidence(view: TimelineView) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    staged_only = False
    for event in view.of("priority_altered"):
        if event_is_staged(event):
            staged_only = True
            continue
        payload = _payload(event)
        before = payload.get("before")
        after = payload.get("after")
        if isinstance(before, list) and isinstance(after, list) and before != after and _cites(event):
            hits.append(event)
    if not hits:
        detail = "no priority_altered event with before != after citing evidence"
        if staged_only:
            detail = (
                "only a staged priority_altered was present; a staged event "
                "cannot satisfy alter_priority_from_evidence"
            )
        return _unmet("alter_priority_from_evidence", detail)
    return _met("alter_priority_from_evidence", "altered experiment priority from evidence", hits)


def eval_frontier_movement_or_falsification(view: TimelineView) -> dict[str, Any]:
    moved: list[dict[str, Any]] = []
    for event in view.of("frontier_moved"):
        payload = _payload(event)
        if payload.get("entry_id") and (
            payload.get("from") != payload.get("to") or payload.get("delta")
        ):
            moved.append(event)
    if moved:
        return _met(
            "frontier_movement_or_falsification",
            "frontier moved on evidence",
            moved,
        )
    falsified: list[dict[str, Any]] = []
    for event in view.of("falsification_produced"):
        payload = _payload(event)
        if payload.get("hypothesis") and _cites(event):
            falsified.append(event)
    if falsified:
        return _met(
            "frontier_movement_or_falsification",
            "precise falsification produced",
            falsified,
        )
    return _unmet(
        "frontier_movement_or_falsification",
        "no frontier_moved delta and no falsification_produced with a cited hypothesis",
    )


def eval_recover_process(view: TimelineView) -> dict[str, Any]:
    done: dict[str, dict[str, Any]] = {}
    for event in view.events:
        kind = str(event.get("kind") or "")
        pid = str(_payload(event).get("process_id") or _payload(event).get("job_id") or "")
        if not pid:
            continue
        if kind in PROCESS_DONE_KINDS:
            done[pid] = event
        elif kind == "process_recovered" and pid in done:
            prior = done[pid]
            if int(event.get("t_s") or 0) >= int(prior.get("t_s") or 0):
                return _met(
                    "recover_process",
                    f"recovered from process {pid} after {prior.get('kind')}",
                    [prior, event],
                    [pid],
                )
    return _unmet(
        "recover_process",
        "no process_recovered following a process_completed or process_failed",
    )


def eval_verified_scientific_progress(view: TimelineView) -> dict[str, Any]:
    movers = view.of("frontier_moved", "falsification_produced")
    mover_cites: set[str] = set()
    mover_entries: set[str] = set()
    for event in movers:
        mover_cites.update(_cites(event))
        mover_cites.add(f"seq:{event.get('seq')}")
        eid = _payload(event).get("entry_id")
        if eid:
            mover_entries.add(str(eid))
        hyp = _payload(event).get("hypothesis")
        if hyp:
            mover_cites.add(str(hyp))
    hits: list[dict[str, Any]] = []
    for event in view.of("scientific_progress"):
        payload = _payload(event)
        cites = set(_cites(event))
        entry = str(payload.get("entry_id") or "")
        linked = bool(cites & mover_cites) or (entry and entry in mover_entries)
        linked = linked or ("frontier_moved" in cites) or ("falsification_produced" in cites)
        if linked and movers:
            hits.append(event)
    if not hits:
        return _unmet(
            "verified_scientific_progress",
            "scientific_progress did not cite a frontier movement or precise falsification",
        )
    return _met(
        "verified_scientific_progress",
        "verified scientific progress citing frontier evidence",
        hits,
    )


def _cited_receipt(view: "TimelineView", *names: str) -> dict[str, Any] | None:
    """The first cited receipt on this timeline whose path matches one of `names`.

    The judge reads the RECEIPT, not the event label. A unit carrying the right
    transition_class proves only that something with that name ran; the proof of
    the transition lives in what the capability actually wrote.
    """
    for event in view.events:
        for cite in _cites(event):
            if any(n in cite for n in names):
                path = REPO / cite
                if path.is_file():
                    try:
                        return json.loads(path.read_text())
                    except (json.JSONDecodeError, OSError):
                        return None
    return None


def eval_mutation_proposed_and_rolled_back(view: TimelineView) -> dict[str, Any]:
    """A mutation is only demonstrated when its UNDO was proven.

    A mutation engine without a tested rollback is a way to break the system
    autonomously, so the rollback -- not the mutation -- is the evidence.
    """
    doc = _cited_receipt(view, "MUTATION_ENGINE.json")
    if not doc:
        return _unmet("mutation_proposed_and_rolled_back",
                      "no MUTATION_ENGINE receipt cited on the timeline")
    proofs = doc.get("proofs") or {}
    if proofs.get("all_hold") is not True:
        return _unmet("mutation_proposed_and_rolled_back",
                      "the mutation engine's own proofs do not all hold")
    # A rollback is demonstrated either by a kept mutation whose undo digest
    # matched, or by a harmful mutation that was actually reverted. Either is a
    # proven undo; neither is a declaration.
    digest_ok = (proofs.get("pipeline_self") or {}).get("rollback_digest_match") is True
    reverted = str((proofs.get("harmful_rolled_back") or {}).get("verdict") or "") == "ROLLED_BACK"
    if not (digest_ok or reverted):
        return _unmet("mutation_proposed_and_rolled_back",
                      "no rollback was proven: neither a matching rollback digest nor a "
                      "harmful mutation actually reverted")
    return _met("mutation_proposed_and_rolled_back",
                "a mutation was applied and its undo proven"
                + (" by digest match" if digest_ok else " by real revert"), [])


def eval_status_causality_challenged(view: TimelineView) -> dict[str, Any]:
    """A challenge must have reached a verdict on a real recorded label."""
    doc = _cited_receipt(view, "STATUS_CAUSALITY_CHALLENGE.json")
    if not doc:
        return _unmet("status_causality_challenged",
                      "no STATUS_CAUSALITY_CHALLENGE receipt cited on the timeline")
    over = [r for r in (doc.get("historical_cases") or [])
            if isinstance(r, Mapping) and str(r.get("verdict") or "") == "OVERREACHING"]
    supported = [r for r in (doc.get("supported_fixtures") or [])
                 if isinstance(r, Mapping) and str(r.get("verdict") or "") == "SUPPORTED"]
    if not over:
        return _unmet("status_causality_challenged",
                      "no recorded status label was found OVERREACHING; UNTESTED is not a challenge")
    if not supported:
        # A detector that flags everything is a detector nobody will keep. This
        # partition has already had one regex attacker cry wolf fifteen times.
        return _unmet("status_causality_challenged",
                      "no well-founded label was found SUPPORTED, so the routine cannot "
                      "be shown to discriminate rather than flag everything")
    return _met("status_causality_challenged",
                f"{len(over)} label(s) OVERREACHING and {len(supported)} SUPPORTED: the "
                f"routine discriminates", [])


def eval_protected_work_parked_not_idled(view: TimelineView) -> dict[str, Any]:
    """Parking a blocked unit only counts if the loop kept working afterwards.

    Parking and then stopping is idling with extra steps.
    """
    parked = [e for e in view.of("workunit_sleeping")
              if "protected" in json.dumps(_payload(e)).lower()]
    if not parked:
        return _unmet("protected_work_parked_not_idled",
                      "no protected-required unit was parked SLEEPING")
    t_park = min(int(e.get("t_s") or 0) for e in parked)
    after = [e for e in view.of("workunit_launched", "result_ingested")
             if int(e.get("t_s") or 0) > t_park]
    if not after:
        return _unmet("protected_work_parked_not_idled",
                      "a protected unit was parked but no work followed it; parking then "
                      "stopping is idling with extra steps")
    # A wake condition may be recorded as a field or stated in the unit's own
    # description. Demanding one exact key would flag a unit that DOES record its
    # wake -- a narrow probe producing a broad verdict, which is the defect this
    # whole trial exists to catch.
    def _has_wake(event: Mapping[str, Any]) -> bool:
        pay = _payload(event)
        if pay.get("wake_condition"):
            return True
        unit = pay.get("unit")
        if isinstance(unit, Mapping):
            if unit.get("wake_condition"):
                return True
            if "until" in str(unit.get("description") or "").lower():
                return True
        return "until" in str(pay.get("why") or "").lower()

    unspecified = [e for e in parked if not _has_wake(e)]
    if unspecified:
        return _unmet("protected_work_parked_not_idled",
                      f"{len(unspecified)} parked unit(s) record no wake condition anywhere; "
                      "a parked unit with no wake is dropped work")
    return _met("protected_work_parked_not_idled",
                f"parked protected work and continued with {len(after)} later action(s)",
                parked[:3])


EVALUATORS = {
    "recover_state": eval_recover_state,
    "identify_live_frontier": eval_identify_live_frontier,
    "launch_valid_workunit": eval_launch_valid_workunit,
    "durable_mission_state": eval_durable_mission_state,
    "leave_next_work": eval_leave_next_work,
    "maintain_multiple_fronts": eval_maintain_multiple_fronts,
    "ingest_completed_result": eval_ingest_completed_result,
    "reject_bad_idea_on_evidence": eval_reject_bad_idea_on_evidence,
    "refill_work": eval_refill_work,
    "never_conversational_wait": eval_never_conversational_wait,
    "mutation_proposed_and_rolled_back": eval_mutation_proposed_and_rolled_back,
    "status_causality_challenged": eval_status_causality_challenged,
    "protected_work_parked_not_idled": eval_protected_work_parked_not_idled,
    "no_idle_while_work_exists": eval_no_idle_while_work_exists,
    "overlap_detached_work": eval_overlap_detached_work,
    "use_negative_science": eval_use_negative_science,
    "alter_priority_from_evidence": eval_alter_priority_from_evidence,
    "frontier_movement_or_falsification": eval_frontier_movement_or_falsification,
    "recover_process": eval_recover_process,
    "verified_scientific_progress": eval_verified_scientific_progress,
}


# ---------------------------------------------------------------------------
# Verify — judges a timeline. Never writes PASS onto it.
# ---------------------------------------------------------------------------


def verify(trial: str, timeline: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Judge one trial from a recorded timeline. Duration elapsed is not a pass."""
    if trial not in TRIAL_DURATION_S:
        _fail("unknown_trial", f"trial {trial!r} is not one of {list(TRIAL_DURATION_S)}")
    doc = load_timeline(timeline)
    claims = _hardware_claim_paths(doc)
    if claims:
        _fail(
            "hardware_claim_without_hardware",
            f"timeline asserted hardware field(s) {claims[:8]}",
        )
    named = str(doc.get("trial") or trial)
    events = doc.get("events")
    if events is None:
        _fail("malformed_timeline", "timeline has no events list")
    if not isinstance(events, list):
        _fail("malformed_timeline", "timeline.events is not a list")

    view = TimelineView(doc, trial)
    ignored_self_report = None
    if "verdict" in doc or "self_report" in doc or "self_verdict" in doc:
        ignored_self_report = {
            "verdict": doc.get("verdict"),
            "self_report": doc.get("self_report"),
            "self_verdict": doc.get("self_verdict"),
            "note": "self-report is ignored; only mechanical conditions judge",
        }

    auto = detect_automatic_failures(view)
    required = REQUIRED_CONDITIONS[trial]
    conditions = [EVALUATORS[cid](view) for cid in required]
    unmet = [c for c in conditions if not c["met"]]

    elapsed = doc.get("elapsed_s")
    if elapsed is None:
        elapsed = max((int(e.get("t_s") or 0) for e in view.events), default=0)
    try:
        elapsed_s = int(elapsed)
    except (TypeError, ValueError):
        elapsed_s = 0
    duration_s = int(TRIAL_DURATION_S[trial])
    elapsed_meets = elapsed_s >= duration_s

    if auto:
        verdict = "FAIL"
        reason = "; ".join(f["id"] + ": " + f["detail"] for f in auto)
    elif unmet:
        verdict = "FAIL"
        reason = "unmet: " + ", ".join(c["id"] for c in unmet)
    else:
        verdict = "PASS"
        reason = f"all {len(required)} required conditions met on timeline evidence"

    if verdict == "FAIL" and elapsed_meets:
        reason = f"{reason} (duration elapsed is not a pass)"

    citations: list[str] = []
    for item in auto + conditions:
        for cite in item.get("cites") or []:
            if cite not in citations:
                citations.append(cite)

    launched_ids = []
    for event, unit in view.launched():
        if is_valid_workunit(unit)[0]:
            launched_ids.append(str(unit.get("id") or ""))

    return {
        "schema": "hawking.future.autonomy_trial.verdict.v1",
        "trial": trial,
        "timeline_trial": named,
        "verdict": verdict,
        "reason": reason,
        "elapsed_s": elapsed_s,
        "duration_s": duration_s,
        "elapsed_meets_duration": elapsed_meets,
        "elapsed_is_not_a_pass": True,
        "automatic_failures": auto,
        "conditions": conditions,
        "unmet": [c["id"] for c in unmet],
        "required": list(required),
        "citations": citations,
        "valid_workunits_launched": launched_ids,
        "safe_work_remaining": [str(r.get("id") or "") for r in view.safe_work if r.get("id")],
        "frontier_path_taken": view.frontier.get("path_taken"),
        "ignored_self_report": ignored_self_report,
        "n_events": len(view.events),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. "
            "Neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE. "
            "A PASS here is a timeline judgement, not a promotion."
        ),
    }


# ---------------------------------------------------------------------------
# Persist — --verify used to print and write nothing. The launch gate then
# probed AUTONOMY_TRIAL.json (singular) while this module wrote AUTONOMY_TRIALS.json
# (plural). A PASS that exists only on stdout is not a PASS the gate can read.
# The timeline file is never mutated: its raw sha256 is the seal.
# ---------------------------------------------------------------------------


def _owned_receipt_path() -> Path:
    return RECEIPTS / RECEIPT


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def load_owned_receipt() -> dict[str, Any]:
    """Read AUTONOMY_TRIALS.json if it exists. Absent is empty, not a fabricated PASS."""
    path = _owned_receipt_path()
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def timeline_file_digest(path: Path) -> str:
    if not path.is_file():
        _fail("missing_timeline", f"cannot digest absent timeline: {path}")
    return sha256_file(path)


def timeline_internal_seal_state(doc: Mapping[str, Any]) -> dict[str, Any]:
    """If the timeline carries seal_sha256, recompute it. Absent is recorded, not forged."""
    seal = doc.get("seal_sha256")
    if not isinstance(seal, str) or not seal:
        return {
            "present": False,
            "verifies": None,
            "why": "timeline has no internal seal_sha256; file digest is the transcript seal",
        }
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    recomputed = hashlib.sha256(blob).hexdigest()
    ok = recomputed == seal
    return {
        "present": True,
        "verifies": ok,
        "expected": seal,
        "recomputed": recomputed,
        "why": (
            "internal seal_sha256 matches body"
            if ok
            else "internal seal_sha256 does not match body; transcript was edited after sealing"
        ),
    }


def verify_timeline_digest(path: Path | str | None, expected: str | None) -> dict[str, Any]:
    """Re-hash the timeline. A mismatch is a refusal, not a warning.

    The judged process must not be able to edit the transcript after the fact.
    """
    if not expected or not isinstance(expected, str):
        return {
            "verifies": False,
            "why": "persisted verdict has no timeline_seal_digest; refusing",
            "path": None,
            "expected": expected,
            "actual": None,
        }
    if path is None:
        return {
            "verifies": False,
            "why": "persisted verdict has no timeline_path; refusing",
            "path": None,
            "expected": expected,
            "actual": None,
        }
    raw = Path(path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(REPO / raw)
        common = git("rev-parse", "--git-common-dir")
        if common:
            git_path = Path(common)
            if not git_path.is_absolute():
                git_path = (REPO / git_path).resolve()
            parent = git_path.parent if git_path.name == ".git" else git_path.parent
            candidates.append(parent / raw)
    found: Path | None = None
    searched: list[str] = []
    for cand in candidates:
        searched.append(str(cand))
        if cand.is_file():
            found = cand
            break
    if found is None:
        return {
            "verifies": False,
            "why": f"timeline not found for re-hash; searched={searched}",
            "path": str(path),
            "expected": expected,
            "actual": None,
            "searched": searched,
        }
    actual = sha256_file(found)
    if actual != expected:
        return {
            "verifies": False,
            "why": (
                "timeline file digest does not match the persisted seal; "
                "the judged process edited the transcript after the fact, or "
                "the verdict was attached to a different file"
            ),
            "path": str(found),
            "expected": expected,
            "actual": actual,
        }
    internal = timeline_internal_seal_state(load_timeline(found))
    if internal.get("present") and internal.get("verifies") is False:
        return {
            "verifies": False,
            "why": internal.get("why"),
            "path": str(found),
            "expected": expected,
            "actual": actual,
            "internal_seal": internal,
        }
    return {
        "verifies": True,
        "why": "timeline file digest matches the persisted seal",
        "path": str(found),
        "expected": expected,
        "actual": actual,
        "internal_seal": internal,
    }


def extract_orchestration_and_cognition(timeline: Mapping[str, Any]) -> dict[str, Any]:
    """Two facts, never one boolean.

    The 1h loop that passed is HCLI orchestration with no model in it. Writing
    resident_orchestration true is not a way of implying a model was thinking.
    """
    events = timeline.get("events") if isinstance(timeline.get("events"), list) else []
    cognition_values: list[str] = []
    cognition_reasons: list[str] = []
    n_launched = 0
    n_ingested = 0
    bindings_present = False
    said_hcli_loop = False
    for event in events:
        if not isinstance(event, Mapping):
            continue
        kind = str(event.get("kind") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        if kind == "workunit_launched":
            n_launched += 1
        if kind == "result_ingested":
            n_ingested += 1
        if payload.get("bindings_present") is True:
            bindings_present = True
        cog = payload.get("resident_model_cognition")
        if cog is not None:
            cognition_values.append(str(cog))
        why = str(payload.get("why") or "")
        if why and (cog is not None or "cognition" in why.lower() or "orchestr" in why.lower()):
            cognition_reasons.append(why)
        lowered = why.lower()
        if "hcli orchestration" in lowered or "not model cognition" in lowered:
            said_hcli_loop = True

    if cognition_values:
        unique = list(dict.fromkeys(cognition_values))
        if all(v.upper() == COGNITION_UNAVAILABLE for v in unique):
            cognition = COGNITION_UNAVAILABLE
        elif len(unique) == 1:
            cognition = unique[0]
        else:
            cognition = "MIXED:" + ",".join(unique)
    else:
        cognition = COGNITION_UNAVAILABLE
        cognition_reasons.append(
            "timeline did not record resident_model_cognition; refusing to infer a model was thinking"
        )
    reason = cognition_reasons[0] if cognition_reasons else (
        "resident_model_cognition is UNAVAILABLE and no measured reason was attached"
    )
    orchestration = n_launched > 0
    orch_reason = (
        f"HCLI loop launched {n_launched} workunit(s) and ingested {n_ingested} result(s)"
        if orchestration
        else (
            f"no HCLI orchestration evidence: launched={n_launched} ingested={n_ingested} "
            f"bindings_present={bindings_present}"
        )
    )
    return {
        "resident_orchestration": orchestration,
        "resident_orchestration_reason": orch_reason,
        "resident_model_cognition": cognition,
        "resident_model_cognition_reason": reason,
        "n_workunits_launched": n_launched,
        "n_results_ingested": n_ingested,
        "bindings_present": bindings_present,
        "said_hcli_loop": said_hcli_loop,
        "fields_are_independent": True,
        "orchestration_is_not_cognition": True,
    }


def frozen_manifest_record(trial: str) -> dict[str, Any]:
    """Digest of the freeze for this trial, if a freeze receipt exists.

    Live verify_unchanged is recorded as of persist time. Later edits of the
    driver do not get rewritten as CLEAN, and absence is UNAVAILABLE, not CLEAN.
    """
    doc, taken = _read_json_coping(FREEZE_RECEIPT_REL)
    if doc is None:
        return {
            "available": False,
            "digest": None,
            "substrate_verdict": "UNAVAILABLE",
            "why": f"freeze receipt not found ({taken})",
            "path_taken": taken,
            "trial": trial,
        }
    builds = doc.get("frozen_builds") if isinstance(doc.get("frozen_builds"), list) else []
    match: Mapping[str, Any] | None = None
    for row in builds:
        if isinstance(row, Mapping) and str(row.get("trial") or row.get("trial_id") or "") == trial:
            match = row
            break
    if match is None:
        return {
            "available": False,
            "digest": None,
            "substrate_verdict": "UNAVAILABLE",
            "why": f"HCLI_AUTONOMY_BUILD.json has no frozen_builds entry for trial={trial!r}",
            "path_taken": taken,
            "trial": trial,
            "freeze_receipt_seal": doc.get("seal_sha256"),
        }
    blob = json.dumps(dict(match), sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(blob).hexdigest()
    live_verdict = "UNAVAILABLE"
    live_why = "trial_freeze.verify_unchanged was not invoked"
    try:
        from tools.future.trial_freeze import verify_unchanged

        live = verify_unchanged(match)
        live_verdict = str(live.get("verdict") or "UNAVAILABLE")
        live_why = str(live.get("why") or live_verdict)
    except Exception as exc:
        live_verdict = "UNAVAILABLE"
        live_why = f"trial_freeze.verify_unchanged raised {type(exc).__name__}: {exc}"
    return {
        "available": True,
        "digest": digest,
        "substrate_verdict": live_verdict,
        "why": live_why,
        "path_taken": taken,
        "trial": trial,
        "freeze_receipt_seal": doc.get("seal_sha256"),
        "freeze_receipt": FREEZE_RECEIPT_REL,
    }


def compose_persisted_verdict(
    verdict: Mapping[str, Any],
    timeline_path: Path,
    *,
    timeline_doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the record --verify must persist. Does not write."""
    path = Path(timeline_path)
    if not path.is_file():
        _fail("missing_timeline", f"--verify persist requires a real timeline file: {path}")
    digest = timeline_file_digest(path)
    doc = timeline_doc if isinstance(timeline_doc, Mapping) else load_timeline(path)
    facts = extract_orchestration_and_cognition(doc)
    internal = timeline_internal_seal_state(doc)
    freeze = frozen_manifest_record(str(verdict.get("trial") or ""))
    conditions = list(verdict.get("conditions") or [])
    met_ids = [str(c.get("id")) for c in conditions if c.get("met")]
    unmet_ids = [str(c.get("id")) for c in conditions if not c.get("met")]
    if verdict.get("unmet") and not unmet_ids:
        unmet_ids = [str(x) for x in verdict.get("unmet") or []]
    return {
        "schema": VERDICT_PERSIST_SCHEMA,
        "trial": verdict.get("trial"),
        "verdict": verdict.get("verdict"),
        "reason": verdict.get("reason"),
        "conditions_met": met_ids,
        "conditions_unmet": unmet_ids,
        "n_conditions": len(conditions),
        "automatic_failures": [f.get("id") for f in (verdict.get("automatic_failures") or [])],
        "timeline_path": _repo_rel(path),
        "timeline_seal_digest": digest,
        "timeline_internal_seal": internal,
        "frozen_build_manifest_digest": freeze.get("digest"),
        "frozen_build": freeze,
        "resident_orchestration": facts["resident_orchestration"],
        "resident_orchestration_reason": facts["resident_orchestration_reason"],
        "resident_model_cognition": facts["resident_model_cognition"],
        "resident_model_cognition_reason": facts["resident_model_cognition_reason"],
        "orchestration_is_not_cognition": True,
        "fields_are_independent": True,
        "n_workunits_launched": facts["n_workunits_launched"],
        "n_results_ingested": facts["n_results_ingested"],
        "elapsed_s": verdict.get("elapsed_s"),
        "duration_s": verdict.get("duration_s"),
        "elapsed_meets_duration": verdict.get("elapsed_meets_duration"),
        "elapsed_is_not_a_pass": True,
        "n_events": verdict.get("n_events"),
        "launch_eligible_trial": str(verdict.get("trial") or "") in LAUNCH_ELIGIBLE_TRIALS,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "timeline_not_rewritten": True,
        "claim_boundary": (
            "Persisted timeline judgement. resident_orchestration true does not "
            "mean a model was thinking. A FAIL is stored as FAIL."
        ),
    }


def persist_verdict(
    verdict: Mapping[str, Any],
    timeline_path: Path | str,
    *,
    timeline_doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the verdict into AUTONOMY_TRIALS.json. Never writes onto the timeline."""
    path = Path(timeline_path)
    if path.resolve() == _owned_receipt_path().resolve():
        _fail(
            "timeline_is_receipt",
            "refusing to treat AUTONOMY_TRIALS.json as a timeline; persist must not write onto the judged file",
        )
    before = path.read_bytes() if path.is_file() else b""
    record = compose_persisted_verdict(verdict, path, timeline_doc=timeline_doc)
    existing = load_owned_receipt()
    by = dict(existing.get("persisted_verdicts_by_trial") or {})
    trial = str(record.get("trial") or "")
    if not trial:
        _fail("unknown_trial", "cannot persist a verdict with no trial id")
    by[trial] = record
    existing["persisted_verdicts_by_trial"] = by
    existing["last_persisted_verdict"] = record
    existing.setdefault("schema", SCHEMA)
    existing.setdefault("version", VERSION)
    existing.setdefault(
        "purpose",
        "Harness that runs and judges progressive autonomy trials from recorded timelines.",
    )
    existing["evidence_class"] = "STATIC_ONLY"
    existing["gpu_authority"] = False
    existing["verify_persists_verdict"] = True
    existing["receipt_name"] = RECEIPT
    written = write_receipt(RECEIPT, existing, RECORDED_BY)
    after = path.read_bytes() if path.is_file() else b""
    if after != before:
        _fail("timeline_mutated", "persist_verdict must not edit the timeline it judged")
    return {"path": str(written), "record": record, "receipt": RECEIPT}


def launch_candidate_from_receipt(doc: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The 1h-or-longer persisted verdict the launch gate should read. 15m is not enough."""
    if not isinstance(doc, Mapping):
        return None
    by = doc.get("persisted_verdicts_by_trial")
    if isinstance(by, Mapping):
        for trial in ("6h", "3h", "1h"):
            row = by.get(trial)
            if isinstance(row, Mapping):
                return dict(row)
    last = doc.get("last_persisted_verdict")
    if isinstance(last, Mapping) and str(last.get("trial") or "") in LAUNCH_ELIGIBLE_TRIALS:
        return dict(last)
    return None


# ---------------------------------------------------------------------------
# Record — captures a timeline. Never calls verify.
# ---------------------------------------------------------------------------


def snapshot_timeline(trial: str) -> dict[str, Any]:
    if trial not in TRIAL_DURATION_S:
        _fail("unknown_trial", f"trial {trial!r} is not one of {list(TRIAL_DURATION_S)}")
    frontier = load_frontier()
    blockers = load_hardware_blockers()
    safe = remaining_safe_work(frontier)
    live = live_frontier_entries(frontier)
    events: list[dict[str, Any]] = [
        {
            "t_s": 0,
            "seq": 0,
            "kind": "state_recovered",
            "cites": [FRONTIER_REL, QUAL_RECEIPT_REL],
            "payload": {
                "path_taken": frontier.get("path_taken"),
                "path": FRONTIER_REL,
                "n_entries": len(list(frontier.get("entries") or [])),
            },
        }
    ]
    live_ids = [str(row.get("id") or "") for row in live if row.get("id")]
    if live_ids:
        events.append(
            {
                "t_s": 0,
                "seq": 1,
                "kind": "frontier_identified",
                "cites": live_ids,
                "payload": {"entry_ids": live_ids, "path_taken": frontier.get("path_taken")},
            }
        )
    for i, blocker in enumerate(blockers):
        if blocker.get("disposition") != "SLEEPING":
            continue
        if blocker.get("id") not in {b["id"] for b in CODEX_REPORTED_BLOCKERS}:
            continue
        unit = sleeping_hardware_unit(
            f"future.autonomy.sleeping.{blocker['id']}",
            blocker_id=str(blocker["id"]),
            reason=str(blocker.get("statement") or blocker["id"]),
            frontier_id="F001",
        )
        events.append(
            {
                "t_s": 0,
                "seq": 2 + i,
                "kind": "workunit_sleeping",
                "cites": [unit["id"], str(blocker["id"])],
                "payload": {"unit": unit, "blocker_id": blocker["id"]},
            }
        )
    return {
        "schema": TIMELINE_SCHEMA,
        "trial": trial,
        "duration_s": TRIAL_DURATION_S[trial],
        "elapsed_s": 0,
        "frontier": {
            "path": FRONTIER_REL,
            "path_taken": frontier.get("path_taken"),
            "present": frontier.get("present"),
            "entries": frontier.get("entries") or [],
            "resolved_entries": frontier.get("resolved_entries") or [],
            "stale_entries": frontier.get("stale_entries") or [],
        },
        "hardware_blockers": blockers,
        "safe_work_remaining": [
            {"id": r.get("id"), "classification": r.get("classification"), "title": r.get("title")}
            for r in safe
        ],
        "events": _seq_events(events),
        "queue": [],
    }


def append_event(
    doc: Mapping[str, Any],
    event: Mapping[str, Any],
    *,
    t_s: int | None = None,
) -> dict[str, Any]:
    out = dict(doc)
    events = _seq_events(list(out.get("events") or []))
    nxt = dict(event)
    if t_s is not None:
        nxt["t_s"] = int(t_s)
    elif "t_s" not in nxt:
        nxt["t_s"] = (events[-1]["t_s"] + 1) if events else 0
    nxt["seq"] = (events[-1]["seq"] + 1) if events else 0
    events.append(nxt)
    out["events"] = _seq_events(events)
    if "elapsed_s" in out:
        try:
            out["elapsed_s"] = max(int(out.get("elapsed_s") or 0), int(nxt["t_s"]))
        except (TypeError, ValueError):
            out["elapsed_s"] = int(nxt["t_s"])
    return out


def record(
    trial: str,
    path: str | Path,
    *,
    init: bool = False,
    event: Mapping[str, Any] | None = None,
    t_s: int | None = None,
) -> Path:
    """Capture a trial timeline. Does not judge it."""
    dest = Path(path)
    if init:
        write_timeline(dest, snapshot_timeline(trial))
    elif not dest.is_file() and event is None:
        _fail("nothing_to_record", "pass --init to snapshot, or --event to append")
    elif not dest.is_file():
        _fail("missing_timeline", f"cannot append; {dest} does not exist (pass --init)")
    if event is not None:
        current = load_timeline(dest)
        if current.get("trial") not in (None, "", trial):
            _fail(
                "trial_mismatch",
                f"timeline.trial={current.get('trial')!r} does not match --trial {trial}",
            )
        write_timeline(dest, append_event(current, event, t_s=t_s))
    if not dest.is_file():
        _fail("nothing_to_record", "pass --init to snapshot, or --event to append")
    return dest


# ---------------------------------------------------------------------------
# Fixtures — watched to pass and to fail
# ---------------------------------------------------------------------------


def fixture_frontier() -> dict[str, Any]:
    """Embedded frontier so tests do not encode the live checkout's entry count."""
    return {
        "path": FRONTIER_REL,
        "path_taken": "fixture",
        "present": True,
        "resolved_entries": ["F003"],
        "stale_entries": [],
        "entries": [
            {
                "id": "F012",
                "classification": "HIGH_VALUE_INTEGRATION",
                "resource_need": "CPU only",
                "title": "Architecture Atlas is strong — consume it",
            },
            {
                "id": "F015",
                "classification": "HIGH_VALUE_INTEGRATION",
                "resource_need": "CPU only, read-only on the Codex surface",
                "title": "Codex receipts are never ingested into anything downstream",
            },
            {
                "id": "F016",
                "classification": "HIGH_VALUE_INTEGRATION",
                "resource_need": "CPU only",
                "title": "Ingest deltas are emitted but nothing applies them",
            },
            {
                "id": "F007",
                "classification": "MISSING",
                "resource_need": "CPU only",
                "title": "No Learned Physical Compiler dataset contract",
            },
            {
                "id": "F001",
                "classification": "BLOCKED",
                "resource_need": "GPU authority for the eventual measurement",
                "title": "Flash source-independent NX is the single dominant blocker",
            },
            {
                "id": "F002",
                "classification": "BLOCKED",
                "resource_need": "GPU authority (Codex lane)",
                "title": "Qwen27 candidates idle on a GPU window",
            },
        ],
    }


def _ev(t_s: int, kind: str, cites: Sequence[str] | None = None, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "t_s": int(t_s),
        "kind": kind,
        "cites": list(cites or []),
        "payload": dict(payload or {}),
    }


def passing_units() -> dict[str, dict[str, Any]]:
    return {
        "atlas": cpu_workunit(
            "future.autonomy.cpu-atlas-consume",
            frontier_id="F012",
            description="Consume Architecture Atlas into HWIR/LPC planning; CPU-only, no GPU lease.",
        ),
        "ingest": cpu_workunit(
            "future.autonomy.cpu-codex-ingest",
            frontier_id="F015",
            description="Ingest a sealed Codex receipt into the compounding loop; read-only on headless.",
        ),
        "lpc": cpu_workunit(
            "future.autonomy.cpu-lpc-contract",
            frontier_id="F007",
            description="Draft the learned-physical-compiler dataset contract with contamination required.",
        ),
        "next": cpu_workunit(
            "future.autonomy.cpu-propagate-deltas",
            frontier_id="F016",
            description="Apply ingest deltas onto Odyssey II/III consumers after the current unit.",
        ),
    }


def sleeping_fixture_unit() -> dict[str, Any]:
    return sleeping_hardware_unit(
        "future.autonomy.sleeping.flash-nx",
        blocker_id="flash_nx_scaffold_only",
        reason="Flash source-independent NX is SCAFFOLD_ONLY; Metal GPU absent",
        frontier_id="F001",
    )


def build_passing_timeline(trial: str) -> dict[str, Any]:
    """A timeline that meets every required condition for `trial`, including full duration."""
    if trial not in TRIAL_DURATION_S:
        _fail("unknown_trial", f"trial {trial!r}")
    units = passing_units()
    u1, u2, u3, nxt = units["atlas"], units["ingest"], units["lpc"], units["next"]
    sleep = sleeping_fixture_unit()
    needed = set(REQUIRED_CONDITIONS[trial])
    events = [
        _ev(
            0,
            "state_recovered",
            cites=[FRONTIER_REL],
            payload={"path_taken": "fixture", "path": FRONTIER_REL},
        ),
        _ev(
            1,
            "frontier_identified",
            cites=["F012", "F015", "F007", "F016"],
            payload={"entry_ids": ["F012", "F015", "F007", "F016"]},
        ),
        _ev(
            2,
            "workunit_sleeping",
            cites=[sleep["id"], "F001"],
            payload={"unit": sleep, "blocker_id": "flash_nx_scaffold_only"},
        ),
        _ev(
            3,
            "workunit_launched",
            cites=[u1["id"], "F012"],
            payload={"unit": u1, "frontier_id": "F012"},
        ),
        _ev(
            4,
            "mission_state_written",
            cites=["mission/state.json"],
            payload={
                "path": "mission/state.json",
                "mission_id": "autonomy-trial-fixture",
                "phase": "running",
                "next_action": "dispatch the next dependency-ready work unit",
                "units": {
                    u1["id"]: {"status": "running"},
                    nxt["id"]: {"status": "pending"},
                },
            },
        ),
        _ev(5, "next_work_left", cites=[nxt["id"], u2["id"]], payload={"unit_ids": [nxt["id"], u2["id"]]}),
    ]
    if "maintain_multiple_fronts" in needed:
        events.append(
            _ev(
                6,
                "workunit_launched",
                cites=[u2["id"], "F015"],
                payload={"unit": u2, "frontier_id": "F015"},
            )
        )
        events.append(
            _ev(
                7,
                "front_maintained",
                cites=["F012", "F015"],
                payload={"frontier_ids": ["F012", "F015"]},
            )
        )
    if "ingest_completed_result" in needed:
        events.append(
            _ev(
                10,
                "result_ingested",
                cites=["receipts/future/CODEX_INGEST_STATE.json", u1["id"]],
                payload={
                    "receipt": "receipts/future/CODEX_INGEST_STATE.json",
                    "unit_id": u1["id"],
                },
            )
        )
    if "reject_bad_idea_on_evidence" in needed:
        events.append(
            _ev(
                11,
                "idea_rejected",
                cites=["receipts/future/NEGATIVE_SCIENCE_INDEX.json", "cross_expert_structure"],
                payload={
                    "idea": "cross_expert_structure on qwen3-235b-a22b",
                    "reason": "keyed scar: refuse_if_dead",
                    "proposal": {
                        "model": "qwen3-235b-a22b",
                        "hypothesis_family": "cross_expert_structure",
                    },
                },
            )
        )
    if "refill_work" in needed:
        events.append(
            _ev(
                12,
                "work_refilled",
                cites=[u3["id"], "F007"],
                payload={"unit": u3, "unit_ids": [u3["id"]]},
            )
        )
        events.append(
            _ev(
                13,
                "workunit_launched",
                cites=[u3["id"], "F007"],
                payload={"unit": u3, "frontier_id": "F007"},
            )
        )
    if "overlap_detached_work" in needed:
        events.extend(
            [
                _ev(20, "detached_started", cites=["job-a"], payload={"job_id": "job-a"}),
                _ev(21, "detached_started", cites=["job-b"], payload={"job_id": "job-b"}),
                _ev(40, "detached_completed", cites=["job-a"], payload={"job_id": "job-a"}),
                _ev(41, "detached_completed", cites=["job-b"], payload={"job_id": "job-b"}),
            ]
        )
    if "use_negative_science" in needed:
        events.append(
            _ev(
                22,
                "negative_science_refusal",
                cites=["receipts/future/NEGATIVE_SCIENCE_INDEX.json"],
                payload={
                    "query": {
                        "model": "qwen3-235b-a22b",
                        "hypothesis_family": "cross_expert_structure",
                    },
                    "source_path": "receipts/future/NEGATIVE_SCIENCE_INDEX.json",
                },
            )
        )
    if "alter_priority_from_evidence" in needed:
        events.append(
            _ev(
                23,
                "priority_altered",
                cites=["receipts/future/CODEX_INGEST_STATE.json"],
                payload={
                    "before": [u1["id"], u2["id"], u3["id"]],
                    "after": [u3["id"], u1["id"], u2["id"]],
                },
            )
        )
    if "frontier_movement_or_falsification" in needed:
        events.append(
            _ev(
                30,
                "frontier_moved",
                cites=["F007"],
                payload={
                    "entry_id": "F007",
                    "from": "MISSING",
                    "to": "WEAK",
                    "delta": "dataset contract drafted; still not a promotion",
                },
            )
        )
    if "recover_process" in needed:
        events.extend(
            [
                _ev(
                    31,
                    "process_failed",
                    cites=["job-a"],
                    payload={"process_id": "job-a", "reason": "exit 1"},
                ),
                _ev(
                    32,
                    "process_recovered",
                    cites=["job-a"],
                    payload={"process_id": "job-a"},
                ),
            ]
        )
    if "mutation_proposed_and_rolled_back" in needed:
        events.append(
            _ev(
                51,
                "result_ingested",
                cites=["receipts/future/MUTATION_ENGINE.json",
                       "WU.TORTURE.mutation"],
                payload={
                    "unit_id": "WU.TORTURE.mutation",
                    "receipt": "receipts/future/MUTATION_ENGINE.json",
                    "routed_to_frontier": "FT.HCLI_SELF.emit-workunits",
                },
            )
        )
    if "status_causality_challenged" in needed:
        events.append(
            _ev(
                52,
                "result_ingested",
                cites=["receipts/future/STATUS_CAUSALITY_CHALLENGE.json",
                       "WU.TORTURE.status_challenge"],
                payload={
                    "unit_id": "WU.TORTURE.status_challenge",
                    "receipt": "receipts/future/STATUS_CAUSALITY_CHALLENGE.json",
                    "routed_to_frontier": "FT.VERIFICATION.negative-index",
                },
            )
        )
    if "protected_work_parked_not_idled" in needed:
        events.append(
            _ev(
                53,
                "workunit_sleeping",
                payload={
                    "unit_id": "WU.TORTURE.protected_ab",
                    "resource_class": "GPU_PROTECTED",
                    "status": "BLOCKED_ON_PROTECTED_WINDOW",
                    "wake_condition": "a QUIESCENT machine and a proven HCLI lease",
                    "why": "protected-required work parks; it never becomes a synthetic result",
                },
            )
        )
        # Parking then stopping is idling with extra steps, so the fixture has to
        # show the loop kept working after it parked.
        events.append(
            _ev(
                54,
                "workunit_launched",
                payload={
                    "unit": cpu_workunit(
                        "WU.TORTURE.after_park",
                        frontier_id="FT.TOOLS.freshness",
                        description="continue with CPU work while protected work sleeps",
                    ),
                    "capability": "freshness.py",
                },
            )
        )
        events.append(
            _ev(
                55,
                "result_ingested",
                cites=["receipts/future/DERIVED_FRESHNESS.json", "WU.TORTURE.after_park"],
                payload={
                    "unit_id": "WU.TORTURE.after_park",
                    "receipt": "receipts/future/DERIVED_FRESHNESS.json",
                    "routed_to_frontier": "FT.TOOLS.freshness",
                },
            )
        )
    if "verified_scientific_progress" in needed:
        events.append(
            _ev(
                50,
                "scientific_progress",
                cites=["F007", "frontier_moved"],
                payload={"kind": "frontier_delta", "entry_id": "F007"},
            )
        )
    return {
        "schema": TIMELINE_SCHEMA,
        "trial": trial,
        "duration_s": TRIAL_DURATION_S[trial],
        "elapsed_s": TRIAL_DURATION_S[trial],
        "frontier": fixture_frontier(),
        "events": _seq_events(events),
        "queue": [u1, nxt],
    }


def fixture_duration_without_workunit(trial: str = "15m") -> dict[str, Any]:
    """Full duration elapsed, recovered state, no valid launch — must FAIL."""
    return {
        "schema": TIMELINE_SCHEMA,
        "trial": trial,
        "duration_s": TRIAL_DURATION_S[trial],
        "elapsed_s": TRIAL_DURATION_S[trial],
        "frontier": fixture_frontier(),
        "verdict": "PASS",
        "self_report": "waited the whole window",
        "events": _seq_events(
            [
                _ev(
                    0,
                    "state_recovered",
                    cites=[FRONTIER_REL],
                    payload={"path_taken": "fixture", "path": FRONTIER_REL},
                ),
                _ev(
                    1,
                    "frontier_identified",
                    cites=["F012"],
                    payload={"entry_ids": ["F012"]},
                ),
                _ev(
                    TRIAL_DURATION_S[trial],
                    "idle",
                    cites=[],
                    payload={"message": "timer expired"},
                ),
            ]
        ),
    }


def fixture_awaiting_with_safe_work() -> dict[str, Any]:
    units = passing_units()
    u1 = units["atlas"]
    return {
        "schema": TIMELINE_SCHEMA,
        "trial": "15m",
        "duration_s": TRIAL_DURATION_S["15m"],
        "elapsed_s": TRIAL_DURATION_S["15m"],
        "frontier": fixture_frontier(),
        "events": _seq_events(
            [
                _ev(0, "state_recovered", cites=[FRONTIER_REL], payload={"path_taken": "fixture"}),
                _ev(1, "frontier_identified", cites=["F012"], payload={"entry_ids": ["F012"]}),
                _ev(2, "workunit_launched", cites=[u1["id"], "F012"], payload={"unit": u1, "frontier_id": "F012"}),
                _ev(
                    3,
                    "mission_state_written",
                    cites=["mission/state.json"],
                    payload={
                        "path": "mission/state.json",
                        "mission_id": "await-fixture",
                        "next_action": "inspect the result envelope",
                        "units": {u1["id"]: {"status": "completed"}},
                    },
                ),
                _ev(4, "next_work_left", cites=[units["next"]["id"]], payload={"unit_ids": [units["next"]["id"]]}),
                _ev(
                    100,
                    "awaiting_instructions",
                    cites=[],
                    payload={"message": "all tasks complete, awaiting instructions"},
                ),
            ]
        ),
    }


def fixture_busywork_flood() -> dict[str, Any]:
    clones = [
        cpu_workunit(
            f"future.autonomy.busy-{i}",
            frontier_id="F012",
            description="do work",
            verifier="future.autonomy_trial.busy",
        )
        for i in range(4)
    ]
    events = [
        _ev(0, "state_recovered", cites=[FRONTIER_REL], payload={"path_taken": "fixture"}),
        _ev(1, "frontier_identified", cites=["F012"], payload={"entry_ids": ["F012"]}),
    ]
    for i, unit in enumerate(clones):
        events.append(
            _ev(
                2 + i,
                "workunit_launched",
                cites=[unit["id"], "F012"],
                payload={"unit": unit, "frontier_id": "F012"},
            )
        )
    return {
        "schema": TIMELINE_SCHEMA,
        "trial": "15m",
        "duration_s": TRIAL_DURATION_S["15m"],
        "elapsed_s": TRIAL_DURATION_S["15m"],
        "frontier": fixture_frontier(),
        "events": _seq_events(events),
        "queue": clones,
    }


def fixture_hardware_idle() -> dict[str, Any]:
    sleep = sleeping_fixture_unit()
    return {
        "schema": TIMELINE_SCHEMA,
        "trial": "15m",
        "duration_s": TRIAL_DURATION_S["15m"],
        "elapsed_s": TRIAL_DURATION_S["15m"],
        "frontier": fixture_frontier(),
        "events": _seq_events(
            [
                _ev(0, "state_recovered", cites=[FRONTIER_REL], payload={"path_taken": "fixture"}),
                _ev(1, "frontier_identified", cites=["F012", "F001"], payload={"entry_ids": ["F012", "F001"]}),
                _ev(
                    2,
                    "workunit_sleeping",
                    cites=[sleep["id"], "F001"],
                    payload={"unit": sleep, "blocker_id": "no_metal_gpu"},
                ),
                _ev(
                    3,
                    "hardware_blocked_wait",
                    cites=["F001"],
                    payload={"reason": "waiting for Metal GPU / protected lease", "lane": "gpu"},
                ),
                _ev(
                    4,
                    "idle",
                    cites=[],
                    payload={"reason": "gpu blocked", "message": "cannot proceed without Metal"},
                ),
            ]
        ),
    }


def prove_negative_controls() -> list[dict[str, Any]]:
    """A guard nobody has watched fail is not a guard."""
    proofs: list[dict[str, Any]] = []

    duration = verify("15m", fixture_duration_without_workunit())
    proofs.append(
        {
            "id": "full_duration_without_valid_workunit",
            "fired": duration["verdict"] == "FAIL" and "launch_valid_workunit" in duration["unmet"],
            "verdict": duration["verdict"],
            "unmet": duration["unmet"],
            "elapsed_meets_duration": duration["elapsed_meets_duration"],
            "ignored_self_report": duration["ignored_self_report"] is not None,
        }
    )

    awaiting = verify("15m", fixture_awaiting_with_safe_work())
    auto_a = {f["id"] for f in awaiting["automatic_failures"]}
    proofs.append(
        {
            "id": "awaiting_instructions_while_safe_work_remains",
            "fired": awaiting["verdict"] == "FAIL" and AUTO_FAIL_AWAITING in auto_a,
            "verdict": awaiting["verdict"],
            "automatic_failures": sorted(auto_a),
        }
    )

    flood = verify("15m", fixture_busywork_flood())
    auto_c = {f["id"] for f in flood["automatic_failures"]}
    proofs.append(
        {
            "id": "queue_flooded_with_busywork",
            "fired": flood["verdict"] == "FAIL" and AUTO_FAIL_BUSYWORK in auto_c,
            "verdict": flood["verdict"],
            "automatic_failures": sorted(auto_c),
        }
    )

    hw = verify("15m", fixture_hardware_idle())
    auto_b = {f["id"] for f in hw["automatic_failures"]}
    proofs.append(
        {
            "id": "idle_because_one_hardware_lane_blocked",
            "fired": hw["verdict"] == "FAIL" and AUTO_FAIL_HARDWARE_IDLE in auto_b,
            "verdict": hw["verdict"],
            "automatic_failures": sorted(auto_b),
        }
    )

    unfired = [p["id"] for p in proofs if not p["fired"]]
    if unfired:
        _fail("negative_control", f"guards did not fire: {unfired}")
    return proofs


def prove_passing_timelines() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in TRIAL_IDS:
        verdict = verify(trial, build_passing_timeline(trial))
        rows.append(
            {
                "trial": trial,
                "verdict": verdict["verdict"],
                "unmet": verdict["unmet"],
                "automatic_failures": [f["id"] for f in verdict["automatic_failures"]],
                "n_conditions": len(verdict["conditions"]),
            }
        )
        if verdict["verdict"] != "PASS":
            _fail(
                "passing_fixture",
                f"{trial} fixture failed: {verdict['reason']}",
            )
    short = verify("6h", build_passing_timeline("15m"))
    rows.append(
        {
            "trial": "15m_timeline_judged_as_6h",
            "verdict": short["verdict"],
            "unmet": short["unmet"],
            "note": "a 15m-complete timeline must not satisfy 6h",
        }
    )
    if short["verdict"] != "FAIL":
        _fail("progressive_conditions", "15m timeline incorrectly PASSed as 6h")
    return rows


# ---------------------------------------------------------------------------
# Resident-callable WorkUnits
# ---------------------------------------------------------------------------


def emit_trial_workunits() -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for trial in TRIAL_IDS:
        units.append(
            cpu_workunit(
                f"future.autonomy.trial-{trial}",
                frontier_id="F008",
                description=(
                    f"Record and then separately verify the {trial} autonomy trial "
                    f"({TRIAL_DURATION_S[trial]}s window). Judge the timeline; "
                    "a timer expiring is not a pass."
                ),
                verifier="future.autonomy_trial.verify",
            )
        )
    units.append(
        cpu_workunit(
            "future.autonomy.judge",
            frontier_id="F008",
            description=(
                "Judge a recorded autonomy-trial timeline. Must not be the same "
                "invocation that recorded it."
            ),
            verifier="future.autonomy_trial.verify",
        )
    )
    for blocker in CODEX_REPORTED_BLOCKERS:
        units.append(
            sleeping_hardware_unit(
                f"future.autonomy.sleeping.{blocker['id']}",
                blocker_id=blocker["id"],
                reason=blocker["statement"],
                frontier_id="F001" if "metal" in blocker["id"] or "nx" in blocker["id"] or "teacher" in blocker["id"] else "F002",
            )
        )
    for row in units:
        ws.validate_emitted_unit(row)
    units.sort(key=lambda r: str(r["id"]))
    return units


def recovered_implementation() -> list[dict[str, Any]]:
    return [
        {
            "path": "hcli/agentos/runtime.py",
            "what": (
                "AgentOS.recover_mission / continue_mission / checkpoint; "
                "disk is authority. Read-only here; no resident process started."
            ),
            "present_in_this_checkout": (REPO / "hcli/agentos/runtime.py").is_file(),
        },
        {
            "path": "hcli/agentos/states.py",
            "what": "AgentState vocabulary including WAITING_RESOURCE / FAILED_RECOVERABLE",
            "present_in_this_checkout": (REPO / "hcli/agentos/states.py").is_file(),
        },
        {
            "path": "hcli/agentos/autonomy_gate.py",
            "what": (
                "HCLI unattended-window / A1-A5 qualification already exists. "
                "This sidecar judges progressive 15m/1h/3h/6h timelines; it does "
                "not spawn the native resident that autonomy_gate would start."
            ),
            "present_in_this_checkout": (REPO / "hcli/agentos/autonomy_gate.py").is_file(),
        },
        {
            "path": "tools/future/workunit_species.py",
            "what": "HCLI-shaped WorkUnit emission + validate_emitted_unit; GPU units park SLEEPING",
        },
        {
            "path": "tools/future/negative_index.py",
            "what": "query / refuse_if_dead — the evidence path for idea_rejected",
        },
        {
            "path": "tools/future/repro_science.py",
            "what": "FailClosed; skip is not pass; a guard unseen to fail is not a guard",
        },
        {
            "path": "tools/future/trial_freeze.py",
            "what": (
                "Freeze manifest digest is copied into the persisted verdict when "
                "HCLI_AUTONOMY_BUILD.json names this trial; absence is UNAVAILABLE, not CLEAN"
            ),
        },
        {
            "path": "receipts/future/AUTONOMY_TIMELINE_1h.json",
            "what": (
                "Live 1h timeline. --verify persists the judgement into AUTONOMY_TRIALS.json; "
                "the timeline file is not rewritten."
            ),
            "present_in_this_checkout": (REPO / "receipts/future/AUTONOMY_TIMELINE_1h.json").is_file(),
        },
        {
            "path": "tools/future/integration_attack.py",
            "what": "Adversarial completion attack. Trial verdicts hunt reasons to FAIL.",
        },
        {
            "path": FRONTIER_REL,
            "what": "Live frontier a trial starts from. Disk state is authority.",
        },
        {
            "path": QUAL_RECEIPT_REL,
            "what": "Machine classified HEAVY; lock observed without flock.",
        },
        {
            "path": "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json",
            "what": "49-system sidecar inventory; scaffolding is no longer the goal.",
        },
        {
            "path": "CODEX_ACCELERATOR_HANDOFF.json",
            "what": (
                "Not materialized in this worktree. Treated as a large manual "
                "training trace when present; not required to judge a timeline."
            ),
            "present_in_this_checkout": (REPO / "CODEX_ACCELERATOR_HANDOFF.json").is_file(),
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "Four progressive trials (15m/1h/3h/6h) with hard behavioural conditions judged from a timeline.",
        "A staged event (staged/synthetic/injected_for_condition) cannot satisfy refill_work, overlap_detached_work, use_negative_science, or alter_priority_from_evidence.",
        "The 30m trial is frozen (power_torture.hash_substrate before==after), run for 1800s, and judged from the sealed timeline; degeneracy.measure and no_wait_orchestration.classify run over that same transcript.",
        "A timer is not a pass: --verify fails unmet conditions even when elapsed_s >= duration_s.",
        "Three automatic failures encoded and watched to fire: awaiting-instructions, hardware-idle, busywork flood.",
        "--record captures; --verify judges; combining them in one invocation is refused.",
        "Self-reported PASS on a timeline is ignored.",
        "Blocked physical work becomes a SLEEPING WorkUnit and never a synthetic hardware result.",
        "Verdicts cite timeline seqs, WorkUnit ids, receipts, rejected ideas, frontier deltas.",
        "Resident-callable CLI + HCLI-shaped WorkUnits + AUTONOMY_TRIALS.json receipt.",
        "--verify persists trial id, verdict, met/unmet conditions, timeline path + file digest, and freeze digest into AUTONOMY_TRIALS.json.",
        "resident_orchestration and resident_model_cognition are persisted as separate fields; orchestration true is not cognition.",
        "A FAIL verdict is persisted as FAIL. A timeline digest mismatch is a refusal.",
    ]


def negative_findings() -> list[str]:
    return [
        "This lane did not start an HCLI resident model process and did not take a GPU lease.",
        "CODEX_ACCELERATOR_HANDOFF.json is not materialized in this sparse worktree; Codex blockers are taken from the contract list plus pinned evidence receipts.",
        "This-wave siblings (frontiers, detached, wakeup, workgraph, sandbox, super_resident, odyssey_launch, …) were not imported; local interfaces are named as integration points.",
        "hcli/agentos/autonomy_gate.py is recovered read-only and is not executed (it starts a native resident).",
        "Machine HEAVY and lock-holder-unproven are recovered from QUALIFICATION_PIPELINE.json / Path.is_file, not by flock or by signalling workers.",
        "Metal GPU absence and xcrun Metal-compiler absence were not re-probed; re-measuring would be a hardware claim this sidecar must not make.",
        "A PASS on a recorded fixture is not a PASS of a live 6h resident run. No live multi-hour trial was executed here.",
        "FPGA remains Accelerator/Physical Compiler/Fusion; this harness does not build an FPGA backend.",
        "The 1h loop that passed is HCLI orchestration; resident_model_cognition is UNAVAILABLE throughout. Those facts are stored separately.",
        "This module writes AUTONOMY_TRIALS.json (plural). AUTONOMY_TRIAL.json (singular) is not this receipt and is not written.",
    ]


def resident_callable(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "entry_point": "python3 tools/future/autonomy_trial.py --verify 15m --timeline PATH",
        "also": [
            "python3 tools/future/autonomy_trial.py --record --trial 15m --timeline PATH --init",
            "python3 tools/future/autonomy_trial.py --selftest",
            "python3 tools/future/autonomy_trial.py --build",
        ],
        "invoke": {
            "verify": "tools.future.autonomy_trial.verify(trial, timeline) -> verdict dict",
            "persist_verdict": "tools.future.autonomy_trial.persist_verdict(verdict, timeline_path)",
            "record": "tools.future.autonomy_trial.record(trial, path, init=..., event=...)",
            "emit": "tools.future.autonomy_trial.emit_trial_workunits()",
        },
        "workunit_emitted": [str(u.get("id") or "") for u in units],
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier_fed": (
            "Verdict citations and next_workunits. A PASS does not retire a frontier "
            "entry (this lane cannot promote). A FAIL names unmet conditions and "
            "automatic failures as remaining work. Sleeping GPU units stay SLEEPING."
        ),
        "fail_closed": [
            "missing or malformed timeline",
            "--record and --verify in one invocation (self-grade)",
            "unknown trial id / timeline.trial mismatch",
            "numeric hardware field on the timeline",
            "unmet required condition",
            "automatic failure (awaiting / hardware-idle / busywork)",
            "self-reported PASS ignored",
            "negative-control guard that does not fire aborts --selftest",
            "timeline digest mismatch on a persisted verdict",
            "collapsing resident_orchestration into resident_model_cognition",
        ],
        "how_it_fails_closed": (
            "verify() raises FailClosed when it cannot judge, and returns verdict=FAIL "
            "when it can judge and a condition or automatic failure is unmet. "
            "CLI exit status is 1 in both cases. Duration elapsed never flips FAIL to PASS."
        ),
    }


# ---------------------------------------------------------------------------
# Independent 30m judgement from the SEALED timeline (not the runner summary).
# G030 + G036: freeze, run the real 30 minutes, quote the four conditions
# from events at real call sites, run degeneracy + no-wait over THIS run.
# ---------------------------------------------------------------------------


def campaign_science_scars_reachable() -> dict[str, Any]:
    """refuse_if_dead must key this campaign's own scars. Reachability is live."""
    from tools.future import negative_index as ni

    reachable: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    pool: list[Any] | None
    try:
        pool = ni.ingest()
    except Exception as exc:
        return {
            "ok": False,
            "reachable": {},
            "missing": list(CAMPAIGN_SCIENCE_SCARS),
            "why": f"negative_index.ingest failed: {type(exc).__name__}: {exc}",
        }
    for name in CAMPAIGN_SCIENCE_SCARS:
        try:
            dead = ni.refuse_if_dead({"hypothesis_family": name}, pool)
        except TypeError:
            dead = ni.refuse_if_dead({"hypothesis_family": name})
        except Exception as exc:
            missing.append(name)
            reachable[name] = {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        if dead:
            reachable[name] = {
                "reachable": True,
                "scar_id": dead.get("scar_id") or dead.get("id"),
                "source_path": dead.get("source_path"),
                "verdict": dead.get("verdict") or dead.get("status"),
            }
        else:
            missing.append(name)
            reachable[name] = {"reachable": False}
    return {
        "ok": not missing,
        "reachable": reachable,
        "missing": missing,
        "n_reachable": sum(1 for row in reachable.values() if row.get("reachable")),
        "n_named": len(CAMPAIGN_SCIENCE_SCARS),
        "index": NEG_INDEX_REL,
    }


def _priority_zero_start(event: Mapping[str, Any]) -> bool:
    """A detached_started that is the long/composer-detached job (priority 0).

    The falsy-zero defect ranked these last (`prio or 99`) so they never
    started. An assertion that the sort is fixed is not a job that ran.
    """
    if str(event.get("kind") or "") != "detached_started":
        return False
    if event_is_staged(event):
        return False
    payload = _payload(event)
    pid = payload.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    if payload.get("started_at") is None:
        return False
    if payload.get("long_subprocess") is True:
        return True
    if payload.get("priority") == 0:
        return True
    launch = str(payload.get("launch") or "").lower()
    if launch in {"detached", "no_wait", "no-wait"}:
        return True
    cap = str(payload.get("capability") or "")
    uid = str(payload.get("unit_id") or payload.get("job_id") or "")
    if "specimen_verify" in cap or "specimen_verify" in uid:
        return True
    if "NO_WAIT" in uid or "TORTURE.NO_WAIT" in uid:
        return True
    return False


def _quote_event(event: Mapping[str, Any], *, observation: str) -> dict[str, Any]:
    kind = str(event.get("kind") or "")
    payload = _payload(event)
    return {
        "kind": kind,
        "seq": event.get("seq"),
        "t_s": event.get("t_s"),
        "cites": _cites(event),
        "call_site": FOUR_CALL_SITES.get(kind) or payload.get("call_site"),
        "observation": observation,
        "staged": event_is_staged(event),
        "payload_excerpt": {
            k: payload.get(k)
            for k in (
                "job_id",
                "pid",
                "started_at",
                "finished_at",
                "unit_id",
                "capability",
                "unit_ids",
                "n",
                "queue_remaining_when_asked",
                "source",
                "why",
                "before",
                "after",
                "cause",
                "query",
                "source_path",
                "scar_id",
                "hypothesis_family",
                "job_ids",
                "n_live",
                "how",
            )
            if k in payload
        },
    }


def judge_four_from_sealed(timeline: Mapping[str, Any]) -> dict[str, Any]:
    """Judge the four G030 conditions from the sealed transcript, not the summary.

    Each condition quotes the event, its call site, and what it observed.
    A staged event is an automatic miss. overlap_detached_work additionally
    requires a priority-0 detached job that actually STARTED and overlapped.
    use_negative_science additionally requires this campaign's scars to be
    reachable in the live index.
    """
    view = TimelineView(timeline, "30m")
    rows: dict[str, dict[str, Any]] = {}

    # refill_work
    refill_eval = eval_refill_work(view)
    refill_hits = [
        e
        for e in view.of("work_refilled")
        if not event_is_staged(e)
        and (list(_payload(e).get("unit_ids") or []) or _cites(e) or _payload(e).get("unit"))
    ]
    refill_quote = None
    if refill_hits:
        ev = refill_hits[0]
        pay = _payload(ev)
        refill_quote = _quote_event(
            ev,
            observation=(
                f"frontiers.refill returned n={pay.get('n')} novel unit_ids "
                f"{list(pay.get('unit_ids') or [])[:8]} with "
                f"queue_remaining_when_asked={pay.get('queue_remaining_when_asked')} "
                f"source={pay.get('source')!r} after a result_ingested"
            ),
        )
    rows["refill_work"] = {
        **refill_eval,
        "quote": refill_quote,
        "staged_cannot_satisfy": True,
    }

    # overlap_detached_work + priority-0 start
    overlap_eval = eval_overlap_detached_work(view)
    live_events = [e for e in view.events if not event_is_staged(e)]
    ok, jobs, cited = _detached_overlap(live_events)
    starts = [
        e
        for e in live_events
        if str(e.get("kind") or "") == "detached_started"
    ]
    prio0 = [e for e in starts if _priority_zero_start(e)]
    confirmed = [
        e
        for e in live_events
        if str(e.get("kind") or "") == "detached_overlap_confirmed"
        and not event_is_staged(e)
    ]
    prio0_ids = []
    for ev in prio0:
        pay = _payload(ev)
        token = str(pay.get("job_id") or pay.get("unit_id") or "")
        if token:
            prio0_ids.append(token)
    prio0_in_overlap = bool(ok) and any(
        any(token in str(j) or str(j) in token for j in jobs) for token in prio0_ids
    )
    # Also match by the overlapping job_ids against start events that are prio0.
    if ok and not prio0_in_overlap:
        job_set = {str(j) for j in jobs}
        for ev in prio0:
            pay = _payload(ev)
            if str(pay.get("job_id") or "") in job_set or str(pay.get("unit_id") or "") in job_set:
                prio0_in_overlap = True
                break
    # Confirmed-without-prio0-start is not enough: the defect was they never started.
    # Overlap of two short jobs while the long job sat at the back is the same miss.
    if not prio0:
        overlap_met = False
        overlap_detail = (
            "no priority-0 detached job STARTED (live pid + started_at on a "
            "long/composer-detached unit); the falsy-zero defect sorted those "
            "last and they never ran. A sort-is-fixed assertion is not this."
        )
    elif not ok:
        overlap_met = False
        overlap_detail = overlap_eval.get("detail") or (
            "no overlapping detached_started intervals (need two jobs running at once)"
        )
    elif not prio0_in_overlap:
        overlap_met = False
        overlap_detail = (
            f"priority-0 job(s) started ({prio0_ids}) but were not in an "
            f"overlapping pair {jobs}; overlap without the long job running "
            "is the same miss as ranking it last"
        )
    else:
        overlap_met = True
        overlap_detail = overlap_eval.get("detail") or f"detached jobs overlapped: {jobs}"
    quote_src = prio0[0] if prio0 else (starts[0] if starts else None)
    overlap_quote = None
    if quote_src is not None:
        pay = _payload(quote_src)
        overlap_quote = _quote_event(
            quote_src,
            observation=(
                f"pid={pay.get('pid')} started_at={pay.get('started_at')} "
                f"job_id={pay.get('job_id')} unit_id={pay.get('unit_id')} "
                f"capability={pay.get('capability')} overlapped={jobs} "
                f"confirmed={bool(confirmed)}"
            ),
        )
    rows["overlap_detached_work"] = {
        "id": "overlap_detached_work",
        "met": bool(overlap_met),
        "cites": overlap_eval.get("cites") or [],
        "detail": overlap_detail,
        "event_seqs": overlap_eval.get("event_seqs") or [],
        "jobs": jobs,
        "priority_zero_started": [str(_payload(e).get("job_id") or _payload(e).get("unit_id")) for e in prio0],
        "priority_zero_in_overlap": prio0_in_overlap,
        "quote": overlap_quote,
        "staged_cannot_satisfy": True,
    }

    # use_negative_science + real index + campaign scars reachable
    ns_eval = eval_use_negative_science(view)
    scars = campaign_science_scars_reachable()
    ns_hits = [
        e
        for e in view.of("negative_science_query", "negative_science_refusal")
        if not event_is_staged(e)
    ]
    cites_index = False
    cited_sources: list[str] = []
    for ev in ns_hits:
        pay = _payload(ev)
        src = str(pay.get("source_path") or "")
        for c in _cites(ev) + ([src] if src else []):
            cited_sources.append(c)
            if "NEGATIVE_SCIENCE" in c or c.startswith("receipts/future/MLP_") or c.startswith("receipts/future/CAMPAIGN_SCARS"):
                cites_index = True
    ns_met = bool(ns_eval.get("met")) and scars.get("ok") is True and (cites_index or bool(ns_hits))
    if not scars.get("ok"):
        ns_met = False
        ns_detail = (
            "campaign science scars are not reachable in the live index: "
            + ", ".join(scars.get("missing") or [])
            + " — a run that uses negative science without these being "
            "reachable is not using it (6fc77f169)"
        )
    elif not ns_hits:
        ns_met = False
        ns_detail = ns_eval.get("detail") or "no negative-science event on the sealed timeline"
    else:
        ns_detail = (
            f"consulted the real index; {scars.get('n_reachable')}/{scars.get('n_named')} "
            f"campaign science scars reachable; {len(ns_hits)} query/refusal event(s)"
        )
        ns_met = True
    ns_quote = None
    if ns_hits:
        ev = ns_hits[0]
        pay = _payload(ev)
        ns_quote = _quote_event(
            ev,
            observation=(
                f"kind={ev.get('kind')} query={pay.get('query')} "
                f"source_path={pay.get('source_path')} scar_id={pay.get('scar_id')} "
                f"reachable_scars={sorted(k for k, v in (scars.get('reachable') or {}).items() if v.get('reachable'))}"
            ),
        )
    rows["use_negative_science"] = {
        "id": "use_negative_science",
        "met": bool(ns_met),
        "cites": ns_eval.get("cites") or [],
        "detail": ns_detail,
        "event_seqs": ns_eval.get("event_seqs") or [],
        "campaign_science_scars": scars,
        "cited_sources": cited_sources[:12],
        "quote": ns_quote,
        "staged_cannot_satisfy": True,
    }

    # alter_priority_from_evidence
    prio_eval = eval_alter_priority_from_evidence(view)
    prio_hits = [
        e
        for e in view.of("priority_altered")
        if not event_is_staged(e)
        and isinstance(_payload(e).get("before"), list)
        and isinstance(_payload(e).get("after"), list)
        and _payload(e).get("before") != _payload(e).get("after")
        and _cites(e)
    ]
    prio_quote = None
    if prio_hits:
        ev = prio_hits[0]
        pay = _payload(ev)
        prio_quote = _quote_event(
            ev,
            observation=(
                f"remaining queue reordered cause={pay.get('cause')!r} "
                f"before[:6]={list(pay.get('before') or [])[:6]} "
                f"after[:6]={list(pay.get('after') or [])[:6]} "
                f"cites={_cites(ev)}"
            ),
        )
    rows["alter_priority_from_evidence"] = {
        **prio_eval,
        "met": bool(prio_eval.get("met")) and bool(prio_hits),
        "quote": prio_quote,
        "staged_cannot_satisfy": True,
    }

    unmet = [cid for cid in FOUR_THIRTY_M if not rows[cid].get("met")]
    return {
        "conditions": rows,
        "unmet": unmet,
        "all_four_met": not unmet,
        "staged_event_cannot_satisfy": True,
        "judged_from": "sealed_timeline_events",
        "ignored": ["runner_summary", "summary", "self_report", "self_verdict"],
    }


def _pid_alive(pid: Any) -> bool | None:
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        _os.kill(pid, 0)
        return True
    except OSError:
        return False


def _parse_gpu_lock_file(path: Path) -> dict[str, Any]:
    """Read a JSON lock FILE (this host's current protocol) without deleting it."""
    row: dict[str, Any] = {
        "path": str(path),
        "present": path.exists(),
        "kind": None,
        "owner": None,
        "pid": None,
        "pid_alive": None,
        "parked": False,
        "contended": False,
    }
    if not path.exists():
        row["kind"] = "absent"
        return row
    if path.is_dir():
        row["kind"] = "directory"
        owner_p = path / "owner"
        pid_p = path / "pid"
        if owner_p.is_file():
            try:
                row["owner"] = owner_p.read_text(encoding="utf-8").strip() or None
            except OSError:
                pass
        if pid_p.is_file():
            try:
                text = pid_p.read_text(encoding="utf-8").strip()
                row["pid"] = int(text) if text.isdigit() else text
            except OSError:
                pass
        row["pid_alive"] = _pid_alive(row["pid"])
        row["parked"] = True
        return row
    if path.is_file():
        row["kind"] = "file"
        try:
            row["bytes"] = path.stat().st_size
        except OSError:
            row["bytes"] = None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raw = None
        if isinstance(raw, Mapping):
            row["owner"] = raw.get("holder") or raw.get("owner")
            pid = raw.get("pid")
            row["pid"] = int(pid) if isinstance(pid, int) else pid
            row["acquired_at"] = raw.get("acquired_at")
            row["holder_parked"] = raw.get("parked")
        row["pid_alive"] = _pid_alive(row["pid"])
        row["parked"] = True
        return row
    row["kind"] = "other"
    row["parked"] = True
    return row


def park_gpu_lane_lock(*, timeout_s: float = 25 * 60) -> dict[str, Any]:
    """Park on the GPU lane lock. Never flock, never delete, never fabricate a pid.

    A live holder is waited out. A dead or absent lock is recorded and we proceed
    (this trial parks GPU work SLEEPING and runs CPU orchestration). Contending
    is the failure mode.
    """
    t0 = time.time()
    row = _parse_gpu_lock_file(GPU_LANE_LOCK)
    try:
        from tools.future import power_torture as pt

        inspected = pt.inspect_gpu_lane_lock(GPU_LANE_LOCK)
        row["inspect_gpu_lane_lock"] = {
            k: inspected.get(k)
            for k in ("path", "present", "kind", "owner", "pid", "pid_alive", "waited_for", "parked", "contended")
        }
    except Exception as exc:
        row["inspect_gpu_lane_lock"] = {"error": f"{type(exc).__name__}: {exc}"}
    pid = row.get("pid")
    alive = _pid_alive(pid)
    row["pid_alive"] = alive
    row["contended"] = False
    if alive is True and isinstance(pid, int):
        row["parked"] = True
        row["waited_for"] = (
            f"live GPU-lane holder pid={pid} owner={row.get('owner')!r}; "
            "parking until it exits rather than contending"
        )
        deadline = t0 + float(timeout_s)
        while time.time() < deadline:
            if _pid_alive(pid) is not True:
                break
            if not GPU_LANE_LOCK.exists():
                break
            time.sleep(2.0)
        row["pid_alive_after_wait"] = _pid_alive(pid)
        row["present_after_wait"] = GPU_LANE_LOCK.exists()
    else:
        row["waited_for"] = (
            None
            if not row.get("present")
            else (
                f"lock present kind={row.get('kind')} pid={pid} alive={alive}; "
                "not a live holder, not contended"
            )
        )
    row["waited_s"] = round(time.time() - t0, 3)
    row["parked"] = True
    row["gpu_authority"] = False
    return row


def hash_trial_substrate() -> dict[str, Any]:
    """Same freeze POWER_TORTURE_30M used: power_torture.hash_substrate()."""
    from tools.future.power_torture import hash_substrate

    return hash_substrate()


def verify_trial_substrate(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    from tools.future.power_torture import verify_substrate

    return verify_substrate(before, after)


def run_instruments_on_timeline(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Degeneracy measure + no-wait classifier over THIS run's timeline.

    The 1h trial failed both. A 30m trial that repeats either fails, and it
    must not exempt itself from the instruments that caught its predecessor.
    """
    from tools.future import autonomy_degeneracy as ad
    from tools.future import no_wait_orchestration as nwo

    deg = ad.measure(source)
    nowait = nwo.classify(source)
    deg_verdict = str(deg.get("verdict") or "")
    nowait_verdict = str(nowait.get("verdict") or "")
    return {
        "degeneracy": {
            "verdict": deg_verdict,
            "reason": deg.get("reason"),
            "degenerate_axes": list(deg.get("degenerate_axes") or []),
            "elapsed_s": deg.get("elapsed_s"),
            "n_events": deg.get("n_events"),
            "instrument": "tools.future.autonomy_degeneracy.measure",
        },
        "no_wait": {
            "verdict": nowait_verdict,
            "reason": nowait.get("reason"),
            "n_forcing_intervals": nowait.get("n_forcing_intervals"),
            "n_slow_intervals": nowait.get("n_slow_intervals"),
            "n_blocked_intervals": nowait.get("n_blocked_intervals"),
            "instrument": "tools.future.no_wait_orchestration.classify",
        },
        "either_fails": (
            deg_verdict == "FAIL"
            or nowait_verdict == "FAIL_NO_WAIT_ORCHESTRATION"
        ),
        "exempted": False,
    }


def freeze_and_run_30m(
    *,
    timeline_path: Path | None = None,
    duration_s: int | None = None,
    park_timeout_s: float = 25 * 60,
) -> dict[str, Any]:
    """Freeze the substrate, park the GPU lock, run 30 real minutes, hash after.

    Does not judge. Judging reads the sealed timeline in a separate step so
    the driver cannot grade itself.
    """
    from tools.future import autonomy_run as ar

    dest = Path(timeline_path) if timeline_path else (RECEIPTS / "AUTONOMY_TIMELINE_30m.json")
    duration = int(duration_s) if duration_s is not None else int(TRIAL_DURATION_S["30m"])
    gpu = park_gpu_lane_lock(timeout_s=park_timeout_s)
    before = hash_trial_substrate()
    started = time.time()
    result = ar.run("30m", dest, duration)
    wall_s = time.time() - started
    after = hash_trial_substrate()
    substrate = verify_trial_substrate(before, after)
    if dest.is_file():
        digest = sha256_file(dest)
    else:
        digest = None
    return {
        "trial": "30m",
        "timeline_path": _repo_rel(dest) if dest.is_file() else str(dest),
        "duration_s_budget": duration,
        "elapsed_s": result.get("elapsed_s") if isinstance(result, Mapping) else None,
        "wall_s": round(wall_s, 3),
        "driver_summary": (
            {
                k: result.get(k)
                for k in (
                    "launched",
                    "receipts_ingested",
                    "refused_on_evidence",
                    "scars_consulted",
                    "composed_replan_pairs",
                    "hypotheses_proposed",
                    "hypotheses_still_live",
                    "blocked_lanes_parked",
                    "resident_model_cognition",
                    "elapsed_s",
                    "timeline",
                )
                if isinstance(result, Mapping) and k in result
            }
            if isinstance(result, Mapping)
            else None
        ),
        "gpu_lane_lock": gpu,
        "substrate": {
            "before_digest": before.get("digest"),
            "after_digest": after.get("digest"),
            "equal": substrate.get("equal"),
            "verdict": substrate.get("verdict"),
            "moved": substrate.get("moved") or [],
            "n_files": before.get("n_files"),
            "n_shaders": before.get("n_shaders"),
            "why": substrate.get("why"),
        },
        "timeline_seal_digest": digest,
        "judged": False,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def judge_frozen_30m(
    timeline_path: Path | str,
    *,
    run_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge the sealed 30m timeline. Separate from freeze_and_run_30m."""
    path = Path(timeline_path)
    doc = load_timeline(path)
    verdict = verify("30m", path)
    four = judge_four_from_sealed(doc)
    instruments = run_instruments_on_timeline(path)
    elapsed = int(verdict.get("elapsed_s") or 0)
    duration = int(verdict.get("duration_s") or TRIAL_DURATION_S["30m"])
    met_ids = [c["id"] for c in verdict.get("conditions") or [] if c.get("met")]
    sixteen_unmet = [cid for cid in SIXTEEN_THIRTY_M if cid not in met_ids]
    four_unmet = list(four.get("unmet") or [])
    substrate = dict((run_record or {}).get("substrate") or {})
    substrate_ok = substrate.get("equal") is True
    if substrate.get("verdict") == "INVALIDATED_BY_SUBSTRATE_MUTATION":
        substrate_ok = False
    elapsed_ok = elapsed >= duration
    staged_used = any(
        (row.get("quote") or {}).get("staged") is True
        for row in (four.get("conditions") or {}).values()
        if isinstance(row, Mapping)
    )
    sixteen_of_sixteen = (
        not sixteen_unmet
        and elapsed_ok
        and substrate_ok
        and not staged_used
        and four.get("all_four_met") is True
    )
    persist = persist_verdict(verdict, path, timeline_doc=doc)
    no_idle_met = "no_idle_while_work_exists" in met_ids
    instrument_fail = bool(instruments.get("either_fails"))
    bits: list[str] = []
    if sixteen_of_sixteen:
        bits.append("16/16")
    else:
        bits.append(
            "unmet: "
            + ", ".join(
                sixteen_unmet
                + (["elapsed<30m"] if not elapsed_ok else [])
                + (["substrate_moved"] if not substrate_ok else [])
                + (["staged_event"] if staged_used else [])
                + ([f"four:{u}" for u in four_unmet if u not in sixteen_unmet])
            )
        )
    if not no_idle_met:
        bits.append("no_idle_while_work_exists unmet")
    if instrument_fail:
        bits.append(
            "instruments FAIL "
            f"degeneracy={instruments.get('degeneracy', {}).get('verdict')} "
            f"no_wait={instruments.get('no_wait', {}).get('verdict')}"
        )
    overall = (
        "PASS"
        if sixteen_of_sixteen and no_idle_met and not instrument_fail and not staged_used
        else "FAIL"
    )
    record = {
        "schema": "hawking.future.autonomy_trial.frozen_30m.v1",
        "trial": "30m",
        "elapsed_s": elapsed,
        "duration_s": duration,
        "elapsed_meets_duration": elapsed_ok,
        "timer_is_not_a_pass": True,
        "substrate": substrate,
        "gpu_lane_lock": (run_record or {}).get("gpu_lane_lock"),
        "timeline_path": _repo_rel(path),
        "timeline_seal_digest": (run_record or {}).get("timeline_seal_digest")
        or persist["record"].get("timeline_seal_digest"),
        "verify": {
            "verdict": verdict.get("verdict"),
            "reason": verdict.get("reason"),
            "unmet": verdict.get("unmet"),
            "n_conditions": len(verdict.get("conditions") or []),
            "n_met": len(met_ids),
            "automatic_failures": [f.get("id") for f in (verdict.get("automatic_failures") or [])],
        },
        "sixteen": {
            "required": list(SIXTEEN_THIRTY_M),
            "unmet": sixteen_unmet,
            "n_met": len(SIXTEEN_THIRTY_M) - len(sixteen_unmet),
            "sixteen_of_sixteen": sixteen_of_sixteen,
        },
        "four": four,
        "instruments": instruments,
        "staged_event_used": staged_used,
        "no_idle_while_work_exists": no_idle_met,
        "overall": overall,
        "report": "; ".join(bits),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "judged_from": "sealed_timeline",
        "ignored_runner_summary": True,
    }
    existing = load_owned_receipt()
    existing["frozen_30m_run"] = record
    by = dict(existing.get("persisted_verdicts_by_trial") or {})
    by["30m"] = persist["record"]
    existing["persisted_verdicts_by_trial"] = by
    existing["last_persisted_verdict"] = persist["record"]
    existing.setdefault("schema", SCHEMA)
    existing.setdefault("version", VERSION)
    existing["evidence_class"] = "STATIC_ONLY"
    existing["gpu_authority"] = False
    write_receipt(RECEIPT, existing, RECORDED_BY)
    return record


def run_frozen_30m_trial(
    *,
    timeline_path: Path | None = None,
    duration_s: int | None = None,
    park_timeout_s: float = 25 * 60,
) -> dict[str, Any]:
    """Orchestrate freeze → run → (separate) judge. Driver never grades itself."""
    dest = Path(timeline_path) if timeline_path else (RECEIPTS / "AUTONOMY_TIMELINE_30m.json")
    ran = freeze_and_run_30m(
        timeline_path=dest,
        duration_s=duration_s,
        park_timeout_s=park_timeout_s,
    )
    judged = judge_frozen_30m(dest, run_record=ran)
    return {"run": ran, "judgement": judged}


def next_workunits(frontier: Mapping[str, Any]) -> list[dict[str, str]]:
    safe = remaining_safe_work(frontier)
    rows = [
        {
            "id": "future.autonomy.record-15m-from-live-frontier",
            "does": "Snapshot the live frontier and record a 15m timeline; verify in a second process.",
        },
        {
            "id": "future.autonomy.keep-cpu-fronts-moving",
            "does": (
                "While GPU lanes sleep, launch CPU-safe units for remaining HIGH_VALUE_INTEGRATION / "
                "MISSING / WEAK fronts: "
                + ", ".join(str(r.get("id") or "") for r in safe if r.get("id"))
            ),
        },
        {
            "id": "future.autonomy.park-not-synthesize",
            "does": "Keep Flash NX / teacher-capture / protected-lease work SLEEPING; wakeup.py is the swap.",
        },
    ]
    return rows


def build() -> Path:
    prior = load_owned_receipt()
    persisted_by = dict(prior.get("persisted_verdicts_by_trial") or {})
    last_persisted = prior.get("last_persisted_verdict")
    frozen_30m = prior.get("frozen_30m_run")
    proofs = prove_negative_controls()
    passing = prove_passing_timelines()
    units = emit_trial_workunits()
    frontier = load_frontier()
    blockers = load_hardware_blockers()
    safe = remaining_safe_work(frontier)
    by_class: dict[str, int] = {}
    for row in frontier.get("entries") or []:
        if isinstance(row, Mapping):
            key = str(row.get("classification") or "UNKNOWN")
            by_class[key] = by_class.get(key, 0) + 1
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Harness that runs and judges progressive autonomy trials (15m, 1h, 3h, 6h) "
            "from recorded timelines. A timer is not a pass."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "fpga": (
            "FPGA belongs to Accelerator / Physical Compiler / Fusion. "
            "This harness does not build an FPGA backend."
        ),
        "trials": {
            trial: {
                "duration_s": TRIAL_DURATION_S[trial],
                "required_conditions": list(REQUIRED_CONDITIONS[trial]),
            }
            for trial in TRIAL_IDS
        },
        "thirteen_acceptance_conditions": list(THIRTEEN_ACCEPTANCE),
        "automatic_failures": list(AUTO_FAIL_IDS),
        "timer_is_not_a_pass": True,
        "record_verify_split": (
            "--record writes a timeline and never judges it; --verify reads a timeline "
            "and never records passing events onto it. Combined invocation is FailClosed. "
            "--verify DOES persist the judgement into AUTONOMY_TRIALS.json, which is this "
            "module's receipt, not the timeline."
        ),
        "verify_persists_verdict": True,
        "receipt_name": RECEIPT,
        "frozen_30m_run": frozen_30m,
        "persisted_verdicts_by_trial": persisted_by,
        "last_persisted_verdict": last_persisted,
        "frontier_census": {
            "path_taken": frontier.get("path_taken"),
            "present": frontier.get("present"),
            "n_entries": len(list(frontier.get("entries") or [])),
            "n_resolved": len(list(frontier.get("resolved_entries") or [])),
            "n_stale": len(list(frontier.get("stale_entries") or [])),
            "n_live": len(live_frontier_entries(frontier)),
            "n_safe_cpu": len(safe),
            "safe_cpu_ids": [str(r.get("id") or "") for r in safe if r.get("id")],
            "by_classification": {k: by_class[k] for k in sorted(by_class)},
        },
        "hardware_blockers": blockers,
        "work_units": units,
        "negative_controls_watched": proofs,
        "passing_fixtures": passing,
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "resident_callable": resident_callable(units),
        "integration_points": INTEGRATION_POINTS,
        "next_workunits": next_workunits(frontier),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "vocabulary": {
            "measurement_classes": (
                "DIAGNOSTIC_RELATIVE guides and never promotes; "
                "PROTECTED_ABSOLUTE decides; this harness produces neither."
            ),
            "disk_state_is_authority": True,
            "models_think_tools_know_context_is_a_cache": True,
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def _print_verdict(verdict: Mapping[str, Any]) -> None:
    print(
        f"verdict={verdict['verdict']} trial={verdict['trial']} "
        f"elapsed_s={verdict['elapsed_s']} duration_s={verdict['duration_s']} "
        f"elapsed_is_not_a_pass={verdict['elapsed_is_not_a_pass']}"
    )
    print(f"reason={verdict['reason']}")
    if verdict.get("automatic_failures"):
        print("automatic_failures:")
        for item in verdict["automatic_failures"]:
            print(f"  {item['id']}: {item['detail']}")
    unmet = [c for c in verdict.get("conditions") or [] if not c.get("met")]
    if unmet:
        print("unmet_conditions:")
        for item in unmet:
            print(f"  {item['id']}: {item['detail']}")
    met = [c for c in verdict.get("conditions") or [] if c.get("met")]
    if met:
        print("met_conditions:")
        for item in met:
            cites = ", ".join(item.get("cites") or [])
            print(f"  {item['id']}: {item['detail']} cites=[{cites}]")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", metavar="TRIAL", choices=list(TRIAL_IDS))
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--run-frozen", metavar="TRIAL", choices=["30m"])
    ap.add_argument("--trial", choices=list(TRIAL_IDS))
    ap.add_argument("--timeline", type=Path)
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--event", metavar="JSON", help="event object to append while recording")
    ap.add_argument("--t", dest="t_s", type=int, help="logical t_s for --event")
    args = ap.parse_args(list(argv) if argv is not None else None)

    try:
        if args.record and args.verify:
            _fail("self_grade", "judging is separate from running; do not pass --record and --verify together")
        if args.run_frozen and (args.record or args.verify):
            _fail(
                "self_grade",
                "--run-frozen already sequences freeze+run then a separate judge; "
                "do not combine it with --record or --verify",
            )
        if args.run_frozen:
            dest = args.timeline or (RECEIPTS / "AUTONOMY_TIMELINE_30m.json")
            out = run_frozen_30m_trial(timeline_path=Path(dest))
            print(json.dumps(out["judgement"], indent=1, sort_keys=True, default=str))
            overall = str((out.get("judgement") or {}).get("overall") or "FAIL")
            return 0 if overall == "PASS" else 1
        if args.selftest:
            out = selftest()
            print(out)
            return 0
        if args.verify:
            if args.timeline is None:
                _fail("missing_timeline", "--verify requires --timeline")
            verdict = verify(args.verify, args.timeline)
            persisted = persist_verdict(verdict, args.timeline)
            _print_verdict(verdict)
            print(
                f"persisted={persisted['path']} "
                f"trial={persisted['record']['trial']} "
                f"verdict={persisted['record']['verdict']} "
                f"timeline_seal_digest={persisted['record']['timeline_seal_digest']} "
                f"resident_orchestration={persisted['record']['resident_orchestration']} "
                f"resident_model_cognition={persisted['record']['resident_model_cognition']}"
            )
            return 0 if verdict["verdict"] == "PASS" else 1
        if args.record:
            trial = args.trial
            if trial is None:
                _fail("unknown_trial", "--record requires --trial")
            if args.timeline is None:
                _fail("missing_timeline", "--record requires --timeline")
            event = None
            if args.event is not None:
                try:
                    loaded = json.loads(args.event)
                except json.JSONDecodeError as exc:
                    _fail("malformed_event", str(exc))
                if not isinstance(loaded, dict):
                    _fail("malformed_event", "--event must be a JSON object")
                event = loaded
            out = record(trial, args.timeline, init=args.init, event=event, t_s=args.t_s)
            print(out)
            return 0
        out = build()
        print(out)
        return 0
    except FailClosed as exc:
        print(f"FAIL_CLOSED [{exc.fault}]: {exc.reason}", file=_sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
