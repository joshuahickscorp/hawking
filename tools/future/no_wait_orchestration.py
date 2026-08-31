#!/usr/bin/env python3
"""NO-WAIT ORCHESTRATION IS AN EXPLICIT FAILURE TARGET.

If runnable safe work exists while the whole HCLI loop reports waiting on a
subprocess, that is FAIL_NO_WAIT_ORCHESTRATION, NOT A SLOW RESIDENT.

A resident that is slow is a performance problem. A resident that BLOCKS
while safe runnable work sits available is an architecture failure. The two
must never be reported the same way: the first invites patience, the second
invites a fix.

This module is the verdict. It does not reimplement the scheduler, the
detached trial, or the degeneracy measure. It composes them:

    no_wait_scheduler.runnable_now   independence from graph/frontier
    detached.refuse_reason           GPU/lease/cargo units SLEEP, not run
    autonomy_degeneracy._events_of   one loader for path / receipt / TrialRecord
    improvement_trial.judge          wrapped through this module's public API
                                     (the trial file is not edited)

The classifier returns exactly one of:

    PASS                       no interval where runnable safe work existed
                               while the loop was blocked
    FAIL_NO_WAIT_ORCHESTRATION runnable safe work existed and the loop waited
    SLOW_BUT_CORRECT           the loop was slow with NOTHING runnable — not
                               a failure of this obligation

    python3 tools/future/no_wait_orchestration.py --record
    python3 tools/future/no_wait_orchestration.py --selftest
    python3 -m pytest tools/future/test_no_wait_orchestration.py -q
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future import autonomy_degeneracy as ad
from tools.future import no_wait_scheduler as nws
from tools.future._common import (
    REPO,
    git,
    sha256_file,
    write_receipt,
    _assert_no_hardware_claims,
)
from tools.future.detached import GPU_RESOURCE_CLASSES, refuse_reason


RECEIPT = "NO_WAIT_ORCHESTRATION.json"
SCHEMA = "hawking.future.no_wait_orchestration.v1"
VERSION = 1
RECORDED_BY = "tools/future/no_wait_orchestration.py"

TIMELINE_1H_REL = "receipts/future/AUTONOMY_TIMELINE_1h.json"
DETACHED_TRIAL_REL = "receipts/future/DETACHED_WORK_TRIAL.json"
POWER_TORTURE_TIMELINE_REL = "receipts/future/POWER_TORTURE_TIMELINE.json"

# The obligation names the 1h timeline's last 2864 seconds as the calibration.
TAIL_S = 2864.0

PASS = "PASS"
FAIL_NO_WAIT_ORCHESTRATION = "FAIL_NO_WAIT_ORCHESTRATION"
SLOW_BUT_CORRECT = "SLOW_BUT_CORRECT"
VERDICTS = (PASS, FAIL_NO_WAIT_ORCHESTRATION, SLOW_BUT_CORRECT)

# Hardware resource classes that cannot start without a lease / qualification
# this sidecar does not hold. STATIC_ANALYSIS of a GPU kernel is not this:
# it is CPU work. FPGA.engine-sim launched as STATIC_ANALYSIS is not this.
HARDWARE_PARKED_CLASSES = frozenset(GPU_RESOURCE_CLASSES) | {
    "GPU_PROTECTED",
    "ANE",
    "FPGA",
}

# CPU-safe classes whose resource is free on this host (no GPU lease).
FREE_RESOURCE_CLASSES = frozenset(
    {
        "STATIC_ANALYSIS",
        "LIGHT_CONTROL",
        "CPU_HEAVY",
        "COMPILE",
        "TEST",
        "TEST_AUTHORING",
        "MEMORY_HEAVY",
        "IO_HEAVY",
        "TOOL_WAIT",
        "GROK",
        "MUTATION",
    }
)

# Synchronous launches — the autonomy loop's blocking invoke (subprocess.run).
# Detached kinds (CHILD_LAUNCHED, INDEPENDENT_STARTED) mean the loop released.
SYNC_LAUNCH_KINDS = frozenset({"workunit_launched", "WORK_LAUNCHED"})
DETACHED_LAUNCH_KINDS = frozenset({"CHILD_LAUNCHED", "INDEPENDENT_STARTED"})
LAUNCH_KINDS = SYNC_LAUNCH_KINDS | DETACHED_LAUNCH_KINDS
COMPLETE_KINDS = frozenset(
    {
        "result_ingested",
        "RESULT_INGESTED",
        "process_failed",
        "INDEPENDENT_COMPLETED",
        "CHILD_TERMINAL",
    }
)
WAIT_KINDS = frozenset({"HANDLE_WAIT", "handle_wait"})
SLEEP_KINDS = frozenset({"workunit_sleeping", "WORKUNIT_SLEEPING"})
QUEUE_KINDS = frozenset(
    {
        "next_work_left",
        "WORK_REFILLED",
        "work_refilled",
        "FRONTIER_HAS_WORK",
        "OPTIONS_RANKED",
        "frontier_identified",
    }
)
REFILL_KINDS = frozenset({"work_refilled", "WORK_REFILLED"})
DECISION_KINDS = frozenset(
    {
        "idea_rejected",
        "IDEA_REJECTED",
        "NEXT_DECISION",
        "OPTIONS_RANKED",
        "BRANCH_KILLED",
        "branch_killed",
        "FALSIFIER_GENERATED",
        "negative_science_refusal",
        "NEGATIVE_SCIENCE_REFUSAL",
    }
)

RUNNABLE_SAFE_WORK_RULE = (
    "RUNNABLE SAFE WORK is a unit that can start NOW without waiting on the "
    "blocked loop, without seizing a GPU/bench lease this process does not "
    "hold, and without a wake condition that is still false. A unit is "
    "runnable safe work when every clause holds: (1) IDENTITY — it has an id "
    "and is not already completed, failed, cancelled, or in-flight. "
    "(2) DEPENDENCIES — every dependency id is in the completed set; a "
    "dependency that is the in-flight blocked unit is unmet. (3) RESOURCE — "
    "GPU_DECODE / GPU_EXCLUSIVE / GPU_DIRTY_OK require a held GPU lease; "
    "GPU_PROTECTED / ANE / FPGA as a hardware class require qualification "
    "this sidecar does not have. STATIC_ANALYSIS (including static analysis "
    "of GPU kernels, and FPGA.engine-sim launched as STATIC_ANALYSIS) is CPU "
    "work and is runnable when the other clauses hold. LIGHT_CONTROL, "
    "CPU_HEAVY, COMPILE, TEST, IO_HEAVY, MEMORY_HEAVY, TOOL_WAIT, "
    "TEST_AUTHORING, GROK are free on this host. (4) SLEEP — a unit in "
    "SLEEPING state, or a workunit_sleeping row for its resource_class, "
    "whose wake_condition is unsatisfied is not runnable. The same unit is "
    "runnable only if that wake_condition is recorded satisfied. (5) SAFETY "
    "— if the unit carries a command, detached.refuse_reason is applied; a "
    "GPU argv, flock, cargo build, or lease seizure is not runnable. "
    "A unit blocked on a GPU lease is not runnable. A unit whose resource "
    "is free and whose dependencies are met IS runnable. A sleeping unit "
    "whose wake condition is unsatisfied is not."
)

CLAIM_BOUNDARY = (
    "STATIC_ONLY sidecar. No GPU lease and no hardware measurement. "
    "Intervals are event-log t_s cardinalities from receipts already on disk. "
    "Cited detached idle_runnable_seconds and kqueue latency are copied as "
    "strings from DETACHED_WORK_TRIAL.json; they are not re-measured here. "
    "FAIL_NO_WAIT_ORCHESTRATION is an architecture failure; SLOW_BUT_CORRECT "
    "is not. A trial that would PASS while FAIL_NO_WAIT_ORCHESTRATION holds "
    "cannot PASS through this module's public API."
)

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

_ORIGINAL_JUDGE: Any = None


# ---------------------------------------------------------------------------
# World + runnable-safe-work rule
# ---------------------------------------------------------------------------


@dataclass
class World:
    """Resource and dependency state against which a unit is classified.

    This sidecar does not hold a GPU lease. Tests may set lease_held=True to
    prove the clause in the other direction; that is a synthetic world, not
    a hardware claim.
    """

    lease_held: bool = False
    completed_ids: frozenset[str] = field(default_factory=frozenset)
    in_flight_ids: frozenset[str] = field(default_factory=frozenset)
    sleeping_resources: dict[str, str] = field(default_factory=dict)
    sleeping_units: dict[str, str] = field(default_factory=dict)
    satisfied_wake_conditions: frozenset[str] = field(default_factory=frozenset)


def _as_unit(unit: Any) -> dict[str, Any]:
    if unit is None:
        return {}
    if isinstance(unit, Mapping):
        return dict(unit)
    return {"id": str(unit)}


def _unit_id_of(unit: Mapping[str, Any]) -> str:
    for key in ("id", "unit_id", "workunit_id", "job_id"):
        token = str(unit.get(key) or "").strip()
        if token:
            return token
    return ""


def _deps_of(unit: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for key in ("dependencies", "verification_depends_on", "depends_on"):
        raw = unit.get(key) or []
        if isinstance(raw, (str, bytes)):
            text = str(raw).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
            continue
        if not isinstance(raw, (list, tuple)):
            continue
        for item in raw:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def _resource_class_of(unit: Mapping[str, Any]) -> str:
    return str(unit.get("resource_class") or "").strip().upper()


def is_runnable_safe_work(unit: Any, world: World | None = None) -> dict[str, Any]:
    """Apply the runnable-safe-work rule. Returns runnable plus the clause that fired.

    The whole verdict turns on this. Each clause is named so a test can pin it.
    """
    world = world or World()
    row = _as_unit(unit)
    uid = _unit_id_of(row)
    rc = _resource_class_of(row)
    deps = _deps_of(row)
    status = str(row.get("status") or row.get("state") or "").strip().upper()
    wake = str(row.get("wake_condition") or "").strip()
    clauses: dict[str, Any] = {
        "id": uid or None,
        "resource_class": rc or None,
        "dependencies": deps,
        "lease_held": bool(world.lease_held),
    }

    if not uid:
        return {
            "runnable": False,
            "reason": "IDENTITY: unit has no id",
            "clause": "IDENTITY",
            **clauses,
        }
    if uid in world.completed_ids:
        return {
            "runnable": False,
            "reason": f"IDENTITY: {uid} is already completed",
            "clause": "IDENTITY",
            **clauses,
        }
    if uid in world.in_flight_ids:
        return {
            "runnable": False,
            "reason": f"IDENTITY: {uid} is already in-flight",
            "clause": "IDENTITY",
            **clauses,
        }

    unmet = [d for d in deps if d not in world.completed_ids]
    if unmet:
        return {
            "runnable": False,
            "reason": f"DEPENDENCIES: unmet {unmet}",
            "clause": "DEPENDENCIES",
            "unmet_dependencies": unmet,
            **clauses,
        }

    if rc in GPU_RESOURCE_CLASSES and not world.lease_held:
        return {
            "runnable": False,
            "reason": (
                f"RESOURCE: {rc} requires a GPU lease this sidecar does not hold; "
                f"{uid} is not runnable"
            ),
            "clause": "RESOURCE",
            **clauses,
        }
    if rc in HARDWARE_PARKED_CLASSES and not world.lease_held:
        return {
            "runnable": False,
            "reason": (
                f"RESOURCE: {rc} is a parked hardware class; wake unsatisfied "
                f"and no lease is held"
            ),
            "clause": "RESOURCE",
            **clauses,
        }

    sleeping_wake = world.sleeping_units.get(uid) or world.sleeping_resources.get(rc)
    if status in {"SLEEPING", "SLEEP"}:
        sleeping_wake = sleeping_wake or wake or "unspecified wake_condition"
    if sleeping_wake:
        if sleeping_wake in world.satisfied_wake_conditions:
            pass
        else:
            return {
                "runnable": False,
                "reason": (
                    f"SLEEP: {uid} is sleeping; wake_condition "
                    f"{sleeping_wake!r} is unsatisfied"
                ),
                "clause": "SLEEP",
                "wake_condition": sleeping_wake,
                **clauses,
            }

    command = row.get("command")
    if command:
        argv = [str(x) for x in command] if isinstance(command, (list, tuple)) else [str(command)]
        refused = refuse_reason(argv, resource_class=rc or None)
        if refused:
            return {
                "runnable": False,
                "reason": f"SAFETY: {refused}",
                "clause": "SAFETY",
                **clauses,
            }

    if rc and rc not in FREE_RESOURCE_CLASSES and rc not in HARDWARE_PARKED_CLASSES:
        # Unknown class: do not invent a lease. Treat as not-free unless the
        # unit is already CPU-classified by a known free class.
        return {
            "runnable": False,
            "reason": f"RESOURCE: resource_class {rc!r} is not a free class on this host",
            "clause": "RESOURCE",
            **clauses,
        }

    return {
        "runnable": True,
        "reason": (
            f"resource {rc or 'STATIC_ANALYSIS'} is free and dependencies are met"
        ),
        "clause": "RUNNABLE",
        **clauses,
    }


def independent_runnable_now(
    blocked: Sequence[Any],
    *,
    candidates: Sequence[Mapping[str, Any]],
    world: World | None = None,
) -> dict[str, Any]:
    """Compose scheduler independence with the runnable-safe-work rule.

    Graph-independent GPU work is still not runnable-safe without a lease.
    """
    world = world or World()
    view = nws.runnable_now(blocked, candidates=candidates)
    safe: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in view.get("runnable") or []:
        row = dict(raw) if isinstance(raw, Mapping) else {"id": raw}
        verdict = is_runnable_safe_work(row, world)
        if verdict["runnable"]:
            safe.append(row)
        else:
            rejected.append({"id": _unit_id_of(row), "reason": verdict["reason"]})
    status = view.get("status")
    if safe:
        composed = nws.RUNNABLE
    elif status == nws.BLOCKED and not safe:
        composed = nws.BLOCKED
    elif status == nws.RUNNABLE and not safe:
        # Independent on the graph, but every candidate failed the safe-work rule.
        composed = nws.IDLE
    else:
        composed = status
    out = dict(view)
    out["status"] = composed
    out["runnable"] = safe
    out["rejected_unsafe"] = rejected
    out["scheduler_status_before_safe_filter"] = status
    return out


# ---------------------------------------------------------------------------
# Event loading (composes autonomy_degeneracy._events_of)
# ---------------------------------------------------------------------------


def _kind(event: Mapping[str, Any]) -> str:
    return str(event.get("kind") or "").strip()


def _payload(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else {}


def _t_s(event: Mapping[str, Any]) -> float:
    try:
        return float(event.get("t_s") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _event_unit(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = _payload(event)
    unit = payload.get("unit")
    if isinstance(unit, Mapping):
        row = dict(unit)
    else:
        row = {}
    for key in ("id", "unit_id", "workunit_id", "job_id"):
        if not str(row.get(key) or "").strip():
            token = payload.get(key)
            if token:
                row[key] = token
    if not _resource_class_of(row) and payload.get("resource_class"):
        row["resource_class"] = payload.get("resource_class")
    if not row.get("wake_condition") and payload.get("wake_condition"):
        row["wake_condition"] = payload.get("wake_condition")
    if not row.get("frontier_id"):
        fid = payload.get("frontier_id") or payload.get("frontier_item")
        if fid:
            row["frontier_id"] = fid
    return row


def _event_unit_id(event: Mapping[str, Any]) -> str:
    return _unit_id_of(_event_unit(event)) or _unit_id_of(_payload(event))


def _id_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    if value is None or value == "":
        return []
    return [str(value)]


def _queue_ids(event: Mapping[str, Any]) -> tuple[list[str], int]:
    payload = _payload(event)
    ids = _id_list(
        payload.get("unit_ids")
        or payload.get("ids")
        or payload.get("entry_ids")
        or payload.get("runnable_unit_ids")
        or payload.get("top")
    )
    if not ids:
        top = payload.get("top")
        if isinstance(top, list):
            ids = [
                str(item.get("id") or item.get("unit_id") or "")
                for item in top
                if isinstance(item, Mapping)
            ]
            ids = [x for x in ids if x]
    try:
        n = int(payload.get("n") if payload.get("n") is not None else len(ids))
    except (TypeError, ValueError):
        n = len(ids)
    return ids, n


def _load_events(source: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events, meta = ad._events_of(source)
    return [dict(e) for e in events if isinstance(e, Mapping)], meta


# ---------------------------------------------------------------------------
# Reconstruct world and blocked intervals from a timeline
# ---------------------------------------------------------------------------


def _scan_state(events: Sequence[Mapping[str, Any]], through: int) -> tuple[World, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """World after events[:through+1]. Also returns known units by id and by frontier."""
    completed: set[str] = set()
    inflight: set[str] = set()
    sleeping_resources: dict[str, str] = {}
    sleeping_units: dict[str, str] = {}
    by_id: dict[str, dict[str, Any]] = {}
    by_frontier: dict[str, dict[str, Any]] = {}
    for event in events[: through + 1]:
        kind = _kind(event)
        payload = _payload(event)
        if kind in SLEEP_KINDS:
            rc = str(payload.get("resource_class") or "").strip().upper()
            wake = str(payload.get("wake_condition") or "unspecified wake_condition")
            uid = _event_unit_id(event)
            if uid:
                sleeping_units[uid] = wake
            if rc:
                sleeping_resources[rc] = wake
            continue
        if kind in LAUNCH_KINDS:
            unit = _event_unit(event)
            uid = _unit_id_of(unit)
            if uid:
                by_id[uid] = unit
                inflight.add(uid)
                completed.discard(uid)
            fid = str(unit.get("frontier_id") or "").strip()
            if fid:
                by_frontier[fid] = unit
            continue
        if kind in COMPLETE_KINDS:
            uid = _event_unit_id(event)
            if uid:
                inflight.discard(uid)
                completed.add(uid)
            continue
    world = World(
        lease_held=False,
        completed_ids=frozenset(completed),
        in_flight_ids=frozenset(inflight),
        sleeping_resources=dict(sleeping_resources),
        sleeping_units=dict(sleeping_units),
    )
    return world, by_id, by_frontier


def _known_units(
    events: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _, by_id, by_frontier = _scan_state(events, len(events) - 1)
    # Re-scan launches only so completed units remain known for leftover lookup.
    by_id = {}
    by_frontier = {}
    for event in events:
        if _kind(event) not in LAUNCH_KINDS and _kind(event) not in SLEEP_KINDS:
            continue
        unit = _event_unit(event)
        uid = _unit_id_of(unit)
        if uid:
            by_id[uid] = unit
        fid = str(unit.get("frontier_id") or "").strip()
        if fid:
            by_frontier[fid] = unit
        payload = _payload(event)
        rc = str(payload.get("resource_class") or "").strip().upper()
        if rc and not uid:
            by_frontier.setdefault(rc, {"id": rc, "resource_class": rc, "status": "SLEEPING",
                                       "wake_condition": payload.get("wake_condition")})
    return by_id, by_frontier


def _hydrate_candidate(
    ident: str,
    by_id: Mapping[str, Mapping[str, Any]],
    by_frontier: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if ident in by_id:
        return dict(by_id[ident])
    if ident in by_frontier:
        row = dict(by_frontier[ident])
        # Leftover frontier ids must be classified as themselves, not the last
        # workunit that frontier emitted.
        row["id"] = ident
        row.pop("dependencies", None)
        row["dependencies"] = []
        return row
    # Unknown leftover: this campaign's leftover 1h ids that were never
    # launched (FT.MODEL_CAPABILITY.hard-gates.drive-tools) are tooling, not
    # a parked GPU class.
    rc = "STATIC_ANALYSIS"
    token = ident.upper()
    if token in HARDWARE_PARKED_CLASSES:
        rc = token
    return {"id": ident, "resource_class": rc, "dependencies": []}



def _verified_detached_ids(events: Sequence[Mapping[str, Any]]) -> set[str]:
    """Units whose launch is PROVEN non-blocking, by evidence not by claim.

    The driver emits workunit_launched for BOTH shapes and marks the payload
    launch=detached for one of them. A payload is the driver's own word for it,
    so the claim alone is not enough: this requires a matching detached_started,
    which emit_detached_started refuses to write unless the job has a live pid.

    Why this matters: a detached launch does not block the loop, so it must not
    OPEN a forcing interval, and it DOES count as concurrent work for a sync
    wait that spans it. Treating all 34 of a run's verified detached launches as
    sync launches both invented intervals and hid real overlap.
    """
    claimed: set[str] = set()
    for event in events:
        if _kind(event) not in SYNC_LAUNCH_KINDS:
            continue
        if str(_payload(event).get("launch") or "").lower() != "detached":
            continue
        uid = _event_unit_id(event)
        if uid:
            claimed.add(uid)
    started: set[str] = set()
    for event in events:
        if _kind(event) != "detached_started":
            continue
        uid = _event_unit_id(event)
        if uid:
            started.add(uid)
    return claimed & started


def _sync_launch_between(
    events: Sequence[Mapping[str, Any]],
    i: int,
    j: int,
    *,
    exclude_uid: str,
) -> bool:
    for event in events[i + 1 : j]:
        if _kind(event) not in LAUNCH_KINDS:
            continue
        uid = _event_unit_id(event)
        if uid and uid != exclude_uid:
            return True
    return False


def _blocked_intervals(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Intervals where the whole HCLI loop was waiting on a subprocess.

    HANDLE_WAIT is explicit. A workunit_launched / WORK_LAUNCHED whose matching
    ingest/fail happens later with no other synchronous launch in between is
    the autonomy_run subprocess.run shape. CHILD_LAUNCHED / INDEPENDENT_STARTED
    are detached: the loop released, so they are not blocked intervals.
    """
    intervals: list[dict[str, Any]] = []

    for event in events:
        if _kind(event) not in WAIT_KINDS:
            continue
        payload = _payload(event)
        t0 = _t_s(event)
        try:
            wait_s = float(payload.get("wait_s") or 0.0)
        except (TypeError, ValueError):
            wait_s = 0.0
        if wait_s <= 0.0:
            continue
        handle = str(payload.get("handle_id") or payload.get("unit_id") or "")
        runnable_ids = _id_list(payload.get("runnable_unit_ids") or payload.get("unit_ids"))
        intervals.append(
            {
                "start_s": t0,
                "end_s": t0 + wait_s,
                "duration_s": wait_s,
                "kind": "HANDLE_WAIT",
                "waited_unit": handle or None,
                "loop_doing": f"HANDLE_WAIT on {handle or 'unnamed handle'}",
                "explicit_runnable_ids": runnable_ids,
                "launch_index": None,
                "complete_index": None,
            }
        )

    detached_ok = _verified_detached_ids(events)
    open_at: dict[str, int] = {}
    for idx, event in enumerate(events):
        kind = _kind(event)
        if kind in SYNC_LAUNCH_KINDS:
            uid = _event_unit_id(event)
            # A launch proven detached does not block the loop, so it cannot be
            # the start of a wait. Proven means a matching detached_started, not
            # the payload's own claim.
            if uid and uid not in detached_ok:
                open_at[uid] = idx
            continue
        if kind not in COMPLETE_KINDS:
            continue
        uid = _event_unit_id(event)
        if not uid or uid not in open_at:
            continue
        i = open_at.pop(uid)
        t0 = _t_s(events[i])
        t1 = _t_s(event)
        if t1 <= t0:
            continue
        if _sync_launch_between(events, i, idx, exclude_uid=uid):
            continue
        unit = _event_unit(events[i])
        desc = str(unit.get("description") or "")[:96]
        loop_doing = f"waiting on subprocess {uid}"
        if desc:
            loop_doing = f"waiting on subprocess {uid}: {desc}"
        intervals.append(
            {
                "start_s": t0,
                "end_s": t1,
                "duration_s": t1 - t0,
                "kind": "SUBPROCESS_WAIT",
                "waited_unit": uid,
                "loop_doing": loop_doing,
                "explicit_runnable_ids": [],
                "launch_index": i,
                "complete_index": idx,
                "waited_frontier": str(unit.get("frontier_id") or "") or None,
                "waited_resource_class": _resource_class_of(unit) or None,
            }
        )
    intervals.sort(key=lambda row: (float(row["start_s"]), float(row["end_s"])))
    return intervals


def _queue_snapshots(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, event in enumerate(events):
        if _kind(event) not in QUEUE_KINDS:
            continue
        ids, n = _queue_ids(event)
        payload = _payload(event)
        out.append(
            {
                "index": idx,
                "t_s": _t_s(event),
                "kind": _kind(event),
                "ids": ids,
                "n": n,
                "source": payload.get("source"),
                "note": payload.get("note"),
            }
        )
    return out


def _offered_before(events: Sequence[Mapping[str, Any]], t: float) -> set[str]:
    offered: set[str] = set()
    for event in events:
        if _t_s(event) > t:
            break
        kind = _kind(event)
        if kind in QUEUE_KINDS or kind in REFILL_KINDS:
            ids, _n = _queue_ids(event)
            offered.update(ids)
        if kind in LAUNCH_KINDS:
            unit = _event_unit(event)
            uid = _unit_id_of(unit)
            if uid:
                offered.add(uid)
            fid = str(unit.get("frontier_id") or "").strip()
            if fid:
                offered.add(fid)
    return offered


def _runnable_during(
    interval: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    world: World,
    by_id: Mapping[str, Mapping[str, Any]],
    by_frontier: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Units that were runnable safe work while this blocked interval held."""
    t0 = float(interval["start_s"])
    t1 = float(interval["end_s"])
    waited = str(interval.get("waited_unit") or "")
    waited_frontier = str(interval.get("waited_frontier") or "")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(ident: str, source: str) -> None:
        ident = str(ident or "").strip()
        if not ident or ident == waited or ident in seen:
            return
        seen.add(ident)
        row = _hydrate_candidate(ident, by_id, by_frontier)
        row = dict(row)
        row["_candidate_source"] = source
        candidates.append(row)

    for ident in interval.get("explicit_runnable_ids") or []:
        add(str(ident), "HANDLE_WAIT.runnable_unit_ids")

    snapshots = _queue_snapshots(events)
    last_before = None
    at_end: list[dict[str, Any]] = []
    for snap in snapshots:
        if snap["t_s"] <= t0:
            last_before = snap
        elif t0 < snap["t_s"] < t1:
            last_before = snap
        elif snap["t_s"] == t1 or abs(snap["t_s"] - t1) < 1e-9:
            at_end.append(snap)
        elif snap["t_s"] > t1 and not at_end:
            # First snapshot after the wait ends (same-second leftover is
            # usually timestamped equal to the complete; a 0-gap leftover
            # still describes what was live when the wait ended).
            if snap["t_s"] - t1 <= 1.0:
                at_end.append(snap)
    if last_before and int(last_before.get("n") or 0) > 0:
        for ident in last_before["ids"]:
            add(ident, f"queue:{last_before['kind']}@t_s={last_before['t_s']}")
    for snap in at_end:
        if int(snap.get("n") or 0) > 0 or snap["ids"]:
            src = snap.get("source") or snap["kind"]
            for ident in snap["ids"]:
                add(ident, f"leftover:{src}@t_s={snap['t_s']}")

    # Remaining catalog / same-queue work: the first synchronous launch at or
    # after the wait ends, plus later same-frontier launches, provided they
    # do not depend on the waited unit.
    complete_idx = interval.get("complete_index")
    start_idx = complete_idx if isinstance(complete_idx, int) else None
    if start_idx is None:
        start_idx = next(
            (i for i, e in enumerate(events) if _t_s(e) >= t1),
            len(events),
        )
    same_frontier_later: list[str] = []
    first_independent = None
    offered = _offered_before(events, t0)
    for event in events[start_idx:]:
        if _kind(event) not in SYNC_LAUNCH_KINDS:
            continue
        unit = _event_unit(event)
        uid = _unit_id_of(unit)
        if not uid or uid == waited:
            continue
        deps = _deps_of(unit)
        if waited and waited in deps:
            continue
        unmet = [d for d in deps if d not in world.completed_ids]
        if unmet:
            continue
        fid = str(unit.get("frontier_id") or "")
        if first_independent is None:
            first_independent = uid
            add(uid, "next_independent_launch_after_wait")
        if waited_frontier and fid == waited_frontier:
            same_frontier_later.append(uid)
            add(uid, "same_frontier_queue_residue")
        elif uid in offered or fid in offered:
            add(uid, "pre_offered_independent_launch")
        # One next launch is enough to name the queue head; keep scanning the
        # same frontier so the remaining specimen queue is visible.
        if first_independent and not waited_frontier:
            break
        if same_frontier_later and _t_s(event) > t1 + 1.0 and fid != waited_frontier:
            break

    runnable_rows: list[dict[str, Any]] = []
    # The waited unit is in-flight in `world` (scan through launch_index).
    for row in candidates:
        verdict = is_runnable_safe_work(row, world)
        if not verdict["runnable"]:
            continue
        runnable_rows.append(
            {
                "id": _unit_id_of(row),
                "resource_class": _resource_class_of(row) or None,
                "frontier_id": row.get("frontier_id"),
                "source": row.get("_candidate_source"),
                "reason": verdict["reason"],
            }
        )
    return runnable_rows


def _public_interval(row: Mapping[str, Any], runnable: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = [str(r.get("id")) for r in runnable if r.get("id")]
    return {
        "start_s": row["start_s"],
        "end_s": row["end_s"],
        "duration_s": row["duration_s"],
        "kind": row.get("kind"),
        "waited_unit": row.get("waited_unit"),
        "loop_doing": row.get("loop_doing"),
        "runnable": list(runnable)[:24],
        "runnable_ids": ids[:24],
        "n_runnable": len(ids),
        "n_runnable_truncated": max(0, len(ids) - 24),
    }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify(source: Any) -> dict[str, Any]:
    """Return exactly one of PASS / FAIL_NO_WAIT_ORCHESTRATION / SLOW_BUT_CORRECT."""
    events, meta = _load_events(source)
    elapsed = ad._elapsed_s(events, meta)
    intervals = _blocked_intervals(events)
    by_id, by_frontier = _known_units(events)
    forcing: list[dict[str, Any]] = []
    slow: list[dict[str, Any]] = []

    for interval in intervals:
        launch_index = interval.get("launch_index")
        if isinstance(launch_index, int):
            world, _, _ = _scan_state(events, launch_index)
        else:
            # HANDLE_WAIT: world just before the wait event.
            t0 = float(interval["start_s"])
            idx = 0
            for i, event in enumerate(events):
                if _t_s(event) <= t0:
                    idx = i
                else:
                    break
            world, _, _ = _scan_state(events, idx)
        runnable = _runnable_during(
            interval, events, world=world, by_id=by_id, by_frontier=by_frontier
        )
        public = _public_interval(interval, runnable)
        if runnable:
            forcing.append(public)
        else:
            slow.append(public)

    if forcing:
        verdict = FAIL_NO_WAIT_ORCHESTRATION
        first = forcing[0]
        reason = (
            f"loop blocked {first['start_s']}->{first['end_s']}s "
            f"({first['duration_s']}s) doing {first['loop_doing']}; "
            f"runnable safe work={first['runnable_ids'][:8]}"
        )
    elif slow:
        verdict = SLOW_BUT_CORRECT
        longest = max(slow, key=lambda r: float(r["duration_s"]))
        reason = (
            f"loop was slow ({longest['duration_s']}s at "
            f"{longest['start_s']}->{longest['end_s']}s doing "
            f"{longest['loop_doing']}) with nothing runnable"
        )
    else:
        verdict = PASS
        reason = "no interval where the loop was blocked while runnable safe work existed"

    tail = adjudicate_tail(events, elapsed, forcing, slow)
    source_path = meta.get("_source_path")
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "verdict": verdict,
        "reason": reason,
        "elapsed_s": elapsed,
        "n_events": len(events),
        "n_blocked_intervals": len(intervals),
        "n_forcing_intervals": len(forcing),
        "n_slow_intervals": len(slow),
        "forcing_intervals": forcing,
        "slow_intervals": slow,
        "tail": tail,
        "runnable_safe_work_rule": RUNNABLE_SAFE_WORK_RULE,
        "verdicts_are": list(VERDICTS),
        "source_path": source_path,
        "fixtures": False,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "score": None,
    }


def measure(source: Any) -> dict[str, Any]:
    """Public alias. A trial asks for this verdict the way it asks degeneracy.measure."""
    return classify(source)


def adjudicate_tail(
    events: Sequence[Mapping[str, Any]],
    elapsed: float,
    forcing: Sequence[Mapping[str, Any]],
    slow: Sequence[Mapping[str, Any]],
    *,
    tail_s: float = TAIL_S,
) -> dict[str, Any]:
    """The 1h last-2864s calibration. Verdict depends entirely on whether anything was runnable."""
    start = max(0.0, float(elapsed) - float(tail_s))
    end = float(elapsed)
    applies = float(elapsed) >= float(tail_s) - 1e-9

    def overlaps(row: Mapping[str, Any]) -> bool:
        return float(row["end_s"]) > start and float(row["start_s"]) < end + 1e-9

    tail_forcing = [dict(r) for r in forcing if overlaps(r)]
    tail_slow = [dict(r) for r in slow if overlaps(r)]

    n_refills = sum(
        1 for e in events if _kind(e) in REFILL_KINDS and _t_s(e) >= start
    )
    n_decisions = sum(
        1 for e in events if _kind(e) in DECISION_KINDS and _t_s(e) >= start
    )
    last_refill = max(
        (_t_s(e) for e in events if _kind(e) in REFILL_KINDS),
        default=None,
    )
    leftover = None
    for event in reversed(list(events)):
        if _kind(event) == "next_work_left":
            ids, n = _queue_ids(event)
            leftover = {
                "t_s": _t_s(event),
                "n": n,
                "ids": ids,
                "source": _payload(event).get("source"),
                "note": _payload(event).get("note"),
            }
            break

    if tail_forcing:
        verdict = FAIL_NO_WAIT_ORCHESTRATION
        why = (
            "runnable safe work existed during the I/O-bound tail: remaining "
            "independent units (deps met, resource free) sat while the loop "
            "waited on a subprocess"
        )
    elif tail_slow:
        verdict = SLOW_BUT_CORRECT
        why = (
            "the tail is a long I/O-bound wait and nothing passing the "
            "runnable-safe-work rule was available"
        )
    else:
        verdict = PASS
        why = "no blocked interval in the tail window"

    parked = []
    for event in events:
        if _kind(event) in SLEEP_KINDS:
            payload = _payload(event)
            parked.append(
                {
                    "resource_class": payload.get("resource_class"),
                    "wake_condition": payload.get("wake_condition"),
                    "t_s": _t_s(event),
                    "counted_runnable": False,
                    "why": "sleeping unit whose wake condition is unsatisfied is not runnable",
                }
            )

    return {
        "applies": applies,
        "start_s": start,
        "end_s": end,
        "duration_s": end - start if applies else 0.0,
        "named_duration_s": float(tail_s),
        "verdict": verdict if applies else None,
        "why": why if applies else "elapsed shorter than the 2864s calibration window",
        "not": (
            SLOW_BUT_CORRECT
            if applies and verdict == FAIL_NO_WAIT_ORCHESTRATION
            else (
                FAIL_NO_WAIT_ORCHESTRATION
                if applies and verdict == SLOW_BUT_CORRECT
                else None
            )
        ),
        "n_forcing_intervals": len(tail_forcing),
        "n_slow_intervals": len(tail_slow),
        "forcing_intervals": tail_forcing,
        "slow_intervals": tail_slow,
        "n_refills_in_tail": n_refills,
        "n_decisions_in_tail": n_decisions,
        "last_refill_t_s": last_refill,
        "leftover_at_end": leftover,
        "parked_sleeping_not_runnable": parked,
        "evidence_class": "STATIC_ONLY",
    }


def replay_disk_timelines() -> dict[str, Any]:
    """Real receipts on disk, not fixtures. POWER_TORTURE_TIMELINE if landed."""
    one_h_path = REPO / TIMELINE_1H_REL
    detached_path = REPO / DETACHED_TRIAL_REL
    torture_path = REPO / POWER_TORTURE_TIMELINE_REL
    one_h = classify(one_h_path)
    detached = classify(detached_path)
    if torture_path.is_file():
        torture: dict[str, Any] = {
            "present": True,
            "path": POWER_TORTURE_TIMELINE_REL,
            **classify(torture_path),
        }
    else:
        torture = {
            "present": False,
            "path": POWER_TORTURE_TIMELINE_REL,
            "verdict": None,
            "reason": (
                "lane e2torture has not landed POWER_TORTURE_TIMELINE.json; "
                "POWER_TORTURE.json is a doctrine receipt without events/timeline"
            ),
            "fixtures": False,
        }
    return {
        "fixtures": False,
        "autonomy_1h": one_h,
        "detached_work_trial": detached,
        "power_torture_timeline": torture,
        "paths": {
            "autonomy_1h": TIMELINE_1H_REL,
            "detached_work_trial": DETACHED_TRIAL_REL,
            "power_torture_timeline": POWER_TORTURE_TIMELINE_REL,
        },
    }


# ---------------------------------------------------------------------------
# Trial gate — public API. Does not edit improvement_trial.py.
# ---------------------------------------------------------------------------


def apply_to_trial_verdict(
    judged: Mapping[str, Any],
    source: Any,
) -> dict[str, Any]:
    """Overlay this obligation on a trial judge result.

    A trial that would PASS while FAIL_NO_WAIT_ORCHESTRATION holds cannot.
    SLOW_BUT_CORRECT does not convert a PASS into a FAIL — that is the
    distinction this obligation exists to protect.
    """
    report = classify(source)
    out = dict(judged)
    out["no_wait_orchestration"] = {
        "verdict": report["verdict"],
        "reason": report["reason"],
        "n_forcing_intervals": report["n_forcing_intervals"],
        "forcing_intervals": report["forcing_intervals"],
        "n_slow_intervals": report["n_slow_intervals"],
        "tail": report.get("tail"),
    }
    conditions = [dict(c) if isinstance(c, Mapping) else c for c in (out.get("conditions") or [])]
    unmet = [str(x) for x in (out.get("unmet") or [])]
    if report["verdict"] == FAIL_NO_WAIT_ORCHESTRATION:
        out["failed_on_no_wait_orchestration"] = True
        if out.get("verdict") == PASS:
            out["verdict"] = "FAIL"
            out["reason"] = (
                "FAIL_NO_WAIT_ORCHESTRATION: "
                + str(report["reason"])
            )
        if "no_wait_orchestration" not in unmet:
            unmet.append("no_wait_orchestration")
        conditions.append(
            {
                "id": "no_wait_orchestration",
                "met": False,
                "detail": report["reason"],
                "cites": [],
                "verdict": FAIL_NO_WAIT_ORCHESTRATION,
            }
        )
        auto = [dict(a) if isinstance(a, Mapping) else a for a in (out.get("automatic_failures") or [])]
        auto.append(
            {
                "id": "no_wait_orchestration",
                "detail": report["reason"],
            }
        )
        out["automatic_failures"] = auto
    else:
        out["failed_on_no_wait_orchestration"] = False
        conditions.append(
            {
                "id": "no_wait_orchestration",
                "met": True,
                "detail": f"{report['verdict']}: {report['reason']}",
                "cites": [],
                "verdict": report["verdict"],
            }
        )
    out["conditions"] = conditions
    out["unmet"] = unmet
    return out


def eval_no_wait_orchestration(record: Any) -> dict[str, Any]:
    """Evaluator a trial can put next to eval_no_degeneracy. Same {id,met,detail,cites} shape."""
    report = classify(record)
    if report["verdict"] == FAIL_NO_WAIT_ORCHESTRATION:
        return {
            "id": "no_wait_orchestration",
            "met": False,
            "detail": report["reason"],
            "cites": [],
            "verdict": FAIL_NO_WAIT_ORCHESTRATION,
        }
    return {
        "id": "no_wait_orchestration",
        "met": True,
        "detail": f"{report['verdict']}: {report['reason']}",
        "cites": [],
        "verdict": report["verdict"],
    }


def judge_improvement_trial(record: Any) -> dict[str, Any]:
    """The callable at the point it matters: improvement_trial.judge, then this gate.

    Does not edit tools/future/improvement_trial.py. A would-PASS trial that
    carries FAIL_NO_WAIT_ORCHESTRATION comes back FAIL.
    """
    from tools.future import improvement_trial as it

    original = getattr(it.judge, "_no_wait_original", None) or _ORIGINAL_JUDGE or it.judge
    if getattr(original, "_no_wait_orchestration_wrapped", False):
        original = getattr(original, "_no_wait_original", original)
    judged = original(record)
    return apply_to_trial_verdict(judged, record)


def install_into_improvement_trial() -> Any:
    """Wrap improvement_trial.judge in place. Idempotent. The trial file is not edited."""
    global _ORIGINAL_JUDGE
    from tools.future import improvement_trial as it

    current = it.judge
    if getattr(current, "_no_wait_orchestration_wrapped", False):
        return current
    _ORIGINAL_JUDGE = current

    def wrapped(record: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        judged = _ORIGINAL_JUDGE(record, *args, **kwargs)
        return apply_to_trial_verdict(judged, record)

    wrapped._no_wait_orchestration_wrapped = True  # type: ignore[attr-defined]
    wrapped._no_wait_original = _ORIGINAL_JUDGE  # type: ignore[attr-defined]
    it.judge = wrapped
    it.EVALUATORS.setdefault("no_wait_orchestration", eval_no_wait_orchestration)
    return wrapped


def uninstall_from_improvement_trial() -> None:
    """Restore the unwrapped judge. Tests use this so they do not leak a wrap."""
    from tools.future import improvement_trial as it

    original = _ORIGINAL_JUDGE
    if original is not None:
        it.judge = original


def _auto_install() -> None:
    try:
        install_into_improvement_trial()
    except Exception:
        # Classify and measure still work if the trial module cannot load.
        return


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def _provenance(rel: str) -> dict[str, Any]:
    path = REPO / rel
    present = path.is_file()
    return {
        "rel": rel,
        "present": present,
        "sha256": sha256_file(path) if present else None,
        "n_bytes": path.stat().st_size if present else None,
    }


def _compact_report(report: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "verdict",
        "reason",
        "elapsed_s",
        "n_events",
        "n_blocked_intervals",
        "n_forcing_intervals",
        "n_slow_intervals",
        "source_path",
        "fixtures",
    )
    out = {k: report.get(k) for k in keep}
    forcing = list(report.get("forcing_intervals") or [])
    slow = list(report.get("slow_intervals") or [])
    out["forcing_intervals"] = forcing
    out["slow_intervals"] = slow
    tail = dict(report.get("tail") or {})
    # Keep tail evidence; it is the calibration.
    out["tail"] = tail
    out["score"] = None
    return out


def _cited_detached_metrics(path: Path) -> dict[str, Any]:
    """Copy, as strings, measurements already on the detached receipt. Do not re-measure."""
    if not path.is_file():
        return {"present": False}
    doc = json.loads(path.read_text())
    wakeup = doc.get("wakeup") or {}
    proofs = doc.get("proofs") or {}
    return {
        "present": True,
        "cited_idle_runnable_seconds": str(doc.get("idle_runnable_seconds")),
        "cited_safe_in_flight_bound": str(doc.get("safe_in_flight_bound")),
        "cited_kqueue_latency_s": str(wakeup.get("kqueue_latency_s")),
        "cited_child_pid": str((doc.get("child") or {}).get("pid")),
        "cited_independent_progress_n": str(
            (proofs.get("independent_progress_during_wait") or {}).get("n_completed_during_wait")
        ),
        "copied_as_strings_not_remeasured": True,
        "source": DETACHED_TRIAL_REL,
    }


def build() -> Path:
    replay = replay_disk_timelines()
    one_h = replay["autonomy_1h"]
    detached = replay["detached_work_trial"]
    torture = replay["power_torture_timeline"]
    tail = one_h.get("tail") or {}
    cited = _cited_detached_metrics(REPO / DETACHED_TRIAL_REL)

    opposite = (
        one_h.get("verdict") == FAIL_NO_WAIT_ORCHESTRATION
        and detached.get("verdict") == PASS
    )
    tail_adjudicated = bool(tail.get("applies")) and tail.get("verdict") in VERDICTS
    module_verdict = (
        PASS
        if opposite and tail_adjudicated
        else "FAIL"
    )

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "obligation": (
            "NO-WAIT ORCHESTRATION IS AN EXPLICIT FAILURE TARGET. If runnable "
            "safe work exists while the whole HCLI loop reports waiting on a "
            "subprocess, that is FAIL NO_WAIT_ORCHESTRATION, NOT A SLOW RESIDENT."
        ),
        "three_way_verdict": list(VERDICTS),
        "runnable_safe_work_rule": RUNNABLE_SAFE_WORK_RULE,
        "tail_s": TAIL_S,
        "sync_launch_kinds_are_the_blocking_loop": list(sorted(SYNC_LAUNCH_KINDS)),
        "detached_launch_kinds_are_not_a_blocked_loop": list(sorted(DETACHED_LAUNCH_KINDS)),
        "hardware_parked_classes": list(sorted(HARDWARE_PARKED_CLASSES)),
        "free_resource_classes": list(sorted(FREE_RESOURCE_CLASSES)),
        "replay": {
            "fixtures": False,
            "autonomy_1h": {
                **_compact_report(one_h),
                "provenance": _provenance(TIMELINE_1H_REL),
            },
            "detached_work_trial": {
                **_compact_report(detached),
                "provenance": _provenance(DETACHED_TRIAL_REL),
                "cited_from_receipt": cited,
            },
            "power_torture_timeline": {
                "present": torture.get("present"),
                "path": torture.get("path"),
                "verdict": torture.get("verdict"),
                "reason": torture.get("reason"),
                "provenance": _provenance(POWER_TORTURE_TIMELINE_REL),
            },
        },
        "one_h_measured": {
            "verdict": one_h.get("verdict"),
            "reason": one_h.get("reason"),
            "elapsed_s": one_h.get("elapsed_s"),
            "n_forcing_intervals": one_h.get("n_forcing_intervals"),
            "n_slow_intervals": one_h.get("n_slow_intervals"),
            "longest_forcing": (
                max(one_h.get("forcing_intervals") or [{}], key=lambda r: float(r.get("duration_s") or 0))
                if one_h.get("forcing_intervals")
                else None
            ),
        },
        "one_h_tail": tail,
        "detached_measured": {
            "verdict": detached.get("verdict"),
            "reason": detached.get("reason"),
            "elapsed_s": detached.get("elapsed_s"),
            "n_forcing_intervals": detached.get("n_forcing_intervals"),
            "n_slow_intervals": detached.get("n_slow_intervals"),
            "cited_from_receipt": cited,
        },
        "distinction": {
            "FAIL_NO_WAIT_ORCHESTRATION": (
                "architecture failure: the loop waited on a subprocess while "
                "runnable safe work existed. Invites a fix."
            ),
            "SLOW_BUT_CORRECT": (
                "performance: the loop was slow and NOTHING was runnable. "
                "Invites patience. Must not be reported as FAIL_NO_WAIT_ORCHESTRATION."
            ),
            "PASS": (
                "no blocked-while-runnable interval. The detached trial is this "
                "shape: the loop released and independent work ran."
            ),
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "does_not_edit": [
            "tools/future/no_wait_scheduler.py",
            "tools/future/detached_trial.py",
            "tools/future/autonomy_degeneracy.py",
            "tools/future/improvement_trial.py",
        ],
        "api_for_callers": {
            "classify": "tools.future.no_wait_orchestration.classify",
            "measure": "tools.future.no_wait_orchestration.measure",
            "is_runnable_safe_work": "tools.future.no_wait_orchestration.is_runnable_safe_work",
            "independent_runnable_now": (
                "tools.future.no_wait_orchestration.independent_runnable_now"
            ),
            "apply_to_trial_verdict": (
                "tools.future.no_wait_orchestration.apply_to_trial_verdict"
            ),
            "judge_improvement_trial": (
                "tools.future.no_wait_orchestration.judge_improvement_trial"
            ),
            "install_into_improvement_trial": (
                "tools.future.no_wait_orchestration.install_into_improvement_trial"
            ),
            "eval_no_wait_orchestration": (
                "tools.future.no_wait_orchestration.eval_no_wait_orchestration"
            ),
            "replay_disk_timelines": (
                "tools.future.no_wait_orchestration.replay_disk_timelines"
            ),
            "note": (
                "no_wait_scheduler.py, detached_trial.py, autonomy_degeneracy.py "
                "and improvement_trial.py were not edited. Importing this module "
                "wraps improvement_trial.judge so a would-PASS trial cannot PASS "
                "while FAIL_NO_WAIT_ORCHESTRATION holds. SLOW_BUT_CORRECT does "
                "not fail a trial."
            ),
        },
        "composed": {
            "no_wait_scheduler.runnable_now": (
                "independence from graph/frontier while handles are open"
            ),
            "detached.refuse_reason": (
                "GPU/lease/cargo units SLEEP; they are not runnable safe work"
            ),
            "autonomy_degeneracy._events_of": (
                "one loader for a path, a receipt, an event list, or a TrialRecord"
            ),
        },
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "head": git("rev-parse", "HEAD"),
        "verdict": module_verdict,
    }
    _assert_no_hardware_claims(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--timeline", default=None, help="path to a timeline JSON")
    args = parser.parse_args(argv)
    if args.timeline:
        report = classify(args.timeline)
        print(
            json.dumps(
                {
                    "verdict": report["verdict"],
                    "reason": report["reason"],
                    "n_forcing_intervals": report["n_forcing_intervals"],
                    "forcing_intervals": report["forcing_intervals"][:8],
                    "tail": {
                        "verdict": (report.get("tail") or {}).get("verdict"),
                        "start_s": (report.get("tail") or {}).get("start_s"),
                        "end_s": (report.get("tail") or {}).get("end_s"),
                        "why": (report.get("tail") or {}).get("why"),
                    },
                },
                indent=2,
            )
        )
        return 0 if report["verdict"] == PASS else 1
    path = build()
    doc = json.loads(path.read_text())
    print(f"wrote {path}")
    print(f"verdict={doc.get('verdict')}")
    replay = doc.get("replay") or {}
    print(
        f"1h={((replay.get('autonomy_1h') or {}).get('verdict'))} "
        f"detached={((replay.get('detached_work_trial') or {}).get('verdict'))} "
        f"tail={((doc.get('one_h_tail') or {}).get('verdict'))}"
    )
    if args.selftest or args.record:
        return 0 if doc.get("verdict") == PASS else 1
    return 0


_auto_install()


if __name__ == "__main__":
    raise SystemExit(main())
