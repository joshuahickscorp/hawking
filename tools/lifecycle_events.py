#!/usr/bin/env python3
"""Civilization lifecycle events: audit + cheap connections to the EXISTING buses.

Do not create a second EventBus. The in-process bus is hcli.events.EventBus
(session/UI). Disk-backed completion is tools.future.wakeup. Specimen seals
already flow through tools.future.modellake_events.consume, invoked by
tools.odyssey.modellake_watch.emit_modellake_events_once. Receipt-only trials
must not be reported as live consumers merely because their source still exists.

This module (1) publishes the subscriber table the audit asked for, and
(2) connects the two named events that had no live consumer:
child_qualified and hardware_profile_changed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from tools.roadmap.hardware import WAKE_CONDITIONS, blocked_hardware_wakes
from tools.successor_select import gate as successor_gate

SCHEMA = "hawking.lifecycle.events.v1"
REPO = Path(__file__).resolve().parents[1]
GRAPH_REL = REPO / "civilization" / "CAPABILITY_GRAPH.json"

# The six named events from the contract. Kinds are snake_case so they can
# sit next to wakeup's disk events without colliding with HCLI session types.
SPECIMEN_SEALED = "specimen_sealed"
EXPERIMENT_COMPLETED = "experiment_completed"
LAW_UPDATED = "law_updated"
CHILD_QUALIFIED = "child_qualified"
RESOURCE_AVAILABLE = "resource_available"
HARDWARE_PROFILE_CHANGED = "hardware_profile_changed"

LIFECYCLE_KINDS: tuple[str, ...] = (
    SPECIMEN_SEALED,
    EXPERIMENT_COMPLETED,
    LAW_UPDATED,
    CHILD_QUALIFIED,
    RESOURCE_AVAILABLE,
    HARDWARE_PROFILE_CHANGED,
)


def subscriber_table() -> list[dict[str, Any]]:
    """STATIC audit: per event, the implementing symbol and a real call site.

    A module import is not a call site. Each row names the symbol that is
    invoked and the file:line (or function) that invokes it. `no consumer`
    is explicit when nothing outside tests/self-description calls it.
    """
    return [
        {
            "event": SPECIMEN_SEALED,
            "bus": "tools.future.modellake_events (not a second EventBus)",
            "consumer": "tools.future.modellake_events.consume",
            "call_site": (
                "tools.odyssey.modellake_watch.emit_modellake_events_once "
                "-> modellake_events.build() -> consume(); "
                "watch loop: maybe_emit_modellake_events"
            ),
            "status": "live_consumer",
            "evidence_tier": "STATIC",
        },
        {
            "event": EXPERIMENT_COMPLETED,
            "bus": "none (receipt-only historical trial)",
            "consumer": "none; tools.future.detached_trial was retired",
            "call_site": "no production call site; historical receipt retained",
            "status": "no consumer",
            "evidence_tier": "STATIC",
        },
        {
            "event": LAW_UPDATED,
            "bus": "none (receipt-only historical trigger)",
            "consumer": "none; tools.future.phase_listeners was retired",
            "call_site": "no production call site; historical receipt retained",
            "status": "no consumer",
            "evidence_tier": "STATIC",
        },
        {
            "event": CHILD_QUALIFIED,
            "bus": "existing successor_select.gate + this router",
            "consumer": "tools.lifecycle_events.on_child_qualified",
            "call_site": (
                "tools.selection_contract.qualify -> on_child_qualified; "
                "on_child_qualified -> successor_select.gate"
            ),
            "status": "live_consumer",
            "evidence_tier": "STATIC",
        },
        {
            "event": RESOURCE_AVAILABLE,
            "bus": "existing lifecycle router",
            "consumer": "tools.lifecycle_events.on_hardware_profile_changed",
            "call_site": "route(RESOURCE_AVAILABLE, payload) -> on_hardware_profile_changed",
            "status": "live_consumer",
            "evidence_tier": "STATIC",
        },
        {
            "event": HARDWARE_PROFILE_CHANGED,
            "bus": "tools.roadmap.hardware probes (no second EventBus)",
            "consumer": "tools.lifecycle_events.on_hardware_profile_changed",
            "call_site": "tools.roadmap.__main__.main -> on_hardware_profile_changed(doc)",
            "status": "live_consumer",
            "evidence_tier": "STATIC",
        },
    ]


def on_child_qualified(record: Mapping[str, Any]) -> dict[str, Any]:
    """Live consumer of child_qualified.

    Re-runs the G151 hard gate on the record's flags when they are present so
    a QUALIFIED event cannot skip successor_select.gate. Does not promote and
    does not install. Called by tools.selection_contract.qualify.
    """
    flags = record.get("flags") or {}
    ident = (record.get("identity") or {}).get("candidate_id") or record.get("name")
    hard = None
    if isinstance(flags, Mapping) and all(
        k in flags for k in ("doctor_pass", "provenance_valid", "native_path", "no_hidden_fallback")
    ):
        metrics = record.get("metrics") or {}
        hard = {
            "name": ident or "child",
            "bpw": metrics.get("effective_bpw") or metrics.get("bpw") or 0,
            "token_ns": metrics.get("token_ns") or 0,
            "doctor_pass": bool(flags.get("doctor_pass")),
            "provenance_valid": bool(flags.get("provenance_valid")),
            "native_path": bool(flags.get("native_path")),
            "no_hidden_fallback": bool(flags.get("no_hidden_fallback")),
        }
        successor_gate(hard)
    return {
        "schema": SCHEMA,
        "event": CHILD_QUALIFIED,
        "candidate_id": ident,
        "state": record.get("state"),
        "installed": False,
        "hard_gate_ran": hard is not None,
        "consumer": "tools.successor_select.gate" if hard is not None else "recorded",
    }


def on_hardware_profile_changed(
    graph: Mapping[str, Any] | None = None,
    *,
    wake_sleeping: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Live consumer of hardware_profile_changed.

    Reads BLOCKED_HARDWARE wake ids off the capability graph (already probed
    by the auditor — this does not re-run system_profiler). Gates whose wake
    condition is present are the ones the daemon should activate. On this
    host FPGA/DGX/eGPU/HMF/newer-M-series are absent, so activable is empty
    and wake_sleeping is not invoked with a fabricated True.
    """
    doc = graph
    if doc is None:
        import json
        doc = json.loads(GRAPH_REL.read_text())
    gates = doc.get("gates") or {}
    wakes = blocked_hardware_wakes(gates)
    activable: list[dict[str, Any]] = []
    sleeping: list[dict[str, Any]] = []
    for gid, wake in wakes:
        detail = (gates[gid].get("wake_condition_detail") or {}) if gid in gates else {}
        present = bool(detail.get("present"))
        row = {
            "gate": gid,
            "wake_condition": wake,
            "present": present,
            "description": WAKE_CONDITIONS.get(wake),
            "evidence_tier": detail.get("evidence_tier") or "STATIC",
        }
        if present:
            activable.append(row)
        else:
            sleeping.append(row)

    woken: list[Any] = []
    if activable and wake_sleeping is not None:
        woken = list(wake_sleeping() or [])
    # HCLI's scheduler/DAG store is the execution owner. Arrival graphs are
    # payloads only; this router does not revive a parallel future scheduler.

    return {
        "schema": SCHEMA,
        "event": HARDWARE_PROFILE_CHANGED,
        "n_blocked_hardware": len(wakes),
        "wake_ids": sorted({w for _, w in wakes}),
        "activable": activable,
        "sleeping": sleeping,
        "woken_unit_ids": woken,
        "consumer": "tools.lifecycle_events.on_hardware_profile_changed",
        "evidence_tier": "STATIC",
        "note": (
            "activable is empty unless a wake_condition_detail.present is true; "
            "absent FPGA/DGX/eGPU is a model of this host, not a measurement; "
            "HCLI owns any eventual wake/schedule operation, "
            "of those boards"
        ),
    }


def route(kind: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch one named lifecycle event to its EXISTING consumer.

    Not an EventBus.subscribe/emit API. Unknown kinds are refused rather than
    queued on a new bus.
    """
    payload = dict(payload or {})
    if kind == CHILD_QUALIFIED:
        return on_child_qualified(payload)
    if kind == HARDWARE_PROFILE_CHANGED:
        return on_hardware_profile_changed(payload.get("graph"))
    if kind == RESOURCE_AVAILABLE:
        # Same hardware-wake path: a resource becoming present is a profile change.
        return on_hardware_profile_changed(payload.get("graph"))
    if kind not in LIFECYCLE_KINDS:
        raise ValueError(f"unknown lifecycle kind {kind!r}; known={list(LIFECYCLE_KINDS)}")
    # Existing consumers stay on their own buses; receipt-only events are
    # reported as such rather than being promoted by this router.
    row = next(r for r in subscriber_table() if r["event"] == kind)
    return {
        "schema": SCHEMA,
        "event": kind,
        "routed": False,
        "reason": (
            "already has a live consumer on an existing bus; this router does "
            "not re-emit"
            if row["status"] == "live_consumer"
            else "no live consumer; historical receipt is retained and this router does not invent one"
        ),
        "consumer": row["consumer"],
        "call_site": row["call_site"],
        "status": row["status"],
    }
