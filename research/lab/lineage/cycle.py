"""One sacrificial generation attempt.

Gate first (external). Checksum next. Launch next. Authority moves last.
A failed launch rolls back to LAST_KNOWN_GOOD. Slot 3 is seized for
qualification and released afterwards.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from lab.lineage.bus import ResearchBus
from lab.lineage.canon import utc_now
from lab.lineage.four_slots import FourSlotScheduler
from lab.lineage.identity import GenesisInstance, Invoker, as_instance, as_invoker
from lab.lineage.promotion import evaluate_promotion
from lab.lineage.state import LineageState
from lab.lineage.transfer import pack_state
from lab.receipts import seal

SCHEMA = "hawking.lineage.reproduction_cycle.v1"

LANE_TO_SLOT: dict[str, int] = {
    "integrator": 0,
    "representation": 1,
    "execution": 2,
}


def _home_slot(child: GenesisInstance) -> int:
    return LANE_TO_SLOT.get(child.lane, 1)


def reproduce(
    *,
    state: LineageState,
    child: GenesisInstance | Mapping[str, Any],
    evidence: Mapping[str, Any],
    invoker: Invoker | Mapping[str, Any],
    parent_payload: Mapping[str, Any],
    launcher: Callable[[GenesisInstance], bool],
    scheduler: FourSlotScheduler | None = None,
    bus: ResearchBus | None = None,
) -> dict[str, Any]:
    """Run one reproduction attempt. Never moves authority on REJECT/PENDING/failed launch."""
    if not state.armed or state.current is None:
        raise ValueError("lineage must be armed with a CURRENT Genesis")
    parent = state.current
    child_i = as_instance(child, "child")
    inv = as_invoker(invoker)

    if state.candidate is None or state.candidate.instance_id != child_i.instance_id:
        state.nominate(child_i)
    state.snapshot_current_as_lkg()

    qualification = None
    if scheduler is not None:
        if scheduler.slot(0).empty:
            scheduler.occupy(0, parent.instance_id, purpose="parent_integrator")
        home = _home_slot(child_i)
        if not scheduler.slot(home).empty and scheduler.slot(home).occupant_id != child_i.instance_id:
            scheduler.release(home, reason="previous child vacated for new candidate")
        if scheduler.slot(home).empty:
            scheduler.occupy(home, child_i.instance_id, purpose=child_i.lane)
        qualification = scheduler.request_qualification(child_i.instance_id)

    verdict = evaluate_promotion(
        parent=parent,
        child=child_i,
        evidence=evidence,
        invoker=inv,
        lineage=state,
    )
    if verdict["verdict"] != "ACCEPT":
        if scheduler is not None:
            scheduler.release(3, reason=f"qualification ended: {verdict['verdict']}")
        return seal(
            {
                "schema": SCHEMA,
                "outcome": verdict["verdict"],
                "authority_moved": False,
                "rolled_back": False,
                "reason": verdict["reason"],
                "verdict": verdict,
                "qualification": qualification,
                "recorded_at": utc_now(),
            }
        )

    package = pack_state(parent, parent_payload, to=child_i)
    launch = state.launch_successor(launcher)
    if not launch.ok:
        if scheduler is not None:
            scheduler.release(3, reason="successor launch failed; rolled back")
        return seal(
            {
                "schema": SCHEMA,
                "outcome": "ROLLBACK",
                "authority_moved": False,
                "rolled_back": True,
                "reason": launch.reason,
                "verdict": verdict,
                "launch": launch.to_dict(),
                "transfer_checksum_sha256": package.get("checksum_sha256"),
                "qualification": qualification,
                "recorded_at": utc_now(),
            }
        )

    handover = state.handover(package=package, invoker=inv, verdict=verdict)
    if scheduler is not None:
        scheduler.release(3, reason="qualification complete; successor seated")
        # Parent vacated slot 0; successor leaves its child slot and becomes integrator.
        if not scheduler.slot(0).empty:
            scheduler.release(0, reason="parent terminated")
        home = _home_slot(child_i)
        if not scheduler.slot(home).empty and scheduler.slot(home).occupant_id == child_i.instance_id:
            scheduler.release(home, reason="successor left child slot to become integrator")
        if scheduler.slot(0).empty:
            scheduler.occupy(0, state.current.instance_id, purpose="parent_integrator")
    if bus is not None and state.current is not None and state.current.research_state:
        nxt = state.current.research_state.get("NEXT_BOTTLENECK")
        if nxt:
            bus.publish(
                {
                    "type": "NEXT_BOTTLENECK",
                    "sender": state.current.instance_id,
                    "artifact_sha": state.current.artifact_sha,
                    "runtime_sha": state.current.runtime_sha,
                    "epistemic_class": "HYPOTHESIS",
                    "receipt_ref": "",
                    "affected_subsystem": "lineage",
                    "body": {"NEXT_BOTTLENECK": nxt},
                }
            )
    return seal(
        {
            "schema": SCHEMA,
            "outcome": "PROMOTED",
            "authority_moved": True,
            "rolled_back": False,
            "reason": "external ACCEPT, checksum verified, successor launched, parent terminated",
            "verdict": verdict,
            "handover": handover,
            "launch": launch.to_dict(),
            "qualification": qualification,
            "current_id": state.current.instance_id if state.current else None,
            "lkg_id": state.last_known_good.instance_id if state.last_known_good else None,
            "valid_count": state.valid_count(),
            "recorded_at": utc_now(),
        }
    )
