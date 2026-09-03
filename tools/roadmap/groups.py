"""Structured capabilities: many audited gates, one thing a reader must think about.

The roadmap listed six U50 gates -- 34_TO_40, 40_TO_50, ... 80_TO_90 -- as though
they were six architectural capabilities. They are milestones on ONE physical
frontier, and presenting them as peers makes the hardware section look six times
larger than the decision it contains.

The gates are NOT deleted. Deleting them would destroy audit lineage and break a
test that floors the gate count, and a roadmap that shrinks by discarding evidence
is not compression. Instead each group declares its members, and the generated
documents present the group once with its members as sub-status.

This is the nomenclature-compression rule applied where it belongs: fewer strong
nouns for a reader, without fewer facts for the auditor.
"""
from __future__ import annotations

from typing import Any

GROUPS: dict[str, dict[str, Any]] = {
    "U50_PROTECTED_PERFORMANCE_FRONTIER": {
        "what": (
            "One physical throughput frontier on the Alveo U50DD / XCU50-class "
            "board, 8 GB HBM2. Its milestones are Pareto points, not separate "
            "capabilities."
        ),
        "members": (
            "U50_34_TO_40", "U50_40_TO_50", "U50_50_TO_60",
            "U50_60_TO_70", "U50_70_TO_80", "U50_80_TO_90",
        ),
        "state_fields": (
            "baseline_tps", "current_tps", "target_tps", "capability_contract",
            "profile", "pareto_points", "receipts",
        ),
        "measured": False,
        "why_unmeasured": (
            "board performance is UNMEASURED until physical activation; every "
            "milestone here is a target, and no receipt may say otherwise"
        ),
    },
    "FPGA_PREBOARD": {
        "what": (
            "Everything completable before the board arrives. One pipeline, not "
            "nine top-level nouns."
        ),
        "members": ("FPGA_PREBOARD_SCHEMAS", "FPGA_LINK_SIM", "FPGA_PARTITION_SIM", "FPGA_HWIR"),
        "state_fields": (
            "machine_genome", "hwir", "hbm_topology", "cycle_resource_model",
            "module_model", "organ_mapper", "placement_model",
            "host_transport_model", "adaptation_clock_model",
        ),
        "measured": False,
        "why_unmeasured": (
            "a cycle/resource model is a MODEL. It may reach CYCLE_APPROX and "
            "never HARDWARE_MEASURED without the board."
        ),
    },
    "HMF_FUSION_SOFTWARE_TRUTH": {
        "what": (
            "The part of heterogeneous object truth that is software-verifiable "
            "NOW, kept separate from the part that needs a second physical domain."
        ),
        "members": ("HMF_DEVICE_VISIBLE_TRUST", "FUSION_FIRST_HETEROGENEOUS_EXECUTABLE"),
        "state_fields": (
            "logical_object_model", "ownership", "versions", "trust_states",
            "move_or_recompute_decision", "functional_sim",
        ),
        "measured": False,
        "why_unmeasured": (
            "object identity, ownership, versioning and move/recompute policy can "
            "be built and adversarially simulated before a second device exists. "
            "Simulation is NEVER promoted to hardware truth: the physical half "
            "stays a hardware gate."
        ),
    },
}


def group_of(gate_id: str) -> str | None:
    for name, spec in GROUPS.items():
        if gate_id in spec["members"]:
            return name
    return None


def summarize(gates: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per group, carrying its members' real statuses."""
    rows = []
    for name, spec in GROUPS.items():
        members = [gates[m] for m in spec["members"] if m in gates]
        rows.append({
            "capability": name,
            "what": spec["what"],
            "members": list(spec["members"]),
            "member_statuses": {m: gates[m]["status"] for m in spec["members"] if m in gates},
            "state_fields": list(spec["state_fields"]),
            "evidence_tiers": sorted({g.get("evidence_tier") for g in members}),
            "measured": spec["measured"],
            "why_unmeasured": spec["why_unmeasured"],
            "collapses": len(spec["members"]),
        })
    return rows
