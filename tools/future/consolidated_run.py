"""ODYSSEY CONSOLIDATED RUN — emit a launch descriptor, or refuse the launch.

A detached supervisor cannot execute a claim. This module is the document that
turns Odyssey I from a gate receipt into a run: identities, specimens, graphs,
lanes, handles, destinations, blocked wake conditions, the qualification
backlog, the time budget, restart state. It consults odyssey_launch's sixteen
criteria and REFUSES with the unmet list. A descriptor is emitted only when
the gate passes. Anything else would look like a launch.

What this refuses to do: invent a field, round an unread collection into `[]`,
launch on an unmet criterion, treat a specimen without verification status as
a source, or accept a fallback that is the resident in disguise.

What this cannot establish: that Odyssey started (the supervisor executes; this
only seals the document), any hardware quantity (STATIC_ONLY), or that a
declared tool ran (naming is not invocation).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import RECEIPTS, git, write_receipt
from tools.future import autonomy_run as ar
from tools.future import autonomy_trial as at
from tools.future import evidence_dag as ed
from tools.future import flash_nr_complete as nr
from tools.future import flash_nx_audit as nx
from tools.future import frontiers as fr
from tools.future import negative_index as ni
from tools.future import odyssey_launch as ol
from tools.future import qualification_pipeline as qp
from tools.future import resident_identity as ri
from tools.future import specimen_verify as sv
from tools.future import super_resident as sr
from tools.future import workgraph as wg
from tools.future import workunit_species as wus
from tools.future.candidate_planner import QueueNotFoundError


RECEIPT = "ODYSSEY_CONSOLIDATED_RUN.json"
SCHEMA = "hawking.future.consolidated_run.v1"
RECORDED_BY = "tools/future/consolidated_run.py"
VERSION = 1

PRESENT = "PRESENT"
EMPTY = "EMPTY"
UNAVAILABLE = "UNAVAILABLE"
TAG_STATES = (PRESENT, EMPTY, UNAVAILABLE)

REQUIRED_FIELDS: tuple[str, ...] = (
    "run_id",
    "start_time",
    "resident_identity",
    "fallback_identity",
    "machine_genome",
    "source_specimens",
    "phase_i_workgraphs",
    "phase_ii_listener",
    "phase_iii_listener",
    "resource_lanes",
    "evidence_dag_handle",
    "negative_science_handle",
    "nr_nx_destinations",
    "blocked_triggers",
    "qualification_backlog",
    "time_budget",
    "restart_state",
)

# Collections whose empty-vs-unread distinction is load-bearing. A bare list
# is rejected: `[]` cannot mean both "none exist" and "could not read".
TAGGED_COLLECTION_FIELDS: tuple[str, ...] = (
    "source_specimens",
    "phase_i_workgraphs",
    "blocked_triggers",
    "qualification_backlog",
    "nr_nx_destinations",
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. A REFUSED receipt is "
    "not Odyssey started. The run descriptor is emitted only when every "
    "odyssey_launch criterion is met; the supervisor executes, this module does not."
)

# Compact projection of a qualification-queue row. Full candidates carry
# expected_gpu_ns_mechanism / measurements; those are not this sidecar's to copy.
_QUEUE_KEEP: tuple[str, ...] = (
    "candidate_id",
    "status",
    "model",
    "blocked_reason",
    "scope_tags",
)


class DescriptorInvalid(ValueError):
    """A run descriptor is missing a required fact, or asserts a false identity."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = list(reasons)
        named = "; ".join(self.reasons) if self.reasons else "<unspecified>"
        super().__init__(f"REJECTED: {named}")


# ---------------------------------------------------------------------------
# Tagged disk facts. EMPTY and UNAVAILABLE never share a representation.
# ---------------------------------------------------------------------------


def tagged(
    status: str,
    *,
    reason: str | None = None,
    items: Sequence[Any] | None = None,
    value: Any = None,
    source: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """The only constructor for a field that might be missing.

    UNAVAILABLE requires a reason (could not read). EMPTY requires a reason
    (looked, none exist). PRESENT carries items and/or value. A caller that
    passes a bare list around this constructor is the bug.
    """
    if status not in TAG_STATES:
        raise ValueError(f"tag status {status!r} is not {TAG_STATES}")
    rec: dict[str, Any] = {"status": status}
    if status == UNAVAILABLE:
        if not reason:
            raise ValueError("UNAVAILABLE requires a reason; missing is not a pass")
        rec["reason"] = reason
        rec["items"] = None
        rec["value"] = None
        rec["n"] = None
    elif status == EMPTY:
        if not reason:
            raise ValueError("EMPTY requires a reason so it cannot be mistaken for unread")
        rec["reason"] = reason
        rec["items"] = []
        rec["n"] = 0
        rec["value"] = []
    else:
        rec["reason"] = reason
        if items is not None:
            rec["items"] = list(items)
            rec["n"] = len(rec["items"])
        if value is not None:
            rec["value"] = value
        if items is None and value is None:
            rec["items"] = []
            rec["n"] = 0
    if source:
        rec["source"] = source
    rec.update(extra)
    return rec


def is_tagged(node: Any) -> bool:
    return isinstance(node, Mapping) and node.get("status") in TAG_STATES


# ---------------------------------------------------------------------------
# Gate. The live evaluators are the authority; this module does not re-score.
# ---------------------------------------------------------------------------


def evaluate_gate(
    results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Consult odyssey_launch. Never stop at the first unmet."""
    rows = list(results) if results is not None else ol.evaluate_launch_criteria()
    verdict = ol.launch_verdict(rows)
    compact = [
        {"id": r.get("id"), "met": bool(r.get("met")), "reason": r.get("reason")}
        for r in rows
        if isinstance(r, Mapping)
    ]
    verdict = dict(verdict)
    verdict["criteria"] = compact
    return verdict


# ---------------------------------------------------------------------------
# Identity. Resident is Qwen27 sealed-3.14; fallback is Flash. Identical = none.
# ---------------------------------------------------------------------------


def _compact_resident(identity: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    nx_slot = identity.get("nx_id") if isinstance(identity, Mapping) else None
    nx_val = nx_slot.get("value") if isinstance(nx_slot, Mapping) else nx_slot
    family_slot = identity.get("model_family") if isinstance(identity, Mapping) else None
    family_val = family_slot.get("value") if isinstance(family_slot, Mapping) else family_slot
    model_id = None
    if isinstance(nx_val, Mapping):
        model_id = nx_val.get("model_id") or nx_val.get("resident_identity")
    return {
        "id": str(model_id or sr.QWEN_ID),
        "role": identity.get("residency_status") or ri.RESIDENCY_STATUS,
        "model_family": family_val,
        "nx_id": nx_val if isinstance(nx_val, Mapping) else None,
        "schema": ri.SCHEMA,
        "receipt": f"receipts/future/{ri.RECEIPT}",
        "source": source,
        "gpu_authority": False,
        "claim_class": ri.CLAIM_CLASS,
    }


def assemble_resident_identity() -> dict[str, Any]:
    """Disk identity, never a conversational reconstruction."""
    try:
        ident = ri.collect()
    except Exception as exc:
        # collect() failed. load() from a sealed receipt is the restart path;
        # a missing receipt is UNAVAILABLE, not a fabricated incumbent.
        try:
            ident = ri.load()
            return tagged(
                PRESENT,
                value=_compact_resident(ident, source="tools.future.resident_identity.load"),
                source=f"receipts/future/{ri.RECEIPT}",
            )
        except Exception as load_exc:
            probe = ol.probe_json(f"receipts/future/{ri.RECEIPT}")
            if probe.get("found") and isinstance(probe.get("doc"), Mapping):
                try:
                    ident = ri.identity_from_receipt(probe["doc"])
                    return tagged(
                        PRESENT,
                        value=_compact_resident(
                            ident, source=f"probe:{probe.get('path_taken')}"
                        ),
                        source=probe.get("resolved"),
                    )
                except Exception as parse_exc:
                    return tagged(
                        UNAVAILABLE,
                        reason=(
                            f"resident identity receipt found via {probe.get('path_taken')} "
                            f"but identity_from_receipt raised {type(parse_exc).__name__}: {parse_exc}"
                        ),
                        source=probe.get("resolved"),
                    )
            return tagged(
                UNAVAILABLE,
                reason=(
                    f"resident_identity.collect raised {type(exc).__name__}: {exc}; "
                    f"load raised {type(load_exc).__name__}: {load_exc}; "
                    f"receipt path_taken={probe.get('path_taken')}"
                ),
                source=f"tools.future.resident_identity / receipts/future/{ri.RECEIPT}",
            )
    return tagged(
        PRESENT,
        value=_compact_resident(ident, source="tools.future.resident_identity.collect"),
        source="tools.future.resident_identity.collect",
        machine_genome_nested=True,
    )


def assemble_fallback_identity() -> dict[str, Any]:
    """Flash is the other body. A copy of the resident is not a fallback."""
    try:
        flash = sr.evaluate_flash()
    except Exception as exc:
        return tagged(
            UNAVAILABLE,
            reason=f"super_resident.evaluate_flash raised {type(exc).__name__}: {exc}",
            source="tools.future.super_resident.evaluate_flash",
        )
    if not isinstance(flash, Mapping) or not flash.get("id"):
        return tagged(
            UNAVAILABLE,
            reason="evaluate_flash returned no identity id",
            source="tools.future.super_resident.evaluate_flash",
        )
    unmet = flash.get("unmet_clauses") or []
    compact = {
        "id": str(flash.get("id") or sr.FLASH_ID),
        "role": flash.get("role"),
        "clears_sandbox_floor": flash.get("clears_sandbox_floor"),
        "clears_singularity": flash.get("clears_singularity"),
        "headline": flash.get("headline"),
        "unmet_clauses": list(unmet) if isinstance(unmet, list) else unmet,
        "schema": sr.SCHEMA,
        "receipt": f"receipts/future/{sr.RECEIPT}",
        "source": "tools.future.super_resident.evaluate_flash",
        "gpu_authority": False,
    }
    return tagged(
        PRESENT,
        value=compact,
        source="tools.future.super_resident.evaluate_flash",
    )


def assemble_machine_genome() -> dict[str, Any]:
    """Prior genome identity from resident_identity. Not a remeasurement."""
    try:
        genome = ri._collect_machine_genome()
    except Exception as exc:
        pin = None
        try:
            pin = ol._machine_genome_pin()
        except Exception as pin_exc:
            return tagged(
                UNAVAILABLE,
                reason=(
                    f"resident_identity._collect_machine_genome raised {type(exc).__name__}: {exc}; "
                    f"odyssey_launch._machine_genome_pin raised {type(pin_exc).__name__}: {pin_exc}"
                ),
                source=ri.MACHINE_GENOME_REL,
            )
        if isinstance(pin, Mapping) and pin:
            return tagged(
                PRESENT,
                value={
                    "knowledge_level": pin.get("knowledge_level"),
                    "genome_digest": pin.get("genome_digest"),
                    "path_taken": pin.get("path_taken"),
                    "gpu_authority": False,
                    "source": pin.get("source"),
                    "note": pin.get("note"),
                },
                source="tools.future.odyssey_launch._machine_genome_pin",
            )
        return tagged(
            UNAVAILABLE,
            reason=f"machine genome collectors returned nothing ({type(exc).__name__}: {exc})",
            source=ri.MACHINE_GENOME_REL,
        )
    if not isinstance(genome, Mapping):
        return tagged(
            UNAVAILABLE,
            reason="machine genome collector returned a non-mapping",
            source=ri.MACHINE_GENOME_REL,
        )
    val = genome.get("value") if "value" in genome else genome
    if genome.get("value") == ri.UNKNOWN or (
        isinstance(genome, Mapping) and genome.get("missing_evidence") and val == ri.UNKNOWN
    ):
        missing = genome.get("missing_evidence") or ["qualified machine genome unread"]
        return tagged(
            UNAVAILABLE,
            reason="; ".join(str(m) for m in missing),
            source=ri.MACHINE_GENOME_REL,
        )
    if not isinstance(val, Mapping):
        return tagged(
            UNAVAILABLE,
            reason="machine genome value is not a mapping",
            source=ri.MACHINE_GENOME_REL,
        )
    # Identity of the machine (soc/arch/cores/os/toolchain). Bandwidth stays
    # nested UNKNOWN; this sidecar does not reattest it.
    compact = {
        "schema": val.get("schema"),
        "soc": val.get("soc"),
        "arch": val.get("arch"),
        "os": val.get("os"),
        "os_product": val.get("os_product"),
        "knowledge_level": val.get("knowledge_level"),
        "toolchain_metal_compiler": val.get("toolchain_metal_compiler"),
        "cpu_cores": val.get("cpu_cores"),
        "perf_cores": val.get("perf_cores"),
        "efficiency_cores": val.get("efficiency_cores"),
        "gpu_cores": val.get("gpu_cores"),
        "memory_bytes": val.get("memory_bytes"),
        "source_state": genome.get("source_state"),
        "read_only": True,
        "measure_bandwidth_invoked": False,
        "gpu_authority": False,
        "source": ri.MACHINE_GENOME_REL,
    }
    return tagged(PRESENT, value=compact, source=ri.MACHINE_GENOME_REL)


def _identity_key(slot: Any) -> str | None:
    if not isinstance(slot, Mapping):
        return None
    if slot.get("status") == UNAVAILABLE:
        return None
    val = slot.get("value") if "value" in slot else slot
    if isinstance(val, Mapping):
        key = val.get("id") or val.get("model_id") or val.get("resident_identity")
        return str(key) if key else None
    return str(val) if val not in (None, "", ri.UNKNOWN) else None


# ---------------------------------------------------------------------------
# Source specimens. Verification status is attached or the row is not a source.
# ---------------------------------------------------------------------------


def _verification_index(results: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in results:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("specimen") or "")
        if not name:
            continue
        out[name] = row
        stem = name.split("@", 1)[0]
        out.setdefault(stem, row)
    return out


def _lookup_verification(
    role: Mapping[str, Any], index: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    local = role.get("local_specimen") if isinstance(role.get("local_specimen"), Mapping) else {}
    local_name = local.get("specimen")
    if local_name and local_name in index:
        return index[str(local_name)]
    repo = str(role.get("repo") or "")
    rev = str(role.get("revision") or "")
    slug = repo.replace("/", "--")
    for cand in (
        f"{slug}@{rev[:12]}" if rev else None,
        slug,
        (str((role.get("modellake") or {}).get("specimen_path") or "")).rstrip("/").split("/")[-1],
    ):
        if cand and cand in index:
            return index[cand]
    needle = repo.split("/")[-1].lower()
    if not needle:
        return None
    hits = [row for name, row in index.items() if needle in name.lower()]
    if len(hits) == 1:
        return hits[0]
    return None


def assemble_source_specimens(
    curriculum: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every curriculum role, with verification status attached. Never implied."""
    try:
        cur = dict(curriculum) if curriculum is not None else ol.propose_specimen_curriculum()
    except Exception as exc:
        return tagged(
            UNAVAILABLE,
            reason=f"propose_specimen_curriculum raised {type(exc).__name__}: {exc}",
            source="tools.future.odyssey_launch.propose_specimen_curriculum",
        )
    roles = [r for r in (cur.get("roles") or []) if isinstance(r, Mapping)]
    if not roles:
        # Distinguish "the curriculum function returned no roles" from unread.
        if cur.get("n_roles") == 0:
            return tagged(
                EMPTY,
                reason="curriculum proposes zero roles; none exist to list as sources",
                source="tools.future.odyssey_launch.propose_specimen_curriculum",
            )
        return tagged(
            UNAVAILABLE,
            reason="curriculum roles missing from a document that claims n_roles>0",
            source="tools.future.odyssey_launch.propose_specimen_curriculum",
        )

    probe = ol.probe_json(f"receipts/future/{sv.RECEIPT}")
    verify_doc = probe.get("doc") if isinstance(probe.get("doc"), Mapping) else None
    results = list(verify_doc.get("results") or []) if verify_doc else []
    index = _verification_index(results) if results else {}
    verify_unavailable = not probe.get("found")

    items: list[dict[str, Any]] = []
    for role in roles:
        hit = _lookup_verification(role, index) if index else None
        if hit is not None:
            status = str(hit.get("status") or "NOT_WHOLE_TREE_VERIFIED")
            whole = bool(hit.get("whole_tree_verified"))
            verify_source = f"receipts/future/{sv.RECEIPT}"
            bytes_hashed = hit.get("bytes_hashed")
            specimen_name = hit.get("specimen")
            owner = hit.get("owner")
        elif verify_unavailable:
            status = "VERIFICATION_RECEIPT_UNAVAILABLE"
            whole = False
            verify_source = None
            bytes_hashed = None
            specimen_name = None
            owner = None
        else:
            status = "NOT_IN_VERIFICATION_RECEIPT"
            whole = False
            verify_source = f"receipts/future/{sv.RECEIPT}"
            bytes_hashed = None
            specimen_name = None
            owner = None
        lake = role.get("modellake") if isinstance(role.get("modellake"), Mapping) else {}
        items.append(
            {
                "role": role.get("role"),
                "purpose": role.get("purpose"),
                "repo": role.get("repo"),
                "revision": role.get("revision"),
                "architecture_family": role.get("architecture_family"),
                "curriculum_ready": bool(role.get("ready")),
                "curriculum_ready_reason": role.get("ready_reason"),
                "verification_status": status,
                "whole_tree_verified": whole,
                "verification_source": verify_source,
                "verification_path_taken": probe.get("path_taken"),
                "specimen_name": specimen_name,
                "specimen_owner": owner,
                "bytes_hashed": bytes_hashed if isinstance(bytes_hashed, int) else None,
                "in_specimens_listing": lake.get("in_specimens_listing"),
                "identity_source": role.get("identity_source"),
            }
        )
    return tagged(
        PRESENT,
        items=items,
        source="tools.future.odyssey_launch.propose_specimen_curriculum + specimen_verify",
        verification_receipt={
            "found": bool(probe.get("found")),
            "path_taken": probe.get("path_taken"),
            "resolved": probe.get("resolved"),
        },
        curriculum_n_ready=cur.get("n_ready"),
        curriculum_n_roles=cur.get("n_roles"),
        curriculum_ready=bool(cur.get("ready")),
    )


# ---------------------------------------------------------------------------
# WorkGraphs and Phase II/III listeners. Emitting is not executing.
# ---------------------------------------------------------------------------


def _compact_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    extras = unit.get("extras") if isinstance(unit.get("extras"), Mapping) else {}
    return {
        "id": unit.get("id"),
        "status": unit.get("status"),
        "resource_class": unit.get("resource_class"),
        "stage": unit.get("stage") or extras.get("stage"),
        "odyssey": unit.get("odyssey") or extras.get("odyssey"),
        "dependencies": list(unit.get("dependencies") or []),
        "sleeping": bool(unit.get("sleeping") or extras.get("sleeping") or unit.get("status") == "sleeping"),
        "blocked_reason": unit.get("blocked_reason") or extras.get("blocked_reason"),
        "verifier": unit.get("verifier"),
        "output_receipt_path": unit.get("output_receipt_path") or extras.get("output_receipt_path"),
    }


def assemble_phase_graphs(
    curriculum: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Phase-I graphs plus the II/III listeners that depend on laws, not each other."""
    try:
        cur = dict(curriculum) if curriculum is not None else ol.propose_specimen_curriculum()
        roles = [r for r in (cur.get("roles") or []) if isinstance(r, Mapping)]
    except Exception as exc:
        miss = tagged(
            UNAVAILABLE,
            reason=f"curriculum unread, cannot emit WorkGraphs: {type(exc).__name__}: {exc}",
            source="tools.future.odyssey_launch.propose_specimen_curriculum",
        )
        return miss, miss, miss

    ready_roles = [r for r in roles if r.get("ready")]
    # A graph for an unready specimen would look like a launch of that specimen.
    # odyssey_launch.build drafts the first role regardless; this inventory only
    # names graphs whose specimen role is actually ready.
    if not ready_roles:
        empty = tagged(
            EMPTY,
            reason=(
                "no curriculum role is ready; emitting a Phase-I WorkGraph would "
                "look like a launch of an unready specimen"
            ),
            source="tools.future.odyssey_launch.emit_first_workgraphs",
            n_roles=len(roles),
        )
        return empty, empty, empty

    graph_rows: list[dict[str, Any]] = []
    ii_listeners: list[dict[str, Any]] = []
    iii_listeners: list[dict[str, Any]] = []
    errors: list[str] = []
    policy: Mapping[str, Any] | None = None
    for role in ready_roles:
        try:
            graphs = ol.emit_first_workgraphs(role)
        except Exception as exc:
            errors.append(f"{role.get('repo')}: {type(exc).__name__}: {exc}")
            continue
        units = [_compact_unit(u) for u in (graphs.get("units") or []) if isinstance(u, Mapping)]
        if policy is None and isinstance(graphs.get("phase_listen"), Mapping):
            policy = graphs["phase_listen"]
        graph_rows.append(
            {
                "specimen": graphs.get("specimen"),
                "role": role.get("role"),
                "stages": list(graphs.get("stages") or []),
                "n_units": graphs.get("n_units"),
                "by_stage": dict(graphs.get("by_stage") or {}),
                "units": units,
                "phase_listen": graphs.get("phase_listen"),
                "launchable_specimen": True,
                "note": (
                    "Real HCLI WorkUnits for a curriculum-ready specimen. "
                    "Emitting them is not executing them."
                ),
            }
        )
        for row in graphs.get("listen_units") or []:
            if not isinstance(row, Mapping):
                continue
            unit = next((u for u in units if u.get("id") == row.get("id")), None)
            packed = {
                "odyssey": row.get("odyssey"),
                "id": row.get("id"),
                "depends_on": list(row.get("depends_on") or []),
                "depends_on_sibling": bool(row.get("depends_on_sibling")),
                "specimen": (graphs.get("specimen") or {}).get("repo"),
                "unit": unit,
            }
            if row.get("odyssey") == "II":
                ii_listeners.append(packed)
            elif row.get("odyssey") == "III":
                iii_listeners.append(packed)

    if not graph_rows:
        miss = tagged(
            UNAVAILABLE,
            reason="emit_first_workgraphs failed for every ready role: " + "; ".join(errors),
            source="tools.future.odyssey_launch.emit_first_workgraphs",
        )
        return miss, miss, miss

    graphs_tag = tagged(
        PRESENT,
        items=graph_rows,
        source="tools.future.odyssey_launch.emit_first_workgraphs",
        n_ready_roles=len(ready_roles),
        emit_errors=errors,
    )
    rule = (policy or {}).get("rule")
    barrier = (policy or {}).get("barrier")
    global_barrier = (policy or {}).get("global_barrier")

    def _listeners(odyssey: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return tagged(
                UNAVAILABLE,
                reason=f"Phase {odyssey} listener was not in emit_first_workgraphs output",
                source="tools.future.odyssey_launch.emit_first_workgraphs",
            )
        return tagged(
            PRESENT,
            items=rows,
            value={
                "odyssey": odyssey,
                "n": len(rows),
                "ids": [r.get("id") for r in rows],
                "barrier": barrier,
                "global_barrier": global_barrier,
                "depends_on_sibling": any(r.get("depends_on_sibling") for r in rows),
                "rule": rule,
            },
            source="tools.future.odyssey_launch.phase_listen_policy",
        )

    return graphs_tag, _listeners("II", ii_listeners), _listeners("III", iii_listeners)


# ---------------------------------------------------------------------------
# Resource lanes: imported from the frontier authority, not restated.
# ---------------------------------------------------------------------------


def assemble_resource_lanes() -> dict[str, Any]:
    return tagged(
        PRESENT,
        value={
            "authority": "tools.future.frontiers",
            "authority_receipt": f"receipts/future/{fr.RECEIPT}",
            "this_host": list(fr.THIS_HOST_LANES),
            "cpu_class": list(fr.CPU_LANES),
            "hardware_blocked": list(fr.HARDWARE_LANES),
            "all": list(fr.ALL_LANES),
            "blocked_on_this_host": list(fr.BLOCKED_ON_THIS_HOST),
            "imported": True,
            "restated": False,
        },
        source="tools.future.frontiers.THIS_HOST_LANES/HARDWARE_LANES/ALL_LANES",
    )


def assemble_blocked_triggers() -> dict[str, Any]:
    """SLEEPING work with wake conditions. Never a synthetic result."""
    items: list[dict[str, Any]] = []
    sources: list[str] = []
    try:
        book = fr.load_book()
        for item in book.items:
            if not isinstance(item, Mapping):
                continue
            if item.get("kind") != "BLOCKED" and not fr._item_sleeping(item, book.wake):
                continue
            items.append(
                {
                    "id": item.get("id"),
                    "frontier": item.get("frontier"),
                    "kind": item.get("kind"),
                    "title": item.get("title"),
                    "required_lanes": list(item.get("required_lanes") or []),
                    "wake_all_of": list(item.get("wake_all_of") or []),
                    "wake_never": list(item.get("wake_never") or []),
                    "hypothesis_family": item.get("hypothesis_family"),
                    "source": "tools.future.frontiers.load_book",
                }
            )
        sources.append("tools.future.frontiers.load_book")
    except Exception as exc:
        return tagged(
            UNAVAILABLE,
            reason=f"frontiers.load_book raised {type(exc).__name__}: {exc}",
            source="tools.future.frontiers.load_book",
        )

    try:
        blockers = ol.physical_blockers()
        for b in blockers:
            if not isinstance(b, Mapping) or not b.get("holds"):
                continue
            wu = b.get("workunit") if isinstance(b.get("workunit"), Mapping) else {}
            items.append(
                {
                    "id": b.get("id"),
                    "frontier": "ODYSSEY_TRANSFER",
                    "kind": "BLOCKED",
                    "title": b.get("title"),
                    "required_lanes": list(fr.HARDWARE_LANES),
                    "wake_all_of": [str(b.get("reason") or "hardware qualifies")],
                    "wake_never": [
                        "synthetic result",
                        "lease seizure / flock",
                        "quiesce standing workers",
                    ],
                    "workunit_id": wu.get("id"),
                    "source": "tools.future.odyssey_launch.physical_blockers",
                }
            )
        sources.append("tools.future.odyssey_launch.physical_blockers")
    except Exception as exc:
        # Frontier items already landed; physical blockers failing is a nested
        # gap, not an unread of the whole collection.
        items.append(
            {
                "id": "physical_blockers.unavailable",
                "kind": "BLOCKED",
                "title": "physical_blockers unread",
                "wake_all_of": [],
                "wake_never": ["synthetic result"],
                "reason": f"{type(exc).__name__}: {exc}",
                "source": "tools.future.odyssey_launch.physical_blockers",
            }
        )

    if not items:
        return tagged(
            EMPTY,
            reason="frontier book and physical blockers were read; no BLOCKED trigger holds",
            source="+".join(sources),
        )
    # Stable order so a receipt seal does not thrash on set iteration.
    items.sort(key=lambda r: str(r.get("id") or ""))
    return tagged(PRESENT, items=items, source="+".join(sources))


# ---------------------------------------------------------------------------
# Handles, destinations, backlog, budget, restart.
# ---------------------------------------------------------------------------


def _handle(module: str, receipt_name: str, schema: str, entry: str) -> dict[str, Any]:
    probe = ol.probe_json(f"receipts/future/{receipt_name}")
    rec_tag: dict[str, Any]
    if probe.get("found"):
        rec_tag = tagged(
            PRESENT,
            value={
                "path": f"receipts/future/{receipt_name}",
                "path_taken": probe.get("path_taken"),
                "resolved": probe.get("resolved"),
                "schema": (probe.get("doc") or {}).get("schema") if isinstance(probe.get("doc"), Mapping) else None,
            },
            source=probe.get("resolved"),
        )
    else:
        rec_tag = tagged(
            UNAVAILABLE,
            reason=(
                f"receipts/future/{receipt_name} not in worktree/snapshot/"
                f"primary/HEAD (path_taken={probe.get('path_taken')})"
            ),
            source=f"receipts/future/{receipt_name}",
        )
    return tagged(
        PRESENT,
        value={
            "module": module,
            "entry": entry,
            "schema": schema,
            "receipt_name": receipt_name,
            "receipt": rec_tag,
        },
        source=module,
    )


def assemble_evidence_dag_handle() -> dict[str, Any]:
    return _handle(
        "tools.future.evidence_dag",
        ed.RECEIPT,
        ed.SCHEMA,
        "required_level / admit_candidate / execute_level",
    )


def assemble_negative_science_handle() -> dict[str, Any]:
    return _handle(
        "tools.future.negative_index",
        ni.RECEIPT,
        ni.SCHEMA,
        "query / refuse_if_dead",
    )


def assemble_nr_nx_destinations() -> dict[str, Any]:
    """Declared NR/NX artifact paths, each with a presence probe. Not a claim they ran."""
    rels: list[tuple[str, str]] = [
        (f"receipts/future/{nr.RECEIPT}", "nr_complete_receipt"),
        (f"receipts/future/{nx.RECEIPT}", "nx_audit_receipt"),
    ]
    for name in nr.EVIDENCE.values():
        rels.append((f"receipts/future/evidence/{name}", "nr_evidence_snapshot"))
        rels.append((f"receipts/headless/{name}", "nr_headless"))
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for rel, kind in rels:
        if rel in seen:
            continue
        seen.add(rel)
        probe = ol.probe_json(rel)
        items.append(
            {
                "rel": rel,
                "kind": kind,
                "found": bool(probe.get("found")),
                "path_taken": probe.get("path_taken"),
                "resolved": probe.get("resolved"),
                "declared_by": (
                    "tools.future.flash_nr_complete"
                    if "NR" in kind.upper() or rel.endswith(nr.RECEIPT) or rel.endswith(tuple(nr.EVIDENCE.values()))
                    else "tools.future.flash_nx_audit"
                ),
            }
        )
    if not items:
        return tagged(
            EMPTY,
            reason="no NR/NX destination paths are declared by flash_nr_complete / flash_nx_audit",
            source="tools.future.flash_nr_complete.EVIDENCE",
        )
    return tagged(
        PRESENT,
        items=items,
        source="tools.future.flash_nr_complete.EVIDENCE + flash_nx_audit.RECEIPT",
        n_found=sum(1 for r in items if r.get("found")),
        n_declared=len(items),
    )


def assemble_qualification_backlog() -> dict[str, Any]:
    """Protected qualification candidates. Compact: identity + status, not measurements."""
    loaded_from = None
    try:
        queue = qp.load_qualification_queue()
        loaded_from = queue.get("_loaded_from") if isinstance(queue, Mapping) else None
    except QueueNotFoundError as exc:
        probe = ol.probe_json(
            "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
            "receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
        )
        if not probe.get("found"):
            return tagged(
                UNAVAILABLE,
                reason=(
                    f"qualification queue not found ({exc}); "
                    f"probe path_taken={probe.get('path_taken')}"
                ),
                source="tools.future.qualification_pipeline.load_qualification_queue",
            )
        queue = probe.get("doc")
        loaded_from = probe.get("resolved")
    except Exception as exc:
        return tagged(
            UNAVAILABLE,
            reason=f"load_qualification_queue raised {type(exc).__name__}: {exc}",
            source="tools.future.qualification_pipeline.load_qualification_queue",
        )
    if not isinstance(queue, Mapping):
        return tagged(
            UNAVAILABLE,
            reason="qualification queue loaded but is not a mapping",
            source=str(loaded_from),
        )
    raw = queue.get("candidates")
    if not isinstance(raw, list):
        return tagged(
            UNAVAILABLE,
            reason="qualification queue has no candidates list (unread, not empty)",
            source=str(loaded_from),
        )
    if not raw:
        return tagged(
            EMPTY,
            reason="qualification queue is present and its candidates list is empty",
            source=str(loaded_from),
        )
    items: list[dict[str, Any]] = []
    by_status: dict[str, int] = {}
    for cand in raw:
        if not isinstance(cand, Mapping):
            continue
        status = str(cand.get("status") or "UNKNOWN")
        by_status[status] = by_status.get(status, 0) + 1
        items.append({k: cand.get(k) for k in _QUEUE_KEEP})
    items.sort(key=lambda r: str(r.get("candidate_id") or ""))
    by_status = {k: by_status[k] for k in sorted(by_status)}
    return tagged(
        PRESENT,
        items=items,
        source=str(loaded_from),
        by_status=by_status,
        compact_rule="candidate_id/status/model/blocked_reason/scope_tags only; measurements not copied",
    )


def assemble_time_budget() -> dict[str, Any]:
    """A run budget is a launch parameter. No Odyssey I run has started, so none is on disk."""
    probes = {
        "autonomy_trial": ol.probe_json(f"receipts/future/{at.RECEIPT}"),
        "autonomy_run": ol.probe_json(f"receipts/future/{ar.RECEIPT}"),
        "mission_state": ol.probe_json("receipts/future/AUTONOMY_MISSION_STATE.json"),
    }
    found_any = any(p.get("found") for p in probes.values())
    catalog = {
        "source": "tools.future.autonomy_trial.TRIAL_DURATION_S",
        "ids": list(at.TRIAL_IDS),
        "note": (
            "autonomy-trial durations are a judging catalog, not this run's budget. "
            "Copying one in would invent a launch parameter."
        ),
    }
    if not found_any:
        return tagged(
            UNAVAILABLE,
            reason=(
                "no sealed Odyssey I run has started, so no run time-budget exists on disk; "
                "autonomy-trial catalog is declared policy, not this run"
            ),
            source="receipts/future/AUTONOMY_TRIALS.json + AUTONOMY_RUN.json + AUTONOMY_MISSION_STATE.json",
            declared_trial_catalog=catalog,
            probes={k: {"found": v.get("found"), "path_taken": v.get("path_taken")} for k, v in probes.items()},
        )
    # A trial receipt existing is not this Odyssey I run's budget. Still UNAVAILABLE
    # for the run itself; the probes are cited so the gap is checkable.
    return tagged(
        UNAVAILABLE,
        reason=(
            "autonomy trial/run receipts exist but they are not an Odyssey I "
            "consolidated-run time budget; no launch descriptor has been executed"
        ),
        source="receipts/future/AUTONOMY_TRIALS.json + AUTONOMY_RUN.json",
        declared_trial_catalog=catalog,
        probes={k: {"found": v.get("found"), "path_taken": v.get("path_taken")} for k, v in probes.items()},
        invalidated_runs=list(ar.INVALIDATED_RUNS),
    )


def assemble_restart_state() -> dict[str, Any]:
    """What a restarted supervisor would re-adopt. No Odyssey I run has launched."""
    durable_rel = f"receipts/future/workgraph_workspace/{wg.DURABLE_NAME}"
    durable_path = wg.DEFAULT_WORKSPACE / wg.DURABLE_NAME
    workgraph_slot: dict[str, Any]
    if durable_path.is_file():
        try:
            doc = wg.load_json(durable_path)
        except Exception as exc:
            workgraph_slot = tagged(
                UNAVAILABLE,
                reason=f"workgraph durable unreadable: {type(exc).__name__}: {exc}",
                source=durable_rel,
            )
        else:
            units = doc.get("units") if isinstance(doc, Mapping) else None
            n_units = len(units) if isinstance(units, dict) else (len(units) if isinstance(units, list) else None)
            workgraph_slot = tagged(
                PRESENT,
                value={
                    "schema": doc.get("schema") if isinstance(doc, Mapping) else None,
                    "tick": doc.get("tick") if isinstance(doc, Mapping) else None,
                    "n_units": n_units,
                    "executes": doc.get("executes") if isinstance(doc, Mapping) else None,
                    "odyssey_run": False,
                    "note": (
                        "durable workgraph exists (selftest / scheduler). It is not an "
                        "Odyssey I run restart until a descriptor has launched."
                    ),
                },
                source=durable_rel,
            )
    else:
        probe = ol.probe_json(durable_rel)
        if probe.get("found"):
            workgraph_slot = tagged(
                PRESENT,
                value={
                    "schema": (probe.get("doc") or {}).get("schema") if isinstance(probe.get("doc"), Mapping) else None,
                    "odyssey_run": False,
                    "path_taken": probe.get("path_taken"),
                },
                source=probe.get("resolved"),
            )
        else:
            workgraph_slot = tagged(
                EMPTY,
                reason="no workgraph durable document on disk; nothing to re-adopt",
                source=durable_rel,
            )

    mission = ol.probe_json("receipts/future/AUTONOMY_MISSION_STATE.json")
    if mission.get("found"):
        mission_slot = tagged(
            PRESENT,
            value={
                "path_taken": mission.get("path_taken"),
                "resolved": mission.get("resolved"),
                "odyssey_run": False,
            },
            source=mission.get("resolved"),
        )
    else:
        mission_slot = tagged(
            EMPTY,
            reason="no AUTONOMY_MISSION_STATE.json; no in-flight mission to resume",
            source="receipts/future/AUTONOMY_MISSION_STATE.json",
        )

    detached = ol.probe_json(f"receipts/future/{wg.RECEIPT}".replace(wg.RECEIPT, "DETACHED_EXECUTION.json"))
    # probe_json already records path_taken; EMPTY vs UNAVAILABLE:
    if detached.get("found"):
        detached_slot = tagged(
            PRESENT,
            value={"path_taken": detached.get("path_taken"), "odyssey_run": False},
            source=detached.get("resolved"),
        )
    else:
        detached_slot = tagged(
            EMPTY,
            reason="no DETACHED_EXECUTION.json; no child pid/start-token to re-adopt",
            source="receipts/future/DETACHED_EXECUTION.json",
        )

    odyssey_run = tagged(
        EMPTY,
        reason=(
            "no Odyssey I descriptor has launched; there is no run restart state. "
            "EMPTY (looked, none) is not UNAVAILABLE (could not read)."
        ),
        source="receipts/future/ODYSSEY_CONSOLIDATED_RUN.json",
    )
    return tagged(
        PRESENT,
        value={
            "odyssey_run": odyssey_run,
            "workgraph_durable": workgraph_slot,
            "autonomy_mission": mission_slot,
            "detached": detached_slot,
        },
        source="workgraph durable + autonomy mission + detached receipt",
    )


# ---------------------------------------------------------------------------
# Inventory / descriptor / validator / build_run
# ---------------------------------------------------------------------------


def assemble_inventory(curriculum: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every required field from disk. Not a launch. Fields are tagged."""
    cur = curriculum
    if cur is None:
        try:
            cur = ol.propose_specimen_curriculum()
        except Exception:
            cur = None
    graphs, phase_ii, phase_iii = assemble_phase_graphs(cur)
    return {
        "resident_identity": assemble_resident_identity(),
        "fallback_identity": assemble_fallback_identity(),
        "machine_genome": assemble_machine_genome(),
        "source_specimens": assemble_source_specimens(cur),
        "phase_i_workgraphs": graphs,
        "phase_ii_listener": phase_ii,
        "phase_iii_listener": phase_iii,
        "resource_lanes": assemble_resource_lanes(),
        "evidence_dag_handle": assemble_evidence_dag_handle(),
        "negative_science_handle": assemble_negative_science_handle(),
        "nr_nx_destinations": assemble_nr_nx_destinations(),
        "blocked_triggers": assemble_blocked_triggers(),
        "qualification_backlog": assemble_qualification_backlog(),
        "time_budget": assemble_time_budget(),
        "restart_state": assemble_restart_state(),
    }


def mint_run_id(*, start_time: str, head: str) -> str:
    short = (head or "unknown")[:12]
    stamp = start_time.replace(":", "").replace("-", "")
    return f"odyssey-i-{stamp}-{short}"


def make_descriptor(
    inventory: Mapping[str, Any],
    verdict: Mapping[str, Any],
    *,
    start_time: str | None = None,
    head: str | None = None,
) -> dict[str, Any]:
    """Shape inventory into a run descriptor. Caller must have already passed the gate."""
    ts = start_time or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    git_head = head if head is not None else git("rev-parse", "HEAD")
    desc = {
        "schema": "hawking.future.odyssey_i.run_descriptor.v1",
        "run_id": mint_run_id(start_time=ts, head=git_head or "unknown"),
        "start_time": ts,
        "git_head": git_head or None,
        "gate": {
            "verdict": verdict.get("verdict"),
            "allowed": bool(verdict.get("allowed")),
            "n_met": verdict.get("n_met"),
            "n_unmet": verdict.get("n_unmet"),
            "unmet": list(verdict.get("unmet") or []),
        },
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for field in REQUIRED_FIELDS:
        if field in {"run_id", "start_time"}:
            continue
        desc[field] = inventory.get(field)
    return desc


def validate_descriptor(desc: Any) -> dict[str, Any]:
    """REJECTED or ACCEPTED. A guard nobody has watched fail is not a guard."""
    reasons: list[str] = []
    if not isinstance(desc, Mapping):
        return {
            "status": "REJECTED",
            "reasons": ["<root> is not a mapping"],
            "named_refusal": "REJECTED: <root> is not a mapping",
        }
    for field in REQUIRED_FIELDS:
        if field not in desc:
            reasons.append(f"missing required field {field}")
            continue
        val = desc.get(field)
        if field in TAGGED_COLLECTION_FIELDS:
            if isinstance(val, list):
                reasons.append(
                    f"{field} is a bare list; EMPTY and UNAVAILABLE must not share []"
                )
            elif not is_tagged(val):
                reasons.append(f"{field} is not a tagged collection (PRESENT/EMPTY/UNAVAILABLE)")
            else:
                status = val.get("status")
                if status == UNAVAILABLE and not val.get("reason"):
                    reasons.append(f"{field} is UNAVAILABLE without a reason")
                if status == EMPTY and not val.get("reason"):
                    reasons.append(f"{field} is EMPTY without a reason")
                if status == UNAVAILABLE and val.get("items") not in (None,):
                    reasons.append(f"{field} UNAVAILABLE must not carry an items list")
                if status == EMPTY and val.get("items") not in ([],):
                    reasons.append(f"{field} EMPTY must carry items=[]")
                if status == PRESENT:
                    items = val.get("items")
                    if field == "source_specimens":
                        if not isinstance(items, list):
                            reasons.append("source_specimens PRESENT without items list")
                        else:
                            for i, spec in enumerate(items):
                                if not isinstance(spec, Mapping):
                                    reasons.append(f"source_specimens[{i}] is not a mapping")
                                    continue
                                if "verification_status" not in spec:
                                    reasons.append(
                                        f"source_specimens[{i}] role={spec.get('role')!r} "
                                        "has no verification_status attached"
                                    )

    resident = desc.get("resident_identity")
    fallback = desc.get("fallback_identity")
    rkey = _identity_key(resident)
    fkey = _identity_key(fallback)
    if rkey and fkey and rkey == fkey:
        reasons.append(
            f"fallback identity {fkey!r} is identical to the resident; "
            "identical means there is no fallback"
        )
    if not is_tagged(resident) and "resident_identity" in desc:
        reasons.append("resident_identity is not a tagged field")
    if not is_tagged(fallback) and "fallback_identity" in desc:
        reasons.append("fallback_identity is not a tagged field")

    seen: set[str] = set()
    uniq: list[str] = []
    for item in reasons:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    if uniq:
        return {
            "status": "REJECTED",
            "reasons": uniq,
            "named_refusal": "REJECTED: " + "; ".join(uniq),
        }
    return {"status": "ACCEPTED", "reasons": [], "named_refusal": None}


def accept_descriptor(desc: Any) -> dict[str, Any]:
    result = validate_descriptor(desc)
    if result["status"] == "REJECTED":
        raise DescriptorInvalid(result["reasons"])
    return result


def build_run(
    *,
    criteria: Sequence[Mapping[str, Any]] | None = None,
    inventory: Mapping[str, Any] | None = None,
    start_time: str | None = None,
    head: str | None = None,
) -> dict[str, Any]:
    """The only public path to a run descriptor.

    Consults odyssey_launch's criteria. On any unmet criterion, returns REFUSED
    with the unmet list and descriptor=None. A descriptor is minted only when
    the gate passes, and then only if validate_descriptor accepts it.
    """
    verdict = evaluate_gate(criteria)
    inv = dict(inventory) if inventory is not None else assemble_inventory()
    unmet = list(verdict.get("unmet") or [])
    if unmet or not verdict.get("allowed"):
        return {
            "allowed": False,
            "verdict": "REFUSED",
            "descriptor": None,
            "unmet": unmet,
            "met": list(verdict.get("met") or []),
            "n_met": verdict.get("n_met"),
            "n_unmet": verdict.get("n_unmet"),
            "n_criteria": verdict.get("n_criteria") or len(ol.CRITERION_IDS),
            "reason": (
                "launch gate unmet; emitting a descriptor would look like a launch. "
                "unmet=" + ", ".join(unmet)
            ),
            "inventory": inv,
            "gate": verdict,
        }
    desc = make_descriptor(inv, verdict, start_time=start_time, head=head)
    accept_descriptor(desc)
    return {
        "allowed": True,
        "verdict": "LAUNCH",
        "descriptor": desc,
        "unmet": [],
        "met": list(verdict.get("met") or []),
        "n_met": verdict.get("n_met"),
        "n_unmet": 0,
        "n_criteria": verdict.get("n_criteria") or len(ol.CRITERION_IDS),
        "reason": "all sixteen launch criteria met; descriptor sealed",
        "inventory": inv,
        "gate": verdict,
    }


def _run_workunit(verdict: Mapping[str, Any]) -> dict[str, Any]:
    row = wus.emit_hcli_workunit(
        id="odyssey-i.consolidated-run",
        role="science",
        description="Build the Odyssey I run descriptor or refuse on unmet launch criteria",
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.consolidated_run.build_run",
        provider="future.consolidated_run",
        effect_class="READ_ONLY",
        status="completed",
        classification="REFUSED" if not verdict.get("allowed") else "STATIC_ONLY",
        extras={
            "verdict": verdict.get("verdict"),
            "unmet": list(verdict.get("unmet") or []),
            "blocked_reason": (
                None
                if verdict.get("allowed")
                else "launch criteria unmet; descriptor not emitted"
            ),
            "requires_quiescence": False,
            "output_receipt_path": f"receipts/future/{RECEIPT}",
        },
    )
    wus.validate_emitted_unit(row)
    return row


def recovered_implementation() -> list[str]:
    return [
        "tools/future/odyssey_launch.py — sixteen _eval_* criteria, unmet_criteria, can_launch, emit_first_workgraphs, propose_specimen_curriculum, physical_blockers, probe_json",
        "tools/future/frontiers.py — THIS_HOST_LANES, HARDWARE_LANES, ALL_LANES, load_book blocked items with wake_all_of",
        "tools/future/workgraph.py — durable workgraph.json restart document (not executed)",
        "tools/future/resident_identity.py — collect() incumbent Qwen27 identity, _collect_machine_genome",
        "tools/future/super_resident.py — evaluate_flash() FALLBACK body, FLASH_ID vs QWEN_ID",
        "tools/future/evidence_dag.py — EVIDENCE_DAG.json handle",
        "tools/future/negative_index.py — NEGATIVE_SCIENCE_INDEX.json handle, query/refuse_if_dead",
        "tools/future/specimen_verify.py — SPECIMEN_VERIFICATION.json whole-tree statuses",
        "tools/future/qualification_pipeline.py — load_qualification_queue protected backlog",
        "tools/future/flash_nr_complete.py — EVIDENCE destination map",
        "tools/future/flash_nx_audit.py — FLASH_NX_COMPLETENESS_AUDIT.json",
        "tools/future/autonomy_trial.py — TRIAL_DURATION_S catalog (declared, not this run's budget)",
        "tools/future/autonomy_run.py — INVALIDATED_RUNS, MISSION_STATE path",
        "tools/future/_common.py — write_receipt / HardwareClaimError",
    ]


def gaps_closed() -> list[str]:
    return [
        "No consolidated run descriptor existed; a gate receipt is not a launch.",
        "build_run() consults odyssey_launch criteria and refuses with the full unmet list.",
        "A descriptor is minted only on a passing gate and only after validate_descriptor.",
        "UNAVAILABLE (could not read, with reason) and EMPTY (looked, none exist) cannot share [].",
        "Every curriculum specimen carries verification_status; unverified rows cannot pass as sources.",
        "Fallback identity identical to the resident is REJECTED (there is no fallback).",
    ]


def negative_findings_from(attempt: Mapping[str, Any]) -> list[str]:
    findings = [
        f"verdict={attempt.get('verdict')} n_unmet={attempt.get('n_unmet')} unmet={list(attempt.get('unmet') or [])}",
        "A descriptor emitted on an unmet gate would be a document that looks like a launch.",
        "orchestration.BINDINGS does not yet name consolidated_run.py; that table is outside this lane's write set.",
        "time_budget for this run is UNAVAILABLE: no Odyssey I launch has started, and copying an autonomy-trial duration would invent a parameter.",
        "restart_state.odyssey_run is EMPTY: there is no in-flight Odyssey I run to re-adopt.",
    ]
    inv = attempt.get("inventory") if isinstance(attempt.get("inventory"), Mapping) else {}
    specs = inv.get("source_specimens") if isinstance(inv.get("source_specimens"), Mapping) else {}
    for spec in specs.get("items") or []:
        if not isinstance(spec, Mapping):
            continue
        if spec.get("whole_tree_verified") is not True:
            findings.append(
                f"curriculum role {spec.get('role')} verification_status={spec.get('verification_status')} "
                f"ready={spec.get('curriculum_ready')}"
            )
    fb = inv.get("fallback_identity") if isinstance(inv.get("fallback_identity"), Mapping) else {}
    if is_tagged(fb) and fb.get("status") == PRESENT:
        val = fb.get("value") if isinstance(fb.get("value"), Mapping) else {}
        if val.get("clears_sandbox_floor") is not True:
            findings.append(
                f"fallback {val.get('id')} does not clear SANDBOX_RESIDENT_FLOOR "
                f"(role={val.get('role')})"
            )
    return findings


def resident_callable_block(attempt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_point": "tools.future.consolidated_run.build_run()",
        "workunit": (
            "one CPU_ANALYSIS unit; odyssey-i.consolidated-run; "
            "refuses rather than emitting a launch-shaped descriptor"
        ),
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier": "FT.ODYSSEY_TRANSFER.re-earn",
        "fails_closed": (
            "build_run returns descriptor=None while any odyssey_launch criterion is unmet; "
            "validate_descriptor rejects a missing required field, a bare list where a tagged "
            "collection is required, a specimen without verification_status, and a fallback "
            "identical to the resident; write_receipt raises HardwareClaimError on numeric "
            "hardware fields; UNAVAILABLE never silently becomes []."
        ),
        "verdict_now": attempt.get("verdict"),
        "unmet_now": list(attempt.get("unmet") or []),
    }


def build() -> Path:
    attempt = build_run()
    wu = _run_workunit(attempt)
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Run descriptor the detached Odyssey supervisor will execute, or an "
            "honest refusal naming every unmet launch criterion."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "phase_transition": "NOT_STARTED",
        "descriptor_emitted": bool(attempt.get("descriptor")),
        "allowed": bool(attempt.get("allowed")),
        "verdict": attempt.get("verdict"),
        "unmet": list(attempt.get("unmet") or []),
        "met": list(attempt.get("met") or []),
        "n_met": attempt.get("n_met"),
        "n_unmet": attempt.get("n_unmet"),
        "n_criteria": attempt.get("n_criteria"),
        "reason": attempt.get("reason"),
        "descriptor": attempt.get("descriptor"),
        "disk_inventory": attempt.get("inventory"),
        "disk_inventory_is_not_a_launch": True,
        "gate": {
            k: attempt.get("gate", {}).get(k)
            for k in ("verdict", "allowed", "unmet", "met", "n_met", "n_unmet", "n_criteria", "rule")
        } if isinstance(attempt.get("gate"), Mapping) else None,
        "workunit": wu,
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings_from(attempt),
        "resident_callable": resident_callable_block(attempt),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
