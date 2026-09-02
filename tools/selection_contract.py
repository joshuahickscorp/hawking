#!/usr/bin/env python3
"""Model-agnostic resident selection contract.

States: CANDIDATE -> SHADOW -> QUALIFIED -> PROMOTED, with ROLLED_BACK restoring
the previously bound identity. Machine binding is a hardware wake class (the
same ids tools.roadmap.hardware.WAKE_CONDITIONS speaks), never a vendor
checkpoint name.

This module records decisions. It does not install, start, or replace a
resident process. `installed` is always False.

Comparison is delegated to tools.pareto_table (identity, metrics, qualify,
dominates, profile, provenance). Hard gates are delegated to
tools.successor_select.gate so G151's refusals stay the authority.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from tools.pareto_table import (
    COMPARISON_AXES,
    Axis,
    Qualification,
    candidate_identity,
    dominates,
    metrics_of,
    pareto_front,
    provenance_of,
    qualify as qualify_metrics,
)
from tools.successor_select import Refused, gate as successor_gate

SCHEMA = "hawking.selection.resident_contract.v1"
STATES = ("CANDIDATE", "SHADOW", "QUALIFIED", "PROMOTED", "ROLLED_BACK")
HARD_FLAGS = ("doctor_pass", "provenance_valid", "native_path", "no_hidden_fallback")

# Machine classes the binder understands. These are hardware/wake ids, not
# bodies. Keep in lockstep with tools.roadmap.hardware.WAKE_CONDITIONS keys
# plus the domains already present on this host.
MACHINE_CLASSES = (
    "UMA",
    "ANE",
    "CPU",
    "GPU",
    "U50_PRESENT",
    "DGX_PRESENT",
    "NEW_M_SERIES_PRESENT",
    "HMF_PRESENT",
    "EGPU_PRESENT",
)


class SelectionRefused(ValueError):
    """The contract refused to move state."""


def _require_state(record: Mapping[str, Any], allowed: Sequence[str]) -> str:
    state = str(record.get("state") or "")
    if state not in allowed:
        raise SelectionRefused(
            f"state {state!r} is not one of {tuple(allowed)}"
        )
    return state


def _identity_of(raw: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(raw, str):
        return candidate_identity(raw).as_dict()
    if "candidate_id" in raw:
        return candidate_identity(
            str(raw["candidate_id"]),
            artifact_digest=raw.get("artifact_digest"),
            artifact_path=raw.get("artifact_path"),
            machine_class=raw.get("machine_class"),
        ).as_dict()
    if "identity" in raw and isinstance(raw["identity"], Mapping):
        return _identity_of(raw["identity"])
    raise SelectionRefused("record has no candidate identity")


def _hard_vector(record: Mapping[str, Any]) -> dict[str, Any] | None:
    flags = record.get("flags") or {}
    if not isinstance(flags, Mapping):
        return None
    if not all(k in flags for k in HARD_FLAGS):
        return None
    ident = _identity_of(record)
    metrics = record.get("metrics") or {}
    return {
        "name": ident["candidate_id"],
        "bpw": metrics.get("effective_bpw") or metrics.get("bpw") or 0,
        "token_ns": metrics.get("token_ns") or 0,
        **{k: bool(flags.get(k)) for k in HARD_FLAGS},
    }


def admit_candidate(
    identity: Mapping[str, Any] | str,
    metrics: Mapping[str, Any] | None = None,
    *,
    provenance: Mapping[str, Any] | None = None,
    flags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Admit a body as a candidate. Not resident. Not installed."""
    ident = _identity_of(identity)
    mets = metrics_of(metrics or {})
    prov = provenance if isinstance(provenance, Mapping) else provenance_of().as_dict()
    return {
        "schema": SCHEMA,
        "state": "CANDIDATE",
        "identity": ident,
        "metrics": dict(mets.values),
        "metric_sources": dict(mets.sources),
        "provenance": dict(prov),
        "flags": dict(flags or {}),
        "machine_binding": None,
        "installed": False,
        "resident": False,
    }


def bind_machine(
    record: Mapping[str, Any],
    *,
    machine_class: str,
    wake_condition: str | None = None,
    present: bool = False,
    evidence_tier: str = "STATIC",
) -> dict[str, Any]:
    """Bind a candidate to a machine class. Does not start anything.

    `wake_condition` is the hardware id the daemon should activate when the
    device arrives. `present=False` is the honest default on this host for
    FPGA/DGX/eGPU; that is a model of absence, not a measurement of those
    boards.
    """
    if machine_class not in MACHINE_CLASSES:
        raise SelectionRefused(
            f"machine_class {machine_class!r} is not one of {MACHINE_CLASSES}"
        )
    wake = wake_condition or (
        machine_class if machine_class.endswith("_PRESENT") else None
    )
    out = dict(record)
    out["machine_binding"] = {
        "machine_class": machine_class,
        "wake_condition": wake,
        "present": bool(present),
        "evidence_tier": evidence_tier,
    }
    ident = dict(out.get("identity") or {})
    ident["machine_class"] = machine_class
    out["identity"] = ident
    out["installed"] = False
    return out


def bind_shadow(record: Mapping[str, Any], *, machine_class: str,
                present: bool = False) -> dict[str, Any]:
    """Shadow: bound to a machine, running beside the resident, not promoted."""
    _require_state(record, ("CANDIDATE", "SHADOW", "QUALIFIED"))
    out = bind_machine(record, machine_class=machine_class, present=present)
    out["state"] = "SHADOW"
    out["resident"] = False
    out["installed"] = False
    return out


def qualify(
    record: Mapping[str, Any],
    floors: Mapping[str, Any],
    *,
    axes: Sequence[Axis] | None = None,
) -> dict[str, Any]:
    """Qualification is fail-closed. Does not promote.

    Calls tools.pareto_table.qualify for metric floors and, when the G151 hard
    flags are present, tools.successor_select.gate so a broken body cannot be
    marked QUALIFIED. A QUALIFIED record emits child_qualified to the existing
    lifecycle router (not a second bus).
    """
    _require_state(record, ("CANDIDATE", "SHADOW", "QUALIFIED"))
    q: Qualification = qualify_metrics(
        record.get("metrics") or {},
        floors,
        flags=record.get("flags"),
        axes=axes,
    )
    out = dict(record)
    out["qualification"] = q.as_dict()
    out["installed"] = False
    if not q.passed:
        out["state"] = record.get("state") or "CANDIDATE"
        out["qualification_refused"] = True
        return out

    hard = _hard_vector(out)
    if hard is not None:
        try:
            successor_gate(hard)
        except Refused as exc:
            out["state"] = record.get("state") or "CANDIDATE"
            out["qualification"] = {
                **q.as_dict(),
                "passed": False,
                "failures": list(q.failures) + [str(exc)],
            }
            out["qualification_refused"] = True
            return out

    out["state"] = "QUALIFIED"
    out["qualification_refused"] = False
    # Live consumer of child_qualified: existing successor gate already ran;
    # the lifecycle router records the event and is the call site the audit
    # table cites. Lazy import avoids a cycle with lifecycle_events.
    from tools.lifecycle_events import on_child_qualified
    on_child_qualified(out)
    return out


def promote(
    record: Mapping[str, Any],
    parent: Mapping[str, Any] | None = None,
    *,
    axes: Sequence[Axis] | None = None,
) -> dict[str, Any]:
    """Promote a QUALIFIED record to the resident *decision*. Never installs.

    If a parent is supplied, the child must Pareto-dominate it (or the parent
    must be incomparable and the child qualified). A dominated child is refused.
    """
    _require_state(record, ("QUALIFIED",))
    if parent is not None:
        child_m = record.get("metrics") or {}
        parent_m = parent.get("metrics") or parent
        if dominates(parent_m, child_m, axes or COMPARISON_AXES):
            raise SelectionRefused(
                f"{_identity_of(record)['candidate_id']} is dominated by parent "
                f"{_identity_of(parent)['candidate_id']}; refusing promotion"
            )
    out = dict(record)
    out["state"] = "PROMOTED"
    out["resident"] = True
    out["installed"] = False
    out["note"] = (
        "decision record only; this contract never installs or starts a resident"
    )
    return out


def rollback(
    current: Mapping[str, Any],
    previous: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore the previously bound identity. Does not install the previous body."""
    prev_ident = _identity_of(previous)
    cur_ident = _identity_of(current)
    restored = dict(previous)
    restored["state"] = "ROLLED_BACK"
    restored["resident"] = False
    restored["installed"] = False
    restored["rolled_back_from"] = cur_ident
    restored["identity"] = prev_ident
    restored["note"] = (
        "rollback restored the previous identity in the ledger; it did not "
        "start or stop a process"
    )
    return restored


def decide_from_table(
    table: Mapping[str, Mapping[str, Any]],
    *,
    floors: Mapping[str, Any] | None = None,
    axes: Sequence[Axis] | None = None,
    machine_class: str = "UMA",
) -> dict[str, Any]:
    """Turn a Pareto table into a selection decision. Production caller:
    tools.pareto_table.main. Never installs.

    Default floors are empty: a G150 table with mostly-null cells must not be
    silently qualified. Callers that have a capability floor pass it in.
    """
    used_axes = tuple(axes) if axes is not None else COMPARISON_AXES
    used_floors = dict(floors or {})
    admitted: dict[str, dict[str, Any]] = {}
    refused: dict[str, Any] = {}
    for cid, row in table.items():
        rec = admit_candidate(cid, row)
        rec = bind_shadow(rec, machine_class=machine_class, present=False)
        rec = qualify(rec, used_floors, axes=used_axes)
        if rec.get("state") == "QUALIFIED":
            admitted[cid] = rec
        else:
            refused[cid] = rec.get("qualification") or {"passed": False}

    front = pareto_front(
        {cid: rec["metrics"] for cid, rec in admitted.items()},
        used_axes,
    ) if admitted else []

    selected = None
    selected_record = None
    if len(front) == 1:
        selected = front[0]
        selected_record = promote(admitted[selected], axes=used_axes)
    elif len(front) > 1:
        # Front is not a singleton: do not break the tie with a missing axis.
        # Leave the decision at QUALIFIED / no promotion.
        selected = None
        selected_record = None

    return {
        "schema": SCHEMA,
        "state": selected_record["state"] if selected_record else (
            "QUALIFIED" if admitted else "CANDIDATE"
        ),
        "selected": selected,
        "installed": False,
        "resident": bool(selected_record and selected_record.get("resident")),
        "pareto_front": front,
        "admitted": sorted(admitted),
        "refused": {k: v for k, v in refused.items()},
        "record": selected_record,
        "machine_class": machine_class,
        "floors": used_floors,
        "note": (
            "decision record only; promotion here does not install a resident"
        ),
    }
