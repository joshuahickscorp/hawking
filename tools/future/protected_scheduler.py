"""PROTECTED_SCHEDULER — decide protected work without a window.

A busy GPU and a missing lease make the WINDOW unavailable. They do not make
the SCHEDULER incapable. `_eval_protected_scheduling` currently ANDs
`contamination == QUIESCENT` and `gpu_authority` into invoke/schedule/frontier/
refill, so a HEAVY machine with no proven holder looks like "the resident cannot
handle protected work". That is the same category error as stamping
`BLOCKED_NO_METAL_GPU` on an M3 Ultra.

This module EXTENDS the landed inspectors (`protected_window.read_protected_locks`,
`qualification_pipeline.read_hcli_lease_state`, `CONTAMINATION_SCIENCE.json`)
with a decision path that is demonstrable without opening a window:

* `recognize(unit)` — protected-required by declared `resource_class` only
* `inspect_contamination()` — class from the real receipt, never a coerced QUIESCENT
* `inspect_lease()` — holder via lsof, never flock, never a fabricated present
* `decide(unit)` — RUNNABLE | BLOCKED_ON_PROTECTED_WINDOW | REFUSED
* `park(unit)` — blocked protected units only, with the wake condition attached
* `continue_with()` — CPU-lane work from `frontiers.next_work`, not idle
* `capability_report()` — CAPABLE and AVAILABLE as separate fields

It does not take a GPU lease, does not flock a bench lock, does not invent a
lease so a test can pass, and does not report the scheduler incapable because
the window is busy. It cannot establish that a real HCLI runner would execute
the unit: this sidecar has no GPU authority even when the window is open.

    python3 tools/future/protected_scheduler.py --build
    python3 -m pytest tools/future/test_protected_scheduler.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.future._common import REPO, write_receipt
from tools.future import contamination as C
from tools.future import frontiers as fr
from tools.future import protected_window as pw
from tools.future import qualification_pipeline as qp
from tools.future import workunit_species as ws

RECEIPT = "PROTECTED_SCHEDULER.json"
SCHEMA = "hawking.future.protected_scheduler.v1"
VERSION = 1
RECORDED_BY = "tools/future/protected_scheduler.py"

ERAS = qp.ERAS
ODYSSEYS = qp.ODYSSEYS

CONTAMINATION_RECEIPT_REL = Path("receipts") / "future" / C.RECEIPT

# Declared exclusive class. GPU_DIRTY_OK / GPU_DECODE run dirty; they are not
# this scheduler's protected set. MUTATION is never grantable.
PROTECTED_RESOURCE_CLASS = "GPU_EXCLUSIVE"
FORBIDDEN_RESOURCE_CLASS = "MUTATION"

VERDICTS = ("RUNNABLE", "BLOCKED_ON_PROTECTED_WINDOW", "REFUSED")

WAKE_ALL_OF: tuple[str, ...] = (
    "contamination_class == QUIESCENT (CONTAMINATION_SCIENCE receipt, never coerced)",
    "existing HCLI protected lease with a proven holder pid (read, never flock)",
)
WAKE_NEVER: tuple[str, ...] = (
    "fabricated or simulated lease presented as live",
    "fcntl.flock / lockf / os.O_EXCL on a bench lock",
    "quiesce standing workers to forge a quiet machine",
    "synthetic QUIESCENT",
    "mark the scheduler incapable because the window is busy",
)

PROBE_UNIT: dict[str, Any] = {
    "id": "future.protected-scheduler.probe",
    "role": "science",
    "description": (
        "Probe unit: declared GPU_EXCLUSIVE so recognize() must classify it as "
        "protected-required. Never executed. Never a synthetic protected result."
    ),
    "resource_class": PROTECTED_RESOURCE_CLASS,
    "requires_quiescence": True,
    "verifier": "future.protected_scheduler.decide",
    "provider": "future.protected_scheduler",
}

# `_eval_protected_scheduling` today. Named so the receipt can say exactly
# which keys that criterion would have to stop conflating. Not imported.
ODYSSEY_LAUNCH_CRITERION = "_eval_protected_scheduling"
ODYSSEY_LAUNCH_PATH = "tools/future/odyssey_launch.py"


class SchedulerRefused(qp.ExecuteRefused):
    """A named refusal the tests can watch. Never a quiet default."""


def refuse_flock(*_a: Any, **_k: Any) -> None:
    """Named guard: taking flock is a seizure even when a lock file exists."""
    raise qp.AuthorityBoundaryError("flock")


def refuse_create_lease(*_a: Any, **_k: Any) -> None:
    qp.refuse_create_lease()


def refuse_start_benchmark(*_a: Any, **_k: Any) -> None:
    qp.refuse_start_benchmark()


def acquire_lease(*_a: Any, **_k: Any) -> None:
    """There is no path that takes the lock. Raises rather than flock."""
    raise SchedulerRefused(
        "lease_seizure",
        "sidecar will not fcntl.LOCK_EX / flock / lockf / O_EXCL a bench lock; "
        "inspect_lease is a READ. A missing holder is present=false, not a reason "
        "to fabricate one",
    )


def seize_lease(*_a: Any, **_k: Any) -> None:
    acquire_lease()


def _authority(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = dict(qp.AUTHORITY_REFUSAL)
    if extra:
        out.update(dict(extra))
    return out


def _as_mapping(unit: Any) -> dict[str, Any] | None:
    if isinstance(unit, Mapping):
        return dict(unit)
    to_dict = getattr(unit, "to_dict", None)
    if callable(to_dict):
        got = to_dict()
        if isinstance(got, Mapping):
            return dict(got)
    return None


def _declared_resource_class(unit: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Exact HCLI class or a refusal. Do not use normalize_resource_class.

    That helper maps unknown strings to LIGHT_CONTROL. Mapping a missing or
    invented class onto a real one is a guess, and recognize() refuses to guess.
    """
    raw = unit.get("resource_class")
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return None, "resource_class is absent; refuse rather than guess from the unit id or title"
    text = str(raw).strip()
    if text not in ws.KNOWN_RESOURCE:
        return None, (
            f"resource_class {text!r} is not a declared HCLI class; "
            "refuse rather than map unknown -> LIGHT_CONTROL"
        )
    return text, None


# ---------------------------------------------------------------------------
# recognize — declared class, never a name heuristic
# ---------------------------------------------------------------------------


def recognize(unit: Any) -> dict[str, Any]:
    """Is this a protected-required WorkUnit? By declared resource class only."""
    body = _as_mapping(unit)
    if body is None:
        return _authority(
            {
                "kind": "RECOGNIZE",
                "recognized": False,
                "protected_required": False,
                "resource_class": None,
                "requires_quiescence": False,
                "guessed": False,
                "reason": (
                    f"unit is {type(unit).__name__}, not a mapping with a declared "
                    "resource_class; refuse rather than infer protection from the name"
                ),
            }
        )
    rc, why = _declared_resource_class(body)
    if rc is None:
        return _authority(
            {
                "kind": "RECOGNIZE",
                "recognized": False,
                "protected_required": False,
                "resource_class": None,
                "requires_quiescence": False,
                "guessed": False,
                "unit_id": body.get("id"),
                "reason": why,
            }
        )
    if rc == FORBIDDEN_RESOURCE_CLASS:
        return _authority(
            {
                "kind": "RECOGNIZE",
                "recognized": False,
                "protected_required": False,
                "resource_class": rc,
                "requires_quiescence": False,
                "guessed": False,
                "unit_id": body.get("id"),
                "reason": "MUTATION is not grantable to a sidecar unit; refuse",
            }
        )
    # Exact True only. A string "true" is not a declaration we will honour as bool.
    declared_quiescence = body.get("requires_quiescence") is True
    protected = rc == PROTECTED_RESOURCE_CLASS or declared_quiescence
    if protected:
        reason = (
            f"declared resource_class={rc!r}"
            + (" and requires_quiescence=True" if declared_quiescence else "")
            + "; protection is a declaration, not a name match"
        )
    else:
        reason = (
            f"declared resource_class={rc!r} is not {PROTECTED_RESOURCE_CLASS} "
            "and requires_quiescence is not True; not a protected-required unit"
        )
    return _authority(
        {
            "kind": "RECOGNIZE",
            "recognized": True,
            "protected_required": protected,
            "resource_class": rc,
            "requires_quiescence": declared_quiescence or rc == PROTECTED_RESOURCE_CLASS,
            "guessed": False,
            "unit_id": body.get("id"),
            "reason": reason,
        }
    )


# ---------------------------------------------------------------------------
# inspect_contamination — the receipt, not a fresh optimistic probe
# ---------------------------------------------------------------------------


def inspect_contamination(*, injected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Current class from the real receipt. Injected values are test INPUTS only."""
    if injected is not None:
        klass = injected.get("contamination_class")
        if klass not in C.CONTAMINATION_CLASSES:
            return _authority(
                {
                    "kind": "READ",
                    "contamination_class": "UNKNOWN",
                    "contamination_reason": (
                        f"injected contamination_class {klass!r} is not in "
                        f"{list(C.CONTAMINATION_CLASSES)}; UNKNOWN, never optimistic QUIESCENT"
                    ),
                    "source": "injected_input",
                    "live": False,
                    "injected_input": True,
                    "coerced": False,
                }
            )
        return _authority(
            {
                "kind": "READ",
                "contamination_class": klass,
                "contamination_reason": injected.get("contamination_reason")
                or f"injected input {klass}",
                "contamination_evidence": list(injected.get("contamination_evidence") or []),
                "source": "injected_input",
                "live": False,
                "injected_input": True,
                "coerced": False,
            }
        )
    doc, src = pw.load_visible_json(CONTAMINATION_RECEIPT_REL)
    if not isinstance(doc, Mapping):
        return _authority(
            {
                "kind": "READ",
                "contamination_class": "UNKNOWN",
                "contamination_reason": (
                    f"{CONTAMINATION_RECEIPT_REL.as_posix()} not visible ({src}); "
                    "UNKNOWN, never optimistic QUIESCENT"
                ),
                "source": src,
                "live": True,
                "injected_input": False,
                "coerced": False,
                "receipt_found": False,
            }
        )
    klass = doc.get("contamination_class")
    if klass not in C.CONTAMINATION_CLASSES:
        return _authority(
            {
                "kind": "READ",
                "contamination_class": "UNKNOWN",
                "contamination_reason": (
                    f"receipt class {klass!r} is not in {list(C.CONTAMINATION_CLASSES)}; "
                    "UNKNOWN, never optimistic QUIESCENT"
                ),
                "source": src,
                "live": True,
                "injected_input": False,
                "coerced": False,
                "receipt_found": True,
            }
        )
    return _authority(
        {
            "kind": "READ",
            "contamination_class": klass,
            "contamination_reason": doc.get("contamination_reason"),
            "contamination_evidence": list(doc.get("contamination_evidence") or []),
            "source": src,
            "live": True,
            "injected_input": False,
            "coerced": False,
            "receipt_found": True,
            "receipt_schema": doc.get("schema"),
        }
    )


# ---------------------------------------------------------------------------
# inspect_lease — READ the existing probe. Never take, never invent.
# ---------------------------------------------------------------------------


def _holders_from(lease: Mapping[str, Any]) -> list[int]:
    holders = lease.get("holders")
    if isinstance(holders, Mapping):
        pids = holders.get("pids") or []
        return [int(p) for p in pids if str(p).lstrip("-").isdigit()]
    observations = lease.get("observations") or []
    pids: list[int] = []
    if isinstance(observations, list):
        for obs in observations:
            if not isinstance(obs, Mapping):
                continue
            inner = obs.get("holders") if isinstance(obs.get("holders"), Mapping) else {}
            for p in inner.get("pids") or []:
                if str(p).lstrip("-").isdigit():
                    pids.append(int(p))
    primary = lease.get("primary_hcli_lock")
    if isinstance(primary, Mapping) and isinstance(primary.get("holders"), Mapping):
        for p in primary["holders"].get("pids") or []:
            if str(p).lstrip("-").isdigit():
                pids.append(int(p))
    return sorted(set(pids))


def inspect_lease(*, injected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Lease state and holder, without taking it.

    Injected values are decide() INPUTS for tests. They never create, flock,
    or write a lock file, and they are marked live=False so they cannot be
    mistaken for a real holder.
    """
    if injected is not None:
        claimed = bool(injected.get("present"))
        pids = _holders_from(injected)
        # present is earned by a proven holder, even on an injected input.
        present = claimed and bool(pids)
        if claimed and not pids:
            reason = (
                "injected present=true without proven holder pids; "
                "fail closed: present=false. a flag is not a lease"
            )
        elif present:
            reason = (
                f"injected input: present with pids {pids}. "
                "TEST INPUT only; no lock file was written and no flock was taken"
            )
        else:
            reason = injected.get("reason") or "injected input: no proven holder"
        return _authority(
            {
                "kind": "READ",
                "present": present,
                "lock_file_exists": bool(injected.get("lock_file_exists")),
                "holders": {"status": "OK" if pids else "SKIPPED", "pids": pids},
                "acquired": False,
                "would_flock": False,
                "fabricated": False,
                "touched_lock_file": False,
                "injected_input": True,
                "live": False,
                "probe": "injected_input; no lock path was opened",
                "reason": reason,
            }
        )
    raw = pw.read_protected_locks()
    # present is the composed inspector's fail-closed verdict. Do not invent a
    # holder, and do not override a proven one by re-parsing pids.
    present = bool(raw.get("present"))
    pids = _holders_from(raw)
    exists = any(
        bool(o.get("lock_file_exists"))
        for o in (raw.get("observations") or [])
        if isinstance(o, Mapping)
    )
    primary = raw.get("primary_hcli_lock") if isinstance(raw.get("primary_hcli_lock"), Mapping) else {}
    exists = exists or bool(primary.get("lock_file_exists"))
    return _authority(
        {
            "kind": "READ",
            "present": present,
            "lock_file_exists": exists,
            "holders": {"status": "OK" if pids else (primary.get("holders") or {}).get("status") or "SKIPPED",
                        "pids": pids},
            "acquired": False,
            "would_flock": False,
            "fabricated": False,
            "touched_lock_file": False,
            "injected_input": False,
            "live": True,
            "probe": raw.get("probe"),
            "composed_from": "tools.future.protected_window.read_protected_locks",
            "not_called": list(raw.get("not_called") or [
                "hcli.agentos.protected_accelerator_benchmark._try_lock",
                "fcntl.flock",
                "fcntl.LOCK_EX",
            ]),
            "lock_rels": list(raw.get("lock_rels") or pw.DEFAULT_LOCK_RELS),
            "observations": list(raw.get("observations") or []),
            "primary_hcli_lock": primary,
            "reason": raw.get("reason"),
        }
    )


def window_available(contamination: Mapping[str, Any], lease: Mapping[str, Any]) -> bool:
    """Clean window AND a proven lease. This sidecar's gpu_authority is not an input.

    Folding gpu_authority into availability would make the window permanently
    closed for this process even on a quiet machine holding a real lease.
    """
    return (
        contamination.get("contamination_class") == "QUIESCENT"
        and bool(lease.get("present"))
    )


# ---------------------------------------------------------------------------
# decide / park / continue_with
# ---------------------------------------------------------------------------


def _wake() -> dict[str, Any]:
    return {"all_of": list(WAKE_ALL_OF), "never": list(WAKE_NEVER)}


def decide(
    unit: Any,
    *,
    contamination: Mapping[str, Any] | None = None,
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """RUNNABLE | BLOCKED_ON_PROTECTED_WINDOW | REFUSED.

    Capability is a property of this path, not of the window. A blocked
    protected unit still proves the scheduler can decide for the right reason.
    """
    rec = recognize(unit)
    if contamination is None:
        cont = inspect_contamination()
    elif (
        isinstance(contamination, Mapping)
        and contamination.get("kind") == "READ"
        and "contamination_class" in contamination
    ):
        cont = dict(contamination)
    else:
        cont = inspect_contamination(injected=contamination)
    if lease is None:
        lease_st = inspect_lease()
    elif (
        isinstance(lease, Mapping)
        and lease.get("kind") == "READ"
        and "present" in lease
        and "acquired" in lease
    ):
        lease_st = dict(lease)
    else:
        lease_st = inspect_lease(injected=lease)

    available = window_available(cont, lease_st)
    klass = cont.get("contamination_class")
    # The scheduler ran the path. A busy window is not incapability.
    capable = True

    if not rec.get("recognized"):
        return _authority(
            {
                "kind": "DECISION",
                "verdict": "REFUSED",
                "reason": rec.get("reason"),
                "wake_condition": None,
                "protected_required": False,
                "recognized": False,
                "contamination_class": klass,
                "lease_present": bool(lease_st.get("present")),
                "window_available": available,
                "scheduler_capable": capable,
                "inputs_simulated": bool(
                    cont.get("injected_input") or lease_st.get("injected_input")
                ),
            }
        )

    if not rec.get("protected_required"):
        return _authority(
            {
                "kind": "DECISION",
                "verdict": "RUNNABLE",
                "reason": (
                    f"{rec.get('reason')}; a unit that does not need protection "
                    "is not gated on the protected window"
                ),
                "wake_condition": None,
                "protected_required": False,
                "recognized": True,
                "resource_class": rec.get("resource_class"),
                "contamination_class": klass,
                "lease_present": bool(lease_st.get("present")),
                "window_available": available,
                "scheduler_capable": capable,
                "unit_id": rec.get("unit_id"),
                "inputs_simulated": bool(
                    cont.get("injected_input") or lease_st.get("injected_input")
                ),
            }
        )

    if available:
        verdict = "RUNNABLE"
        reason = (
            "contamination_class is QUIESCENT and a proven HCLI holder is present; "
            "the protected window is available. This sidecar still has no GPU "
            "authority and will not execute the unit"
        )
        wake = None
    else:
        verdict = "BLOCKED_ON_PROTECTED_WINDOW"
        bits = []
        if klass != "QUIESCENT":
            bits.append(
                f"contamination_class={klass!r} (needs QUIESCENT; "
                f"{cont.get('contamination_reason') or 'see receipt'})"
            )
        if not lease_st.get("present"):
            bits.append(
                f"no proven HCLI lease ({lease_st.get('reason') or 'present=false'})"
            )
        reason = (
            "protected-required unit cannot start: "
            + "; ".join(bits)
            + ". scheduler_capable remains true; the window is what is missing"
        )
        wake = _wake()

    return _authority(
        {
            "kind": "DECISION",
            "verdict": verdict,
            "reason": reason,
            "wake_condition": wake,
            "protected_required": True,
            "recognized": True,
            "resource_class": rec.get("resource_class"),
            "contamination_class": klass,
            "lease_present": bool(lease_st.get("present")),
            "window_available": available,
            "scheduler_capable": capable,
            "unit_id": rec.get("unit_id"),
            "inputs_simulated": bool(
                cont.get("injected_input") or lease_st.get("injected_input")
            ),
        }
    )


def park(
    unit: Any,
    *,
    contamination: Mapping[str, Any] | None = None,
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach the wake condition. A unit that does not need protection is not parked."""
    rec = recognize(unit)
    decision = decide(unit, contamination=contamination, lease=lease)
    if not rec.get("protected_required"):
        return _authority(
            {
                "kind": "PARK",
                "parked": False,
                "reason": (
                    "unit does not need protection; parking it would stall work "
                    "that is not gated on the protected window"
                ),
                "verdict": decision.get("verdict"),
                "wake_condition": None,
                "unit_id": rec.get("unit_id"),
                "protected_required": False,
            }
        )
    if decision.get("verdict") != "BLOCKED_ON_PROTECTED_WINDOW":
        return _authority(
            {
                "kind": "PARK",
                "parked": False,
                "reason": (
                    f"verdict is {decision.get('verdict')!r}; only "
                    "BLOCKED_ON_PROTECTED_WINDOW is parked"
                ),
                "verdict": decision.get("verdict"),
                "wake_condition": None,
                "unit_id": rec.get("unit_id"),
                "protected_required": True,
            }
        )
    body = _as_mapping(unit) or {}
    parked_unit = {
        "id": body.get("id") or rec.get("unit_id"),
        "resource_class": rec.get("resource_class"),
        "requires_quiescence": rec.get("requires_quiescence"),
        "status": "blocked",
        "classification": "SLEEPING",
        "blocked_reason": decision.get("reason"),
        "wake_condition": decision.get("wake_condition"),
        "claim_boundary": ws.SIDECAR_CLAIM_BOUNDARY,
    }
    return _authority(
        {
            "kind": "PARK",
            "parked": True,
            "reason": decision.get("reason"),
            "verdict": "BLOCKED_ON_PROTECTED_WINDOW",
            "wake_condition": decision.get("wake_condition"),
            "unit": parked_unit,
            "unit_id": parked_unit.get("id"),
            "protected_required": True,
            "scheduler_capable": True,
            "window_available": False,
        }
    )


def _compact_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": unit.get("id"),
        "resource_class": unit.get("resource_class"),
        "requires_quiescence": unit.get("requires_quiescence"),
        "status": unit.get("status"),
        "classification": unit.get("classification"),
        "frontier": unit.get("frontier"),
    }


def continue_with(*, excluding: set[str] | None = None) -> dict[str, Any]:
    """Unrelated CPU-class work the resident does instead of waiting on the window."""
    skip = set(excluding or ())
    try:
        raw = fr.next_work(fr.THIS_HOST_LANES)
    except Exception as exc:
        return _authority(
            {
                "kind": "CONTINUE",
                "units": [],
                "n": 0,
                "lanes": list(fr.THIS_HOST_LANES),
                "source": "tools.future.frontiers.next_work(THIS_HOST_LANES)",
                "failed_closed": True,
                "reason": (
                    f"frontiers.next_work refused ({type(exc).__name__}: {exc}); "
                    "empty continue, never invented work"
                ),
            }
        )
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for unit in raw:
        if not isinstance(unit, Mapping):
            continue
        ident = str(unit.get("id") or "")
        rc = str(unit.get("resource_class") or "")
        if ident in skip:
            dropped.append(ident)
            continue
        if rc == PROTECTED_RESOURCE_CLASS or unit.get("requires_quiescence") is True:
            dropped.append(ident)
            continue
        kept.append(_compact_unit(unit))
    return _authority(
        {
            "kind": "CONTINUE",
            "units": kept,
            "n": len(kept),
            "n_dropped_protected": len(dropped),
            "dropped_ids": dropped,
            "lanes": list(fr.THIS_HOST_LANES),
            "excluded_resource_classes": [PROTECTED_RESOURCE_CLASS],
            "source": "tools.future.frontiers.next_work(THIS_HOST_LANES)",
            "failed_closed": False,
            "reason": (
                "protected unit is parked; resident continues with CPU-class work "
                "that does not need a protected window"
            ),
        }
    )


def drive(
    unit: Any | None = None,
    *,
    contamination: Mapping[str, Any] | None = None,
    lease: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run recognize → inspect → decide → park → continue_with as one proof."""
    probe = unit if unit is not None else dict(PROBE_UNIT)
    rec = recognize(probe)
    cont = (
        inspect_contamination(injected=contamination)
        if contamination is not None
        else inspect_contamination()
    )
    lease_st = inspect_lease(injected=lease) if lease is not None else inspect_lease()
    decision = decide(probe, contamination=cont, lease=lease_st)
    parked = park(probe, contamination=cont, lease=lease_st)
    continued = continue_with(excluding={str((_as_mapping(probe) or {}).get("id") or "")})
    return _authority(
        {
            "kind": "DRIVE",
            "recognize": rec,
            "contamination": {
                "contamination_class": cont.get("contamination_class"),
                "contamination_reason": cont.get("contamination_reason"),
                "source": cont.get("source"),
                "live": cont.get("live"),
                "injected_input": cont.get("injected_input"),
            },
            "lease": {
                "present": lease_st.get("present"),
                "lock_file_exists": lease_st.get("lock_file_exists"),
                "holders": lease_st.get("holders"),
                "acquired": lease_st.get("acquired"),
                "would_flock": lease_st.get("would_flock"),
                "fabricated": lease_st.get("fabricated"),
                "touched_lock_file": lease_st.get("touched_lock_file"),
                "live": lease_st.get("live"),
                "injected_input": lease_st.get("injected_input"),
                "reason": lease_st.get("reason"),
            },
            "decide": decision,
            "park": parked,
            "continue_with": {
                "n": continued.get("n"),
                "failed_closed": continued.get("failed_closed"),
                "unit_ids": [u.get("id") for u in (continued.get("units") or [])],
                "n_dropped_protected": continued.get("n_dropped_protected"),
                "reason": continued.get("reason"),
            },
        }
    )


def capability_report() -> dict[str, Any]:
    """CAPABLE and AVAILABLE as separate fields. Live inputs only.

    This is the shape `_eval_protected_scheduling` would have to read. That
    module is not in this lane's WRITE list, so the criterion is not edited;
    the keys and today's honest values are recorded here.
    """
    live = drive()
    decision = live.get("decide") or {}
    cont = live.get("contamination") or {}
    lease_st = live.get("lease") or {}
    available = bool(decision.get("window_available"))
    capable = bool(decision.get("scheduler_capable")) and decision.get("verdict") in VERDICTS
    # A blocked window must not flip capable off. That is the whole module.
    if decision.get("verdict") == "BLOCKED_ON_PROTECTED_WINDOW" and not available:
        capable = True
    if decision.get("verdict") == "RUNNABLE" and decision.get("protected_required") and available:
        capable = True
    return _authority(
        {
            "kind": "CAPABILITY",
            "PROTECTED_SCHEDULER_CAPABLE": capable,
            "PROTECTED_WINDOW_AVAILABLE": available,
            "contamination_class": cont.get("contamination_class"),
            "lease_present": bool(lease_st.get("present")),
            "lease_lock_file_exists": bool(lease_st.get("lock_file_exists")),
            "lease_holders": (lease_st.get("holders") or {}).get("pids"),
            "live_verdict": decision.get("verdict"),
            "live_reason": decision.get("reason"),
            "parked": bool((live.get("park") or {}).get("parked")),
            "continue_with_n": (live.get("continue_with") or {}).get("n"),
            "did_not_fabricate_lease": lease_st.get("fabricated") is False,
            "did_not_flock": lease_st.get("would_flock") is False and lease_st.get("acquired") is False,
            "did_not_touch_lock_file": lease_st.get("touched_lock_file") is False,
            "did_not_mark_incapable_because_window_unavailable": capable is True,
            "drive": live,
            "odyssey_launch_read_plan": {
                "do_not_edit": ODYSSEY_LAUNCH_PATH,
                "current_criterion": ODYSSEY_LAUNCH_CRITERION,
                "current_conflation": (
                    "lease_ok = (contamination_class == QUIESCENT) and gpu_auth; "
                    "invoke/schedule/frontier/refill are all ANDed with lease_ok, "
                    "so a HEAVY machine with no proven holder makes the scheduler "
                    "look non-operational"
                ),
                "should_read": {
                    "PROTECTED_SCHEDULER_CAPABLE": (
                        "tools.future.protected_scheduler.capability_report()"
                        "['PROTECTED_SCHEDULER_CAPABLE']"
                    ),
                    "PROTECTED_WINDOW_AVAILABLE": (
                        "tools.future.protected_scheduler.capability_report()"
                        "['PROTECTED_WINDOW_AVAILABLE']"
                    ),
                },
                "how_to_use": (
                    "discover/invoke/schedule/verify/frontier/persist/refill for "
                    "'can the resident HANDLE protected work' AND with CAPABLE, "
                    "never with AVAILABLE. Record AVAILABLE as its own field. "
                    "'can protected work START right now' is CAPABLE and AVAILABLE"
                ),
                "honest_values_today": {
                    "PROTECTED_SCHEDULER_CAPABLE": capable,
                    "PROTECTED_WINDOW_AVAILABLE": available,
                    "contamination_class": cont.get("contamination_class"),
                    "lease_present": bool(lease_st.get("present")),
                    "gpu_authority": False,
                    "live_verdict": decision.get("verdict"),
                },
            },
            "category_error_refused": (
                "PROTECTED_SCHEDULER_CAPABLE is not derived from "
                "PROTECTED_WINDOW_AVAILABLE. A busy window is "
                "BLOCKED_ON_PROTECTED_WINDOW, not scheduler-incapable"
            ),
        }
    )


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def recovered_implementation() -> list[dict[str, Any]]:
    return [
        {
            "path": "tools/future/protected_window.py",
            "role": (
                "eviction envelope; read_protected_locks / load_visible_json / "
                "refuse_flock / acquire_lease raise rather than seize"
            ),
            "composed_as": "inspect_lease delegates to read_protected_locks; load_visible_json finds the contamination receipt across checkout roots",
            "adequate_for": "observing an existing HCLI lock without flocking it; planning a window this sidecar will not open",
            "not_adequate_for": "separating scheduler capability from window availability; decide/park/continue_with",
            "extended_not_forked": True,
        },
        {
            "path": "tools/future/qualification_pipeline.py",
            "role": "read_hcli_lease_state; AUTHORITY_REFUSAL; ExecuteRefused / AuthorityBoundaryError",
            "composed_as": "inspect_lease consumes the primary_hcli_lock field already attached by read_protected_locks; refuse_* guards are reused",
            "extended_not_forked": True,
        },
        {
            "path": "tools/future/contamination.py",
            "role": "QUIESCENT/LIGHT/HEAVY/UNKNOWN taxonomy; CONTAMINATION_SCIENCE.json",
            "composed_as": "inspect_contamination reads that receipt; it does not re-probe and does not coerce QUIESCENT",
            "extended_not_forked": True,
        },
        {
            "path": "tools/future/frontiers.py",
            "role": "THIS_HOST_LANES / next_work — CPU work while GPU_PROTECTED sleeps",
            "composed_as": "continue_with() is next_work(THIS_HOST_LANES) with GPU_EXCLUSIVE dropped",
            "extended_not_forked": True,
        },
        {
            "path": "tools/future/workunit_species.py",
            "role": "HCLI resource_class set; requires_quiescence derived from GPU_EXCLUSIVE",
            "composed_as": "recognize() matches against KNOWN_RESOURCE exactly; it does not call normalize_resource_class because that maps unknown -> LIGHT_CONTROL",
        },
        {
            "path": "tools/future/odyssey_launch.py",
            "role": "_eval_protected_scheduling — the criterion this lane must satisfy honestly",
            "composed_as": "cited, not imported, not edited (not in WRITE list). capability_report() is the shape it would need to read",
        },
        {
            "path": "tools/odyssey/protected_window.py",
            "role": "G013 crash-safe pause/resume lease at /tmp/hawking_protected_window.lease",
            "composed_as": "READ ONLY. Different lease (SIGSTOP of downloaders) than the HCLI bench lock. Not taken, not healed, not written",
            "on_disk_in_this_worktree": (REPO / "tools/odyssey/protected_window.py").is_file(),
        },
        {
            "path": "hcli/resources.py",
            "role": "ResourceClass including GPU_EXCLUSIVE / GPU_DIRTY_OK / GPU_DECODE",
            "composed_as": "declared-class vocabulary; MUTATION refused",
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "PROTECTED_SCHEDULER_CAPABLE and PROTECTED_WINDOW_AVAILABLE are separate fields",
        "decide() returns BLOCKED_ON_PROTECTED_WINDOW on a HEAVY / no-lease host without marking the scheduler incapable",
        "decide() returns RUNNABLE when contamination QUIESCENT and a proven holder are supplied as INPUTS, without writing a lease file",
        "recognize() uses declared resource_class only; a unit named 'protected-*' with STATIC_ANALYSIS is not protected-required",
        "park() refuses to park a unit that does not need protection",
        "continue_with() is frontiers.next_work(THIS_HOST_LANES), not idle and not a GPU_EXCLUSIVE substitute",
        "inspect_lease composes protected_window.read_protected_locks; flock / lockf / O_EXCL / _try_lock are never called",
        "capability_report() is the shape _eval_protected_scheduling would need to read; odyssey_launch.py is not edited",
    ]


def negative_findings() -> list[str]:
    return [
        "this sidecar has no GPU authority and cannot produce DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE even when decide() returns RUNNABLE",
        "odyssey_launch.py is not in this lane's WRITE list; _eval_protected_scheduling still ANDs lease_ok into invoke/schedule/frontier/refill",
        "orchestration.py BINDINGS is not in the WRITE list; this module is unbound there until a later lane adds it",
        "inspect_contamination does not re-snapshot the machine; a stale receipt would be believed. Staleness is a recorded source, not a coerced class",
        "an empty .hcli/locks/*.lock file is lock_file_exists=true and present=false; lsof without a pid is not a lease",
        "tools/odyssey/protected_window.py's /tmp lease is a downloader SIGSTOP record, not the HCLI bench lock; this scheduler does not heal or write it",
        "RUNNABLE does not mean this process will execute the unit; it means the scheduler would allow it given window inputs",
        "normalize_resource_class maps unknown classes to LIGHT_CONTROL; recognize() refuses unknown classes instead of inheriting that default",
    ]


def build() -> Path:
    report = capability_report()
    live = report.get("drive") or {}
    decision = live.get("decide") or {}
    unit = ws.emit_hcli_workunit(
        id="future.protected-scheduler.capability",
        role="science",
        description=(
            "Decide protected-required WorkUnits without opening a window. "
            "Park them with a wake condition and continue with CPU-class work."
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.protected_scheduler.capability_report",
        provider="future.protected_scheduler",
        effect_class="READ_ONLY",
        status="pending",
        classification="STATIC_ONLY",
        extras={
            "species": "accelerator_candidate_qualification",
            "sleeping": False,
            "requires_quiescence": False,
            "claim_boundary": ws.SIDECAR_CLAIM_BOUNDARY,
            "PROTECTED_SCHEDULER_CAPABLE": report.get("PROTECTED_SCHEDULER_CAPABLE"),
            "PROTECTED_WINDOW_AVAILABLE": report.get("PROTECTED_WINDOW_AVAILABLE"),
        },
    )
    ws.validate_emitted_unit(unit)
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Separate protected-scheduler capability from protected-window "
            "availability so a busy GPU cannot be misread as 'the resident "
            "cannot handle protected work'."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "vocabulary": {
            "PROTECTED_SCHEDULER_CAPABLE": (
                "the resident can recognize, inspect, decide, park, and continue; "
                "proven by driving that path against real contamination and real lease state"
            ),
            "PROTECTED_WINDOW_AVAILABLE": (
                "contamination_class == QUIESCENT AND a proven HCLI holder pid, "
                "observed read-only. This sidecar's gpu_authority is not an input"
            ),
            "RUNNABLE": "the scheduler would allow the unit given current (or injected) window inputs",
            "BLOCKED_ON_PROTECTED_WINDOW": "protected-required, and the window is not available",
            "REFUSED": "undeclared or forbidden resource_class; no guess from the unit name",
        },
        "capability": {
            "PROTECTED_SCHEDULER_CAPABLE": report.get("PROTECTED_SCHEDULER_CAPABLE"),
            "PROTECTED_WINDOW_AVAILABLE": report.get("PROTECTED_WINDOW_AVAILABLE"),
            "contamination_class": report.get("contamination_class"),
            "lease_present": report.get("lease_present"),
            "live_verdict": report.get("live_verdict"),
            "live_reason": report.get("live_reason"),
            "parked": report.get("parked"),
            "continue_with_n": report.get("continue_with_n"),
        },
        "odyssey_launch_read_plan": report.get("odyssey_launch_read_plan"),
        "category_error_refused": report.get("category_error_refused"),
        "did_not_fabricate_lease": report.get("did_not_fabricate_lease"),
        "did_not_flock": report.get("did_not_flock"),
        "did_not_touch_lock_file": report.get("did_not_touch_lock_file"),
        "drive": live,
        "authority_boundary": dict(qp.AUTHORITY_REFUSAL),
        "never": [
            "fabricate or simulate a live lease to make anything pass",
            "flock, lockf, O_EXCL, or otherwise contend for a bench lock",
            "report the scheduler incapable merely because the window is unavailable",
            "park a unit that does not need protection",
            "guess protected-required from the unit id or title",
        ],
        "workunit": {
            "id": unit.get("id"),
            "resource_class": unit.get("resource_class"),
            "requires_quiescence": unit.get("requires_quiescence"),
            "status": unit.get("status"),
            "classification": unit.get("classification"),
            "verifier": unit.get("verifier"),
        },
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "resident_callable": {
            "entry_point": "tools.future.protected_scheduler.capability_report()",
            "workunit": (
                "one CPU_ANALYSIS unit; recognize/decide/park/continue_with against "
                f"declared {PROTECTED_RESOURCE_CLASS} without opening a window"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.GPU_KERNELS.ready-protected",
            "fails_closed": (
                "absent contamination receipt -> UNKNOWN, never QUIESCENT; "
                "lock file without proven holder -> present=false; "
                "unknown resource_class -> REFUSED, not LIGHT_CONTROL; "
                "acquire_lease/refuse_flock raise rather than flock"
            ),
            "functions": {
                "recognize": "recognize(unit) -> dict",
                "inspect_contamination": "inspect_contamination(*, injected=None) -> dict",
                "inspect_lease": "inspect_lease(*, injected=None) -> dict  # READ; never flock",
                "decide": "decide(unit, *, contamination=None, lease=None) -> dict",
                "park": "park(unit, *, contamination=None, lease=None) -> dict",
                "continue_with": "continue_with(*, excluding=None) -> dict",
                "capability_report": "capability_report() -> dict  # live inputs only",
            },
        },
        "live_decision_today": {
            "verdict": decision.get("verdict"),
            "scheduler_capable": decision.get("scheduler_capable"),
            "window_available": decision.get("window_available"),
            "contamination_class": (live.get("contamination") or {}).get("contamination_class"),
            "lease_present": (live.get("lease") or {}).get("present"),
        },
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
