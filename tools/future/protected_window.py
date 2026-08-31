"""PROTECTED_WINDOW — the resident removes itself from the resource path.

Protected evidence outranks resident convenience. This sidecar plans the
sequence that checkpoints the resident, frees the GPU, runs Codex's staged
qualification, and restores the mission. It is structurally incapable of
seizing a lease it does not hold.

The 13-stage qualification sequencer already exists
(`tools/future/qualification_pipeline.py`). This module EXTENDS it with the
resident-eviction envelope; it does not fork those stages.

    python3 tools/future/protected_window.py --dry-run
    python3 tools/future/protected_window.py --build
    python3 -m pytest tools/future/test_protected_window.py -q

Everything emitted here is STATIC_ONLY, bench state UNKNOWN, gpu_authority
false. This module produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.future import candidate_planner as cp
from tools.future import contamination as C
from tools.future import qualification_pipeline as qp
from tools.future import repro_science as rs
from tools.future import workunit_species as ws
from tools.future._common import HARDWARE_FIELDS, git

RECEIPT = "PROTECTED_WINDOW_PLAN.json"
SCHEMA = "hawking.future.protected_window.v1"
VERSION = 1
RECORDED_BY = "tools.future.protected_window.py"
CHECKPOINT_SCHEMA = "hawking.future.protected_window.checkpoint.v1"

ERAS = qp.ERAS
ODYSSEYS = qp.ODYSSEYS

# Envelope around the landed 13-stage sequencer. Order is the contract sequence.
WINDOW_STAGES: tuple[str, ...] = (
    "estimate_qualification_value",
    "decide_eviction",
    "checkpoint_resident",
    "freeze_safe_work",
    "pause_unload_resident",
    "establish_protected_lease",
    "run_staged_qualification",
    "record_protected_receipts",
    "restore_resident",
    "resume_mission",
)

# Occupancy ledger. UNLOADED_STATES must restore or the mission is stalled.
RESIDENT_STATES: tuple[str, ...] = (
    "LOADED",
    "CHECKPOINTED",
    "FROZEN",
    "UNLOADED",
    "LEASE_PENDING",
    "QUALIFYING",
    "RECEIPTS_PENDING",
    "RESTORED",
    "RESUMED",
)
UNLOADED_STATES = frozenset(
    {"UNLOADED", "LEASE_PENDING", "QUALIFYING", "RECEIPTS_PENDING"}
)
NEEDS_ROLLBACK_STATES = UNLOADED_STATES | frozenset({"FROZEN"})

DEFAULT_LOCK_RELS: tuple[str, ...] = (
    qp.HCLI_LOCK_REL.as_posix(),
    ".hcli/locks/qwen-protected-bench.lock",
)

STAGED_PLAN_REL = Path("receipts") / "future" / "CANDIDATE_STAGED_PLAN.json"
HANDOFF_REL = Path("CODEX_ACCELERATOR_HANDOFF.json")
NX_AUDIT_REL = Path("receipts") / "future" / "FLASH_NX_COMPLETENESS_AUDIT.json"
FRONTIER_PATH = qp.FRONTIER_PATH
QUAL_RECEIPT_REL = Path("receipts") / "future" / "QUALIFICATION_PIPELINE.json"

DIRTY_RESOURCE_CLASSES = frozenset({"GPU_DIRTY_OK", "GPU_DECODE"})
MEASUREMENT_DIRTY = frozenset({"DIAGNOSTIC_RELATIVE", "STATIC_ONLY"})

# Local DirtyWin schema. Concurrent sibling dirty_measure.py is the swap target.
DIRTY_WIN_FIELDS = (
    "id",
    "candidate_id",
    "decision_target",
    "measurement_class",
    "would_change_decision_if_protected",
    "apparent_outcome",
)

AUTHORITY_REFUSAL = qp.AUTHORITY_REFUSAL


class WindowRefused(qp.ExecuteRefused):
    """execute() / acquire_lease() refused. Named so tests can watch each guard."""


class WindowInterrupted(qp.PipelineInterrupted):
    """Fault-injected interruption. Resume or rollback; do not leave the resident unloaded."""


# ---------------------------------------------------------------------------
# Structural refusals. A guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------


def refuse_start_benchmark(*_a: Any, **_k: Any) -> None:
    qp.refuse_start_benchmark()


def refuse_create_lease(*_a: Any, **_k: Any) -> None:
    qp.refuse_create_lease()


def refuse_signal_process(*_a: Any, **_k: Any) -> None:
    qp.refuse_signal_process()


def refuse_quiesce_worker(*_a: Any, **_k: Any) -> None:
    qp.refuse_quiesce_worker()


def refuse_flock(*_a: Any, **_k: Any) -> None:
    """Named guard: taking flock is a seizure even when a lock file exists."""
    raise qp.AuthorityBoundaryError("flock")


def acquire_lease(*_a: Any, **_k: Any) -> None:
    """There is no path that takes the lock. Raises rather than flock."""
    raise WindowRefused(
        "lease_seizure",
        "sidecar will not fcntl.LOCK_EX / flock the HCLI lock; flock would be a seizure. "
        "queue_policy.protected_start_requires_existing_hcli_lease",
    )


def seize_lease(*_a: Any, **_k: Any) -> None:
    acquire_lease()


def _authority(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = dict(AUTHORITY_REFUSAL)
    if extra:
        out.update(dict(extra))
    return out


def _strip_hardware(node: Any) -> Any:
    """Copy a mapping without hardware-measurement fields. Counts of decisions stay."""
    if isinstance(node, Mapping):
        return {
            k: _strip_hardware(v)
            for k, v in node.items()
            if k not in HARDWARE_FIELDS
        }
    if isinstance(node, list):
        return [_strip_hardware(v) for v in node]
    return node


# ---------------------------------------------------------------------------
# Checkout roots. A missing path in this sparse tree is not campaign-absence.
# ---------------------------------------------------------------------------


def _checkout_roots() -> list[Path]:
    roots: list[Path] = [REPO]
    common = git("rev-parse", "--git-common-dir")
    if common:
        path = Path(common)
        if not path.is_absolute():
            path = (REPO / path).resolve()
        else:
            path = path.resolve()
        parent = path.parent if path.name == ".git" else path.parent
        if parent not in roots:
            roots.append(parent)
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def load_visible_json(rel: str | Path) -> tuple[dict[str, Any] | None, str]:
    """Load a JSON document from this worktree, the primary checkout, or git.

    Records which path it took. Never treats invisibility as proof of absence.
    """
    rel_s = Path(rel).as_posix()
    for root in _checkout_roots():
        path = root / rel_s
        if path.is_file():
            try:
                doc = load_json(path)
            except (OSError, json.JSONDecodeError, UnicodeError) as exc:
                return None, f"unreadable:{path}:{type(exc).__name__}"
            if isinstance(doc, dict):
                return doc, str(path)
    blob = git("show", f"HEAD:{rel_s}")
    if blob:
        try:
            doc = json.loads(blob)
        except json.JSONDecodeError:
            return None, f"git:HEAD:{rel_s}:not_json"
        if isinstance(doc, dict):
            return doc, f"git:HEAD:{rel_s}"
    return None, f"not_visible:{rel_s}"


# ---------------------------------------------------------------------------
# Batch order — Codex's staged protected batch, never a fresh guess
# ---------------------------------------------------------------------------


def _ids_from_run_order(raw: Any) -> list[str]:
    ids: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item:
                ids.append(item)
            elif isinstance(item, Mapping):
                cid = item.get("candidate_id") or item.get("cell_id")
                if cid:
                    ids.append(str(cid))
    return ids


def _cells_from(raw: Any) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return cells
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        cid = str(item.get("cell_id") or "")
        members = [str(m) for m in (item.get("candidates") or []) if m]
        if not cid and members:
            cid = "__".join(members)
        if not cid:
            continue
        cells.append(
            {
                "cell_id": cid,
                "kind": item.get("kind"),
                "stage": item.get("stage"),
                "candidates": members,
                "requires_survivors": [
                    str(s) for s in (item.get("requires_survivors") or []) if s
                ],
            }
        )
    return cells


def _normalize_batch(doc: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Accept either the Codex handoff shape or CANDIDATE_STAGED_PLAN.protected_batch."""
    if "protected_batch" in doc:
        pb = doc.get("protected_batch") or {}
    elif "qwen27_first_batch" in doc or "qwen_first_batch" in doc:
        pb = doc
    else:
        pb = doc.get("current_staged_protected_batch") or doc

    qwen = pb.get("qwen27_first_batch") or pb.get("qwen_first_batch") or {}
    flash = pb.get("flash_return_batch") or {}
    qwen_ids = _ids_from_run_order(
        qwen.get("run_order") or qwen.get("singleton_order") or []
    )
    flash_ids = _ids_from_run_order(
        flash.get("run_order") or flash.get("singleton_order") or []
    )
    after = (
        qwen.get("dependency_cells_after_singletons")
        or (qwen.get("after_singletons") or {}).get("predicted_interaction_and_union_cells")
        or []
    )
    cells = _cells_from(after)
    lock_rels = list(DEFAULT_LOCK_RELS)
    env = pb.get("current_environment") or {}
    env_locks = env.get("protected_lock_paths")
    if isinstance(env_locks, list) and env_locks:
        lock_rels = [str(p) for p in env_locks if p]
    status = str(pb.get("status") or qwen.get("execution_state") or "WAITING_FOR_AUTHORITY")
    return {
        "source": source,
        "schema": pb.get("schema") or doc.get("schema"),
        "status": status,
        "qwen_batch_name": qwen.get("name") or "QWEN_FIRST_PROTECTED_SINGLES",
        "flash_batch_name": flash.get("name") or "FLASH_RETURN_PROTECTED_SINGLETONS_AND_COMPOSITIONS",
        "qwen_singleton_order": qwen_ids,
        "qwen_cells_after": cells,
        "flash_singleton_order": flash_ids,
        "lock_rels": lock_rels,
        "n_qwen_singletons": len(qwen_ids),
        "n_qwen_cells_after": len(cells),
        "n_flash_singletons": len(flash_ids),
        "flash_runnable_now": False,
        "claim_boundary": (
            "batch order copied from Codex/staged-plan disk state; "
            "this sidecar does not invent a new order"
        ),
    }


def load_staged_batch(
    injected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load Codex's staged protected batch. Prefer the handoff, fall back to the sealed plan.

    Both sources may be invisible in a sparse worktree. The path taken is recorded.
    """
    if injected is not None:
        if "qwen_singleton_order" in injected:
            out = dict(injected)
            out.setdefault("source", "injected")
            out.setdefault("flash_runnable_now", False)
            out.setdefault("lock_rels", list(DEFAULT_LOCK_RELS))
            out.setdefault("n_qwen_singletons", len(list(out.get("qwen_singleton_order") or [])))
            out.setdefault("n_qwen_cells_after", len(list(out.get("qwen_cells_after") or [])))
            out.setdefault("n_flash_singletons", len(list(out.get("flash_singleton_order") or [])))
            return out
        return _normalize_batch(injected, "injected")

    handoff, handoff_from = load_visible_json(HANDOFF_REL)
    if isinstance(handoff, Mapping):
        nested = handoff.get("current_staged_protected_batch")
        if isinstance(nested, Mapping):
            batch = _normalize_batch(nested, f"codex_handoff:{handoff_from}#current_staged_protected_batch")
            batch["handoff_visible"] = True
            batch["handoff_from"] = handoff_from
            nx = flash_authority_state()
            batch["flash_runnable_now"] = bool(nx.get("runnable"))
            batch["flash_authority"] = nx
            return batch

    plan, plan_from = load_visible_json(STAGED_PLAN_REL)
    if isinstance(plan, Mapping) and plan.get("protected_batch"):
        batch = _normalize_batch(plan, f"staged_plan:{plan_from}#protected_batch")
        batch["handoff_visible"] = handoff is not None
        batch["handoff_from"] = handoff_from
        nx = flash_authority_state()
        batch["flash_runnable_now"] = bool(nx.get("runnable"))
        batch["flash_authority"] = nx
        return batch

    # Last resort: compose candidate_planner from a visible queue. Still not a guess
    # of identity — the planner already encodes Codex's named order.
    try:
        queue = qp.load_qualification_queue()
        staged = cp.plan_from_queue(queue)
        plan_doc = {"protected_batch": cp.protected_batch_plan(queue, staged.get("staged_factorial_plan") or staged, [])}
        batch = _normalize_batch(plan_doc, "composed:candidate_planner.protected_batch_plan")
        nx = flash_authority_state()
        batch["flash_runnable_now"] = bool(nx.get("runnable"))
        batch["flash_authority"] = nx
        batch["handoff_visible"] = handoff is not None
        batch["handoff_from"] = handoff_from
        return batch
    except Exception as exc:
        return {
            "source": f"unavailable:{type(exc).__name__}",
            "qwen_singleton_order": [],
            "qwen_cells_after": [],
            "flash_singleton_order": [],
            "lock_rels": list(DEFAULT_LOCK_RELS),
            "n_qwen_singletons": 0,
            "n_qwen_cells_after": 0,
            "n_flash_singletons": 0,
            "flash_runnable_now": False,
            "status": "UNAVAILABLE",
            "handoff_from": handoff_from,
            "plan_from": plan_from,
            "reason": str(exc)[:240],
        }


def flash_authority_state() -> dict[str, Any]:
    """Flash rows stay closed until source-independent NX is qualified. Fail closed."""
    doc, src = load_visible_json(NX_AUDIT_REL)
    if not isinstance(doc, Mapping):
        return {
            "runnable": False,
            "reason": "Flash NX audit not visible; fail closed: not qualified",
            "source": src,
        }
    seven = doc.get("seven_all_met")
    meta = doc.get("meta_measurement_state") if isinstance(doc.get("meta_measurement_state"), Mapping) else {}
    promotion = meta.get("promotion_allowed")
    runnable = seven is True and promotion is True
    return {
        "runnable": False if not runnable else True,
        "seven_all_met": seven,
        "promotion_allowed": promotion,
        "source": src,
        "reason": (
            "Flash source-independent NX is not qualified; return batch stays closed"
            if not runnable
            else "Flash NX audit reports all requirements met"
        ),
    }


def runnable_candidate_ids(batch: Mapping[str, Any]) -> list[str]:
    """Qwen first-batch identities, in Codex order. Flash only if actually qualified."""
    ids = [str(i) for i in (batch.get("qwen_singleton_order") or []) if i]
    if batch.get("flash_runnable_now"):
        ids.extend(str(i) for i in (batch.get("flash_singleton_order") or []) if i)
    return ids


# ---------------------------------------------------------------------------
# Qualification value — expected decisions changed, never a fabricated speedup
# ---------------------------------------------------------------------------


def _sanitize_win(win: Mapping[str, Any]) -> dict[str, Any]:
    body = {k: win.get(k) for k in DIRTY_WIN_FIELDS if k in win}
    # Extra non-hardware keys the caller used for identity are kept if present.
    for key in ("kind", "evidence_rung", "notes"):
        if key in win:
            body[key] = win[key]
    return _strip_hardware(body)


def estimate_qualification_value(
    dirty_wins: Sequence[Mapping[str, Any]] | None,
    batch: Mapping[str, Any],
    *,
    resident_convenience: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Value of a protected window vs continuing dirty work.

    DIAGNOSTIC_RELATIVE never promotes, so continuing dirty work changes zero
    decisions. A protected window changes a decision when a dirty win maps onto
    Codex's staged batch and would_change_decision_if_protected, and when an
    undecided staged-batch candidate would become PASS/REJECT.

    Resident convenience is read and discarded. It does not enter the count.
    No speedup number is produced.
    """
    wins = [_sanitize_win(w) for w in (dirty_wins or []) if isinstance(w, Mapping)]
    wins.sort(key=lambda w: str(w.get("id") or ""))
    runnable = runnable_candidate_ids(batch)
    runnable_set = set(runnable)
    cell_ids = {
        str(c.get("cell_id"))
        for c in (batch.get("qwen_cells_after") or [])
        if isinstance(c, Mapping) and c.get("cell_id")
    }
    protectable_targets = {f"qualify:{cid}" for cid in runnable_set} | cell_ids | runnable_set

    pending: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    already_decided: list[str] = []
    for win in wins:
        klass = str(win.get("measurement_class") or "STATIC_ONLY")
        ident = str(win.get("id") or "")
        cid = str(win.get("candidate_id") or "") or None
        target = str(win.get("decision_target") or "") or (f"qualify:{cid}" if cid else None)
        if klass == "PROTECTED_ABSOLUTE":
            already_decided.append(ident)
            continue
        if klass not in MEASUREMENT_DIRTY:
            deferred.append(
                {
                    "id": ident,
                    "reason": f"measurement_class {klass!r} is not a dirty win",
                }
            )
            continue
        in_batch = False
        if cid and cid in runnable_set:
            in_batch = True
        if target and (target in protectable_targets):
            in_batch = True
        if not in_batch:
            deferred.append(
                {
                    "id": ident,
                    "candidate_id": cid,
                    "reason": "not in Codex staged batch; this window will not invent a new order",
                }
            )
            continue
        if win.get("would_change_decision_if_protected"):
            pending.append(
                {
                    "id": ident,
                    "candidate_id": cid,
                    "decision_target": target,
                    "apparent_outcome": win.get("apparent_outcome"),
                }
            )
        else:
            deferred.append(
                {
                    "id": ident,
                    "candidate_id": cid,
                    "reason": "would_change_decision_if_protected is false; a protected window would not flip a decision",
                }
            )

    decision_targets = sorted(
        {str(p.get("decision_target")) for p in pending if p.get("decision_target")}
    )
    pending_cids = {p.get("candidate_id") for p in pending if p.get("candidate_id")}
    undecided_batch = [cid for cid in runnable if cid not in pending_cids]
    batch_targets = [f"qualify:{cid}" for cid in undecided_batch]

    n_from_dirty = len(decision_targets)
    n_from_batch = len(batch_targets)
    n_if_protected = n_from_dirty + n_from_batch
    n_if_dirty = 0  # DIAGNOSTIC_RELATIVE never promotes. This is the load-bearing zero.

    _ = resident_convenience  # explicitly ignored
    convenience_ignored = True

    justified_by_dirty = n_from_dirty > n_if_dirty
    justified_by_batch = n_from_batch > 0
    justified = justified_by_dirty or justified_by_batch

    return _authority(
        {
            "kind": "ESTIMATE",
            "expected_decisions_changed_if_protected": n_if_protected,
            "expected_decisions_changed_if_continue_dirty": n_if_dirty,
            "from_dirty_wins": n_from_dirty,
            "from_undecided_batch": n_from_batch,
            "window_justified": justified,
            "window_justified_by_dirty_wins": justified_by_dirty,
            "window_justified_by_batch": justified_by_batch,
            "n_dirty_wins": len(wins),
            "n_pending_dirty_in_batch": len(pending),
            "n_deferred": len(deferred),
            "n_already_protected": len(already_decided),
            "n_runnable_batch_candidates": len(runnable),
            "decision_targets_from_dirty_wins": decision_targets,
            "undecided_batch_candidate_ids": list(undecided_batch),
            "deferred": deferred,
            "pending": pending,
            "resident_convenience_ignored": convenience_ignored,
            "speedup_claimed": None,
            "not_a_fabricated_speedup": True,
            "measurement_class": "STATIC_ONLY",
            "rule": (
                "value is expected decisions changed. DIAGNOSTIC_RELATIVE never "
                "promotes, so continuing dirty work changes 0 decisions. A "
                "protected window decides staged-batch candidates and any dirty "
                "win that maps onto that batch. No speedup number is emitted."
            ),
            "batch_source": batch.get("source"),
            "promotion_gate": (
                "tools.future.contamination.assert_promotable refuses STATIC_ONLY "
                "and DIAGNOSTIC_RELATIVE; that is why continue-dirty is zero"
            ),
        }
    )


# ---------------------------------------------------------------------------
# No self-interest. Convenience is recorded and cannot veto.
# ---------------------------------------------------------------------------


def resident_convenience_may_veto(*_a: Any, **_k: Any) -> bool:
    """Always false. Named so tests can watch the guard refuse to fire as a veto."""
    return False


def _evict_from_protected_evidence(requires: bool) -> bool:
    """Identity: eviction iff protected evidence requires it. Convenience is not an input."""
    return bool(requires)


def decide_eviction(
    value: Mapping[str, Any],
    resident: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evict when protected evidence requires a window. Resident convenience cannot win."""
    convenience = dict(resident or {})
    recorded = {
        "cost_to_unload": convenience.get("cost_to_unload"),
        "busy": convenience.get("busy"),
        "hot": convenience.get("hot"),
        "dirty_wins_per_hour": convenience.get("dirty_wins_per_hour"),
        "prefer_stay_loaded": convenience.get("prefer_stay_loaded"),
        "convenience_weight": convenience.get("convenience_weight"),
        "resident_producing_dirty_wins": convenience.get("resident_producing_dirty_wins"),
    }
    requires = bool(value.get("window_justified"))
    veto = resident_convenience_may_veto(recorded)
    evict = _evict_from_protected_evidence(requires)
    if veto:
        # Unreachable by construction; kept so a future regression is a watched failure.
        raise rs.FailClosed(
            "partial_result",
            "resident_convenience_may_veto returned True; convenience must not veto protected evidence",
        )
    if requires:
        reason = (
            "protected evidence requires a window "
            f"(dirty_decisions={value.get('from_dirty_wins')} "
            f"batch_undecided={value.get('from_undecided_batch')}); "
            "resident convenience cannot veto"
        )
    else:
        reason = (
            "no pending protected decision and no runnable staged-batch candidate; "
            "not evicting. this is absence of protected evidence, not resident preference"
        )
    return _authority(
        {
            "kind": "DECISION",
            "evict": evict,
            "protected_evidence_requires": requires,
            "resident_convenience_read": recorded,
            "resident_convenience_vetoed": False,
            "self_preference_path": False,
            "rule": (
                "protected evidence outranks resident convenience. "
                "evict == window_justified. convenience is recorded and discarded."
            ),
            "reason": reason,
        }
    )


# ---------------------------------------------------------------------------
# Occupancy ledger — a model of loadedness. This sidecar never signals a PID.
# ---------------------------------------------------------------------------


def make_occupancy(
    *,
    state: str = "LOADED",
    resident_unloaded: bool = False,
    frozen_ids: Sequence[str] | None = None,
    mutate: bool = False,
) -> dict[str, Any]:
    if state not in RESIDENT_STATES:
        raise rs.FailClosed("stale_pipeline_cache", f"unknown occupancy state {state!r}")
    return {
        "state": state,
        "resident_unloaded": bool(resident_unloaded),
        "frozen_work_ids": list(frozen_ids or []),
        "mutate": bool(mutate),
        "sidecar_signalled_process": False,
        "observed_not_mutated": not mutate,
    }


def occupancy_needs_rollback(occupancy: Mapping[str, Any]) -> bool:
    return str(occupancy.get("state") or "") in NEEDS_ROLLBACK_STATES


def resident_left_unloaded(occupancy: Mapping[str, Any]) -> bool:
    return str(occupancy.get("state") or "") in UNLOADED_STATES or bool(
        occupancy.get("resident_unloaded")
    )


def _set_occupancy(occ: dict[str, Any], state: str, *, unloaded: bool | None = None) -> dict[str, Any]:
    if not occ.get("mutate"):
        # Live dry-run observes; it does not pretend a PID went away.
        occ["intended_state"] = state
        return occ
    occ["state"] = state
    if unloaded is not None:
        occ["resident_unloaded"] = bool(unloaded)
    occ["intended_state"] = state
    return occ


# ---------------------------------------------------------------------------
# Locks — READ only. Never flock, never mkdir, never seize.
# ---------------------------------------------------------------------------


def _lock_rel_present(lock_rel: str, root: Path) -> dict[str, Any]:
    path = root / lock_rel
    exists = path.is_file()
    holders: dict[str, Any] = {"status": "SKIPPED", "pids": [], "reason": "lock file absent"}
    if exists:
        holders = qp._lsof_holders(path)
    present = bool(exists and holders.get("status") == "OK" and holders.get("pids"))
    if present:
        reason = (
            f"{lock_rel} is held by pids {holders['pids']} under {root}; "
            "observed read-only, lock was not taken"
        )
    elif not exists:
        reason = (
            f"no existing HCLI lease at {path}: lock file not visible from this root. "
            "sidecar will not create .hcli/locks or call _try_lock"
        )
    else:
        reason = (
            f"lock file exists at {path} but no holder could be proven without flock "
            f"(lsof status={holders.get('status')!r} reason={holders.get('reason')!r}). "
            "fail closed: present=false. flock would be a seizure"
        )
    return {
        "lock_rel": lock_rel,
        "root": str(root),
        "path": str(path),
        "lock_file_exists": exists,
        "holders": holders,
        "present": present,
        "reason": reason,
    }


def read_protected_locks(
    *,
    lock_rels: Sequence[str] | None = None,
    roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Identify whether any existing HCLI protected lease is present.

    Fail closed: present is True only when a holder can be observed WITHOUT
    taking the lock. Exclusive flock is a seizure and is never attempted.
    """
    rels = [str(r) for r in (lock_rels or DEFAULT_LOCK_RELS)]
    search = list(roots) if roots is not None else _checkout_roots()
    observations: list[dict[str, Any]] = []
    for rel in rels:
        for root in search:
            observations.append(_lock_rel_present(rel, root))
    present = any(o.get("present") for o in observations)
    any_exists = any(o.get("lock_file_exists") for o in observations)
    unproven = [
        o
        for o in observations
        if o.get("lock_file_exists") and not o.get("present")
    ]
    if present:
        reason = "at least one protected lock has a proven holder; sidecar did not take it"
    elif unproven:
        reason = (
            "protected lock file(s) exist but no holder could be proven without flock; "
            "fail closed: present=false. flock would be a seizure"
        )
    elif not any_exists:
        reason = (
            "no existing HCLI protected lease is visible from this checkout. "
            "sidecar will not create one"
        )
    else:
        reason = "lease present is fail-closed false"
    primary = qp.read_hcli_lease_state(repo=REPO)
    return _authority(
        {
            "kind": "READ",
            "present": present,
            "lock_rels": rels,
            "observations": observations,
            "primary_hcli_lock": {
                "present": bool(primary.get("present")),
                "lock_file_exists": primary.get("lock_file_exists"),
                "reason": primary.get("reason"),
                "holders": primary.get("holders"),
            },
            "probe": "lsof -t on existing path only; never fcntl.LOCK_EX, never mkdir",
            "not_called": [
                "hcli.agentos.protected_accelerator_benchmark._try_lock",
                "hcli.agentos.protected_accelerator_benchmark.run_protected_accelerator_benchmark",
                "fcntl.flock",
                "fcntl.LOCK_EX",
            ],
            "reason": reason,
            "execution_ok": present,
        }
    )


# ---------------------------------------------------------------------------
# WorkUnits — SLEEPING until hardware qualifies. Never a synthetic result.
# ---------------------------------------------------------------------------


def emit_window_workunits(
    *,
    batch: Mapping[str, Any],
    lease_present: bool,
    evict: bool,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    wake_when = [
        "existing_hcli_lease",
        "machine_quiescence",
        "metal_capable_gpu",
        "metal_compiler",
    ]
    sleeping = not lease_present
    status = "blocked"
    classification = "SLEEPING" if sleeping else "STATIC_ONLY"
    blocked_reason = (
        None
        if lease_present
        else (
            "SLEEPING: protected start requires an existing HCLI lease this sidecar "
            "does not hold; HCLI wakes this unit when the hardware qualifies. "
            "Not a synthetic result."
        )
    )
    units: list[dict[str, Any]] = []

    def _emit(
        *,
        ident: str,
        description: str,
        verifier: str,
        resource_class: str,
        extras: Mapping[str, Any],
    ) -> None:
        row = ws.emit_hcli_workunit(
            id=ident,
            role="science",
            description=description,
            dependencies=[],
            resource_class=resource_class,
            verifier=verifier,
            provider="future.protected_window",
            effect_class="READ_ONLY",
            status=status,
            classification=classification,
            extras={
                "species": "accelerator_candidate_qualification",
                "sleeping": sleeping,
                "wake_when": list(wake_when),
                "blocked_reason": blocked_reason,
                "claim_boundary": ws.SIDECAR_CLAIM_BOUNDARY,
                "requires_quiescence": resource_class == "GPU_EXCLUSIVE",
                **dict(extras),
            },
        )
        ws.validate_emitted_unit(row)
        units.append(row)

    _emit(
        ident="future.protected-window.plan",
        description=(
            "Plan a protected window: estimate dirty-win value, decide eviction, "
            "and sequence restore. Does not seize a lease."
        ),
        verifier="future.protected_window.plan",
        resource_class="STATIC_ANALYSIS",
        extras={"evict": evict, "window_justified": bool(value.get("window_justified"))},
    )
    _emit(
        ident="future.protected-window.evict-resident",
        description=(
            "Proposal to checkpoint and unload the resident so a protected "
            "qualification can run. This unit does not signal a process."
        ),
        verifier="future.protected_window.evict",
        resource_class="STATIC_ANALYSIS",
        extras={"signals_process": False, "planned_evict": evict},
    )
    n_runnable = len(runnable_candidate_ids(batch))
    _emit(
        ident="future.protected-window.staged-qualification",
        description=(
            f"Run Codex's staged protected batch ({n_runnable} runnable identities "
            "derived from disk). Composes qualification_pipeline; does not execute "
            "a benchmark from this sidecar."
        ),
        verifier="future.protected_window.qualify",
        resource_class="GPU_EXCLUSIVE",
        extras={
            "batch_source": batch.get("source"),
            "n_runnable": n_runnable,
            "composed": "tools.future.qualification_pipeline.run_pipeline",
        },
    )
    _emit(
        ident="future.protected-window.restore-resident",
        description=(
            "Restore the resident from the window checkpoint and unfreeze dirty "
            "work. An interrupted window must land here rather than stay unloaded."
        ),
        verifier="future.protected_window.restore",
        resource_class="STATIC_ANALYSIS",
        extras={"rollback_required_if_unloaded": True},
    )
    units.sort(key=lambda r: str(r.get("id") or ""))
    compact = [
        {
            "id": u.get("id"),
            "status": u.get("status"),
            "classification": u.get("classification"),
            "resource_class": u.get("resource_class"),
            "sleeping": u.get("sleeping"),
            "wake_when": u.get("wake_when"),
            "blocked_reason": u.get("blocked_reason"),
            "requires_quiescence": u.get("requires_quiescence"),
            "claim_boundary": u.get("claim_boundary"),
            "verifier": u.get("verifier"),
        }
        for u in units
    ]
    return _authority(
        {
            "kind": "PROPOSAL",
            "n": len(units),
            "sleeping": sleeping,
            "does_not_schedule": True,
            "does_not_dispatch": True,
            "never_a_synthetic_result": True,
            "units": compact,
        }
    )


# ---------------------------------------------------------------------------
# Checkpoint / resume / rollback
# ---------------------------------------------------------------------------


def make_checkpoint(
    *,
    completed: Sequence[str],
    payloads: Mapping[str, Any],
    ctx: Mapping[str, Any],
    occupancy: Mapping[str, Any],
    in_progress_stage: str | None = None,
) -> dict[str, Any]:
    body = {
        "schema": CHECKPOINT_SCHEMA,
        "completed_stage_ids": list(completed),
        "stage_payloads": deepcopy(dict(payloads)),
        "in_progress_stage": in_progress_stage,
        "occupancy": deepcopy(dict(occupancy)),
        "ctx": {
            "lease": ctx.get("lease"),
            "batch": ctx.get("batch"),
            "value": ctx.get("value"),
            "decision": ctx.get("decision"),
            "dirty_wins": ctx.get("dirty_wins"),
            "dirty_units": ctx.get("dirty_units"),
            "dry_run": ctx.get("dry_run"),
            "contamination_class": ctx.get("contamination_class"),
        },
    }
    return rs.seal_doc(body)


def admit_checkpoint(doc: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise rs.FailClosed("corrupt_receipt", "checkpoint is not an object")
    if not rs.seal_is_valid(dict(doc)):
        raise rs.FailClosed(
            "corrupt_receipt",
            "checkpoint seal does not match canonical body; a broken seal is not a resume point",
        )
    if doc.get("schema") != CHECKPOINT_SCHEMA:
        raise rs.FailClosed(
            "stale_pipeline_cache",
            f"checkpoint schema {doc.get('schema')!r} is not {CHECKPOINT_SCHEMA}",
        )
    completed = list(doc.get("completed_stage_ids") or [])
    if not all(name in WINDOW_STAGES for name in completed):
        raise rs.FailClosed("stale_pipeline_cache", f"checkpoint names unknown stages: {completed}")
    prefix = list(WINDOW_STAGES[: len(completed)])
    if completed != prefix:
        raise rs.FailClosed(
            "stale_pipeline_cache",
            "completed_stage_ids is not a prefix of WINDOW_STAGES; refusing rather than skipping a hole",
        )
    in_progress = doc.get("in_progress_stage")
    if in_progress:
        if in_progress in completed:
            raise rs.FailClosed(
                "partial_result",
                f"stage {in_progress!r} is both completed and in_progress; partial is not a result",
            )
        if in_progress not in WINDOW_STAGES:
            raise rs.FailClosed(
                "stale_pipeline_cache",
                f"in_progress_stage {in_progress!r} is not a window stage",
            )
    occ = doc.get("occupancy")
    if not isinstance(occ, Mapping) or occ.get("state") not in RESIDENT_STATES:
        raise rs.FailClosed("corrupt_receipt", "checkpoint occupancy is missing or unknown")
    return dict(doc)


def rollback(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Restore the resident and unfreeze work. Never leaves occupancy UNLOADED.

    Safe to call on a checkpoint that is already restored (idempotent).
    Does not signal a process: occupancy is the ledger tests and HCLI watch.
    """
    ck = admit_checkpoint(checkpoint)
    occ = dict(ck.get("occupancy") or {})
    occ["mutate"] = True  # rollback is allowed to repair the ledger
    before = {
        "state": occ.get("state"),
        "resident_unloaded": bool(occ.get("resident_unloaded")),
        "frozen_work_ids": list(occ.get("frozen_work_ids") or []),
    }
    needed = occupancy_needs_rollback(occ)
    if occ.get("state") in UNLOADED_STATES or occ.get("resident_unloaded"):
        occ["state"] = "RESTORED"
        occ["resident_unloaded"] = False
        occ["sidecar_signalled_process"] = False
        occ["frozen_work_ids"] = []
        occ["intended_state"] = "RESUMED"
        occ["state"] = "RESUMED"
    elif occ.get("state") == "FROZEN":
        occ["frozen_work_ids"] = []
        occ["state"] = "CHECKPOINTED"
        occ["resident_unloaded"] = False
        occ["intended_state"] = "CHECKPOINTED"
    else:
        occ["resident_unloaded"] = False
    if resident_left_unloaded(occ):
        raise rs.FailClosed(
            "partial_result",
            "rollback left the resident unloaded; that is a campaign-level failure",
        )
    repaired = dict(ck)
    repaired["occupancy"] = occ
    repaired["rollback"] = {
        "needed": needed,
        "before": before,
        "after": {
            "state": occ.get("state"),
            "resident_unloaded": occ.get("resident_unloaded"),
            "frozen_work_ids": list(occ.get("frozen_work_ids") or []),
        },
        "resident_left_unloaded": False,
        "mission_stalled": False,
        "signals_process": False,
    }
    # Re-seal after repair so a subsequent admit sees a consistent snapshot.
    repaired.pop("seal_sha256", None)
    return rs.seal_doc(repaired)


def ensure_resident_restored(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    ck = admit_checkpoint(checkpoint)
    if occupancy_needs_rollback(ck.get("occupancy") or {}):
        return rollback(ck)
    return dict(ck)


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def _compact_qualification(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "composed": "tools.future.qualification_pipeline.run_pipeline",
        "schema": result.get("schema"),
        "n_stages": result.get("n_stages"),
        "planning_walk_complete": result.get("planning_walk_complete"),
        "execution_stop": result.get("execution_stop"),
        "lease_present": result.get("lease_present"),
        "contamination_class": result.get("contamination_class"),
        "survivor_ids": list(result.get("survivor_ids") or []),
        "dropped_ids": list(result.get("dropped_ids") or []),
        "measurement_class": "STATIC_ONLY",
        "gpu_authority": False,
        "executes_benchmark": False,
    }


def _run_one_stage(name: str, ctx: dict[str, Any]) -> dict[str, Any]:
    occ: dict[str, Any] = ctx.setdefault(
        "occupancy", make_occupancy(state="LOADED", mutate=False)
    )
    batch = ctx.get("batch") or {}
    if name == "estimate_qualification_value":
        value = estimate_qualification_value(
            ctx.get("dirty_wins") or [],
            batch,
            resident_convenience=ctx.get("resident"),
        )
        ctx["value"] = value
        return value
    if name == "decide_eviction":
        decision = decide_eviction(ctx.get("value") or {}, ctx.get("resident"))
        ctx["decision"] = decision
        ctx["evict"] = bool(decision.get("evict"))
        return decision
    if name == "checkpoint_resident":
        _set_occupancy(occ, "CHECKPOINTED", unloaded=False)
        return _authority(
            {
                "kind": "PLAN",
                "slots": [
                    "nx_identity",
                    "executable_identity",
                    "tokenizer_session",
                    "launch_args",
                    "shutdown",
                    "unload",
                    "crash_recovery",
                ],
                "recovered_from": "tools/future/resident_install.py PHASES",
                "wrote_process_checkpoint": False,
                "occupancy_state": occ.get("state"),
                "intended_state": occ.get("intended_state") or "CHECKPOINTED",
                "note": (
                    "SPEC of a resident checkpoint. This sidecar does not launch, "
                    "stop, or snapshot a live model process. Integration point: "
                    "resident_identity.py / resident_api.py (concurrent, not imported)."
                ),
            }
        )
    if name == "freeze_safe_work":
        dirty_units = [
            u
            for u in (ctx.get("dirty_units") or [])
            if isinstance(u, Mapping)
            and str(u.get("resource_class") or "") in DIRTY_RESOURCE_CLASSES
        ]
        frozen_ids = sorted({str(u.get("id")) for u in dirty_units if u.get("id")})
        if occ.get("mutate"):
            occ["frozen_work_ids"] = list(frozen_ids)
        _set_occupancy(occ, "FROZEN", unloaded=False)
        return _authority(
            {
                "kind": "PLAN",
                "frozen_work_ids": frozen_ids,
                "n_frozen": len(frozen_ids),
                "resource_classes_frozen": sorted(DIRTY_RESOURCE_CLASSES),
                "signalled": False,
                "occupancy_state": occ.get("state"),
                "note": (
                    "GPU_DIRTY_OK / GPU_DECODE work is marked frozen in the ledger "
                    "so GPU_EXCLUSIVE qualification can run. Integration point: "
                    "workgraph.py / sandbox.py (concurrent, not imported)."
                ),
            }
        )
    if name == "pause_unload_resident":
        evict = bool(ctx.get("evict"))
        will_signal = False
        if evict:
            _set_occupancy(occ, "UNLOADED", unloaded=True if occ.get("mutate") else False)
        payload = _authority(
            {
                "kind": "PLAN",
                "evict": evict,
                "sidecar_will_signal": will_signal,
                "sidecar_will_unload": False,
                "occupancy_state": occ.get("state"),
                "intended_state": occ.get("intended_state") or occ.get("state"),
                "resident_unloaded": bool(occ.get("resident_unloaded")),
                "reason": (
                    "PLAN to pause/unload the resident so the GPU is free for a "
                    "protected window. This sidecar never SIGSTOP/SIGKILL; "
                    "refuse_signal_process is the watched guard."
                    if evict
                    else "not evicting; resident stays loaded"
                ),
            }
        )
        return payload
    if name == "establish_protected_lease":
        lease = ctx.get("lease")
        if not isinstance(lease, Mapping):
            lease = read_protected_locks(lock_rels=batch.get("lock_rels"))
            ctx["lease"] = lease
        present = bool(lease.get("present"))
        evict = bool(ctx.get("evict"))
        if evict:
            _set_occupancy(occ, "LEASE_PENDING", unloaded=bool(occ.get("resident_unloaded")))
        if not evict:
            reason = "window not opened; no lease requested"
            exec_ok = True
        elif present:
            reason = (
                "an existing HCLI lease is present and was observed read-only. "
                "this sidecar still does not take flock and does not run the benchmark"
            )
            exec_ok = True
        else:
            reason = str(lease.get("reason") or "no existing HCLI lease")
            exec_ok = False
        return _authority(
            {
                "kind": "REQUEST",
                "never_seizure": True,
                "acquired": False,
                "observed_present": present,
                "would_flock": False,
                "would_call_try_lock": False,
                "execution_ok": exec_ok,
                "lock_rels": list(lease.get("lock_rels") or batch.get("lock_rels") or DEFAULT_LOCK_RELS),
                "not_called": list(lease.get("not_called") or []),
                "reason": reason,
                "occupancy_state": occ.get("state"),
            }
        )
    if name == "run_staged_qualification":
        _set_occupancy(occ, "QUALIFYING", unloaded=bool(occ.get("resident_unloaded")))
        nested = ctx.get("qualification")
        if not isinstance(nested, Mapping):
            if ctx.get("live"):
                try:
                    nested = _compact_qualification(qp.run_pipeline(dry_run=True, live=True))
                    nested["path_taken"] = "qp.run_pipeline(dry_run=True, live=True)"
                except Exception as exc:
                    nested = {
                        "composed": "tools.future.qualification_pipeline.run_pipeline",
                        "path_taken": f"refused:{type(exc).__name__}",
                        "reason": str(exc)[:240],
                        "planning_walk_complete": False,
                        "gpu_authority": False,
                        "executes_benchmark": False,
                    }
            else:
                sealed, src = load_visible_json(QUAL_RECEIPT_REL)
                if isinstance(sealed, Mapping) and isinstance(sealed.get("pipeline"), Mapping):
                    nested = _compact_qualification(sealed["pipeline"])
                    nested["path_taken"] = f"disk:{src}"
                else:
                    nested = {
                        "composed": "tools.future.qualification_pipeline.run_pipeline",
                        "path_taken": src,
                        "planning_walk_complete": False,
                        "note": (
                            "qualification receipt not visible and live=False; "
                            "stage records composition, does not guess survivors"
                        ),
                        "gpu_authority": False,
                        "executes_benchmark": False,
                    }
        ctx["qualification"] = nested
        order = runnable_candidate_ids(batch)
        return _authority(
            {
                "kind": "COMPOSE",
                "batch_order": order,
                "n_batch": len(order),
                "batch_source": batch.get("source"),
                "flash_runnable_now": bool(batch.get("flash_runnable_now")),
                "qualification": nested,
                "executes_benchmark": False,
                "occupancy_state": occ.get("state"),
                "note": (
                    "batch order is Codex's staged protected batch. this sidecar "
                    "composes qualification_pipeline and does not spawn protected_command"
                ),
            }
        )
    if name == "record_protected_receipts":
        _set_occupancy(occ, "RECEIPTS_PENDING", unloaded=bool(occ.get("resident_unloaded")))
        order = runnable_candidate_ids(batch)
        rows = [
            {
                "candidate_id": cid,
                "would_write": False,
                "measurement_class": "STATIC_ONLY",
                "bench_state": "UNKNOWN",
                "gpu_authority": False,
            }
            for cid in order
        ]
        return _authority(
            {
                "kind": "SPEC",
                "this_sidecar_emits": "STATIC_ONLY",
                "would_write_protected_absolute": False,
                "n": len(rows),
                "proposed_rows": rows,
                "reason": (
                    "protected receipts are Codex/HCLI's to write under a real lease. "
                    "this sidecar records a SPEC and does not invent a measurement"
                ),
                "occupancy_state": occ.get("state"),
            }
        )
    if name == "restore_resident":
        # Always restore if the ledger says unloaded — even when the lease was refused.
        _set_occupancy(occ, "RESTORED", unloaded=False)
        occ["resident_unloaded"] = False
        return _authority(
            {
                "kind": "PLAN",
                "restored": True,
                "resident_unloaded": False,
                "sidecar_will_signal": False,
                "occupancy_state": occ.get("state"),
                "note": (
                    "restore is mandatory after pause_unload so an interrupted or "
                    "refused window cannot leave the resident unloaded. Integration "
                    "point: succession.py / resident_api.py (concurrent, not imported)."
                ),
            }
        )
    if name == "resume_mission":
        occ["frozen_work_ids"] = []
        _set_occupancy(occ, "RESUMED", unloaded=False)
        occ["resident_unloaded"] = False
        units = emit_window_workunits(
            batch=batch,
            lease_present=bool((ctx.get("lease") or {}).get("present")),
            evict=bool(ctx.get("evict")),
            value=ctx.get("value") or {},
        )
        ctx["workunits"] = units
        return _authority(
            {
                "kind": "PLAN",
                "resumed": True,
                "resident_unloaded": False,
                "frozen_work_ids": [],
                "occupancy_state": occ.get("state"),
                "workunits": units,
                "note": (
                    "unfreeze GPU_DIRTY_OK work and refill the frontier from the "
                    "window result. Integration point: frontiers.py / wakeup.py "
                    "(concurrent, not imported)."
                ),
            }
        )
    raise rs.FailClosed("stale_pipeline_cache", f"unknown window stage {name!r}")


def _stage_record(
    name: str,
    index: int,
    status: str,
    payload: Mapping[str, Any],
    *,
    reason: str | None = None,
    execution_ok: bool | None = None,
) -> dict[str, Any]:
    exec_ok = payload.get("execution_ok") if execution_ok is None else execution_ok
    if exec_ok is None:
        exec_ok = True
    rec = {
        "index": index,
        "name": name,
        "status": status,
        "execution_ok": bool(exec_ok),
        "payload": dict(payload),
    }
    if reason is not None:
        rec["reason"] = reason
    return rec


def execution_stop(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for rec in stages:
        if not rec.get("execution_ok", True):
            payload = rec.get("payload") or {}
            return {
                "stage_index": rec["index"],
                "stage_id": rec["name"],
                "status": rec.get("status"),
                "reason": rec.get("reason") or payload.get("reason") or "execution_ok is false",
            }
    return {
        "stage_index": 6,
        "stage_id": "establish_protected_lease",
        "status": "REQUEST_EMITTED",
        "reason": (
            "planning walk found no earlier blocker, but this sidecar has no GPU "
            "authority and must not seize a lease"
        ),
    }


_STAGE_STATUS = {
    "establish_protected_lease": "REQUEST_EMITTED",
    "record_protected_receipts": "SPEC_EMITTED",
    "run_staged_qualification": "COMPOSED",
    "resume_mission": "PROPOSED",
}


def run_window(
    *,
    dry_run: bool = True,
    live: bool = False,
    dirty_wins: Sequence[Mapping[str, Any]] | None = None,
    dirty_units: Sequence[Mapping[str, Any]] | None = None,
    batch: Mapping[str, Any] | None = None,
    lease: Mapping[str, Any] | None = None,
    resident: Mapping[str, Any] | None = None,
    occupancy: Mapping[str, Any] | None = None,
    qualification: Mapping[str, Any] | None = None,
    snap: Mapping[str, Any] | None = None,
    contamination_class: str | None = None,
    interrupt_after: str | None = None,
    resume_from: Mapping[str, Any] | None = None,
    on_stage: Callable[[str], None] | None = None,
    mutate_occupancy: bool = False,
) -> dict[str, Any]:
    """Walk the eviction envelope as STATIC_ONLY planning. Never seizes a lease.

    mutate_occupancy=True is the test double for resident loadedness. Live
    --dry-run leaves occupancy observed-not-mutated so a PID is never signalled.
    interrupt_after raises WindowInterrupted after sealing a resume checkpoint.
    """
    completed: list[str] = []
    payloads: dict[str, Any] = {}
    records: list[dict[str, Any]] = []

    if resume_from is not None:
        ck = admit_checkpoint(resume_from)
        completed = list(ck["completed_stage_ids"])
        payloads = deepcopy(ck.get("stage_payloads") or {})
        restored = ck.get("ctx") or {}
        occ = dict(ck.get("occupancy") or make_occupancy())
        if mutate_occupancy:
            occ["mutate"] = True
        ctx: dict[str, Any] = {
            "dry_run": bool(dry_run),
            "live": bool(live),
            "lease": restored.get("lease") if restored.get("lease") is not None else lease,
            "batch": restored.get("batch") if restored.get("batch") is not None else batch,
            "value": restored.get("value"),
            "decision": restored.get("decision"),
            "dirty_wins": restored.get("dirty_wins") if restored.get("dirty_wins") is not None else dirty_wins,
            "dirty_units": restored.get("dirty_units") if restored.get("dirty_units") is not None else dirty_units,
            "resident": resident,
            "qualification": qualification,
            "occupancy": occ,
            "contamination_class": restored.get("contamination_class") or contamination_class,
        }
        if ctx.get("decision"):
            ctx["evict"] = bool(ctx["decision"].get("evict"))
        in_progress = ck.get("in_progress_stage")
        if in_progress:
            ctx["discarded_partial_stage"] = in_progress
        for name in completed:
            payload = payloads[name]
            index = WINDOW_STAGES.index(name) + 1
            records.append(
                _stage_record(
                    name,
                    index,
                    "RESUMED",
                    payload,
                    reason="restored from checkpoint; not recomputed",
                )
            )
    else:
        ctx = {"dry_run": bool(dry_run), "live": bool(live)}
        ctx["dirty_wins"] = list(dirty_wins or [])
        ctx["dirty_units"] = list(dirty_units or [])
        ctx["resident"] = dict(resident or {})
        if batch is not None:
            ctx["batch"] = load_staged_batch(batch)
        elif live:
            ctx["batch"] = load_staged_batch()
        else:
            raise rs.FailClosed(
                "incomplete_replication_bundle",
                "run_window(live=False) requires an injected batch; refusing to guess",
            )
        if lease is not None:
            ctx["lease"] = lease
        elif live:
            ctx["lease"] = read_protected_locks(lock_rels=ctx["batch"].get("lock_rels"))
        ctx["occupancy"] = dict(
            occupancy
            if occupancy is not None
            else make_occupancy(state="LOADED", mutate=mutate_occupancy)
        )
        if mutate_occupancy:
            ctx["occupancy"]["mutate"] = True
        if qualification is not None:
            ctx["qualification"] = qualification
        if snap is not None:
            klass = C.classify_contamination(snap)
            ctx["contamination_class"] = str(klass.get("contamination_class") or "UNKNOWN")
        elif contamination_class is not None:
            ctx["contamination_class"] = str(contamination_class)
        if interrupt_after is not None and interrupt_after not in WINDOW_STAGES:
            raise rs.FailClosed(
                "stale_pipeline_cache",
                f"interrupt_after {interrupt_after!r} is not a window stage",
            )

    for index, name in enumerate(WINDOW_STAGES, start=1):
        if name in completed:
            continue
        if on_stage is not None:
            on_stage(name)
        payload = _run_one_stage(name, ctx)
        status = _STAGE_STATUS.get(name, "COMPLETED")
        rec = _stage_record(name, index, status, payload, reason=payload.get("reason"))
        records.append(rec)
        payloads[name] = payload
        completed.append(name)
        if interrupt_after is not None and name == interrupt_after:
            ck = make_checkpoint(
                completed=completed,
                payloads=payloads,
                ctx=ctx,
                occupancy=ctx.get("occupancy") or {},
                in_progress_stage=None,
            )
            raise WindowInterrupted(name, ck)

    stop = execution_stop(records)
    occ = ctx.get("occupancy") or {}
    decision = ctx.get("decision") or {}
    value = ctx.get("value") or {}
    lease_state = ctx.get("lease") or {}
    # A completed walk always ends restored/resumed when occupancy was mutated.
    left_unloaded = resident_left_unloaded(occ) if occ.get("mutate") else False
    out = {
        "schema": SCHEMA,
        "dry_run": bool(dry_run),
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "n_stages": len(WINDOW_STAGES),
        "stages": records,
        "execution_stop": stop,
        "planning_walk_complete": len(records) == len(WINDOW_STAGES),
        "resumed_from_stage_count": sum(1 for r in records if r.get("status") == "RESUMED"),
        "lease_present": bool(lease_state.get("present")),
        "evict": bool(decision.get("evict")),
        "window_justified": bool(value.get("window_justified")),
        "expected_decisions_changed_if_protected": value.get("expected_decisions_changed_if_protected"),
        "expected_decisions_changed_if_continue_dirty": value.get("expected_decisions_changed_if_continue_dirty"),
        "batch_source": (ctx.get("batch") or {}).get("source"),
        "qwen_singleton_order": list((ctx.get("batch") or {}).get("qwen_singleton_order") or []),
        "occupancy": dict(occ),
        "resident_left_unloaded": left_unloaded,
        "self_preference_path": bool(decision.get("self_preference_path")),
        "workunits": (ctx.get("workunits") or {}).get("units"),
        "note": (
            "planning walk emits the eviction envelope as STATIC_ONLY specs. "
            "execute() is a separate entry point and always raises on this sidecar: "
            "no GPU authority, no flock."
        ),
    }
    if not dry_run:
        # Execution path: never flock, never return a synthetic protected result.
        execute(
            explicit_execute=True,
            lease=lease_state,
            contamination_class=ctx.get("contamination_class"),
            occupancy=occ,
        )
    return out


def execute(
    *,
    explicit_execute: bool = False,
    lease: Mapping[str, Any] | None = None,
    contamination_class: str | None = None,
    occupancy: Mapping[str, Any] | None = None,
) -> None:
    """RAISE unless an existing lease is present — and then raise anyway.

    Fail closed BEFORE mutating occupancy. An execute() refusal must not leave
    the resident unloaded. Flock is never attempted.
    """
    occ = occupancy
    try:
        if not explicit_execute:
            raise WindowRefused(
                "explicit_execute",
                "explicit --execute was not passed; sidecar will not open a protected window",
            )
        lease_state = dict(lease) if lease is not None else read_protected_locks()
        if not lease_state.get("present"):
            observed = str(lease_state.get("reason") or "no existing HCLI lease")
            raise WindowRefused(
                "existing_lease",
                f"{observed}; queue_policy.protected_start_requires_existing_hcli_lease; "
                "sidecar will not create one and will not flock",
            )
        klass = contamination_class
        if klass is None:
            _snap, klass_doc = qp.assess_quiescence(None)
            klass = str(klass_doc.get("contamination_class") or "UNKNOWN")
        if klass != "QUIESCENT":
            raise WindowRefused(
                "machine_quiescence",
                f"machine is {klass}; queue_policy.protected_start_requires_machine_quiescence; "
                "sidecar will not quiesce a worker",
            )
        # Existing lease + quiet + --execute still must not seize.
        raise WindowRefused(
            "gpu_authority",
            "existing lease and quiescence and --execute are not sufficient: "
            "this sidecar has no GPU authority and must not seize the HCLI lock. "
            "establish_protected_lease emits a request and stops",
        )
    except WindowRefused:
        if isinstance(occ, dict) and occupancy_needs_rollback(occ):
            repaired = rollback(
                make_checkpoint(
                    completed=[],
                    payloads={},
                    ctx={},
                    occupancy=occ,
                )
            )
            occ.clear()
            occ.update(repaired.get("occupancy") or {})
        raise


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def _frontier_f002() -> dict[str, Any]:
    path = FRONTIER_PATH
    if not path.is_file():
        return {
            "id": "F002",
            "present": None,
            "path_taken": "frontier receipt not visible in this checkout",
        }
    doc = load_json(path)
    for entry in doc.get("entries") or []:
        if entry.get("id") == "F002":
            return {
                "id": entry.get("id"),
                "title": entry.get("title"),
                "classification": entry.get("classification"),
                "prerequisite": entry.get("prerequisite"),
                "integration_target": entry.get("integration_target"),
                "path_taken": str(path),
            }
    return {"id": "F002", "present": None, "path_taken": f"{path}:no F002 entry"}


def recovered_implementation() -> list[dict[str, Any]]:
    return [
        {
            "path": "tools/future/qualification_pipeline.py",
            "role": "13-stage qualification sequencer; three ExecuteRefused conditions; no flock",
            "composed_as": "run_window stage run_staged_qualification + execute() conditions + refuse_* + read_hcli_lease_state / _lsof_holders",
            "adequate_for": "candidate selection, preflight DROP, A/B spec, lease request, resumable qualification walk",
            "not_adequate_for": "resident eviction, restore-on-interrupt, dirty-win value estimate, no-self-interest planner",
            "extended_not_forked": True,
        },
        {
            "path": "tools/future/contamination.py",
            "role": "QUIESCENT/LIGHT/HEAVY/UNKNOWN; assert_promotable refuses STATIC_ONLY",
            "composed_as": "continue-dirty = 0 because assert_promotable refuses non-PROTECTED_ABSOLUTE; assess_quiescence via qp",
        },
        {
            "path": "tools/future/candidate_planner.py",
            "role": "protected_batch_plan — Codex Qwen-first order and Flash return batch",
            "composed_as": "load_staged_batch prefers Codex handoff, then CANDIDATE_STAGED_PLAN.protected_batch, then protected_batch_plan",
        },
        {
            "path": "tools/future/workunit_species.py",
            "role": "HCLI WorkUnit constructor with bounded authority",
            "composed_as": "emit_window_workunits — SLEEPING until a real lease exists",
        },
        {
            "path": "tools/future/repro_science.py",
            "role": "FailClosed, checkpoint seal, interrupt/resume",
            "composed_as": "WindowRefused/WindowInterrupted; admit_checkpoint; rollback reseal",
        },
        {
            "path": "tools/future/resident_install.py",
            "role": "generic resident lifecycle slots including protected_benchmark_evacuation",
            "composed_as": "checkpoint_resident stage names those slots; this sidecar does not launch a resident",
        },
        {
            "path": "hcli/agentos/protected_accelerator_benchmark.py",
            "role": "the real protected lease (LOCK_NAME, _try_lock)",
            "composed_as": "cited read-only; runner NOT imported and NOT called",
            "on_disk_in_this_worktree": (REPO / "hcli/agentos/protected_accelerator_benchmark.py").is_file(),
        },
        {
            "path": "hcli/agentos/benchmark_boundary.py",
            "role": "QUALIFIED_PROTECTED vs DIAGNOSTIC_CONTAMINATED",
            "composed_as": "composed inside qualification_pipeline; this envelope does not re-classify a window",
        },
        {
            "path": "CODEX_ACCELERATOR_HANDOFF.json",
            "role": "exact_next_protected_qualification_sequence, current_staged_protected_batch, processes_leases_must_not_be_disturbed",
            "composed_as": "load_staged_batch primary source when visible via checkout roots",
        },
        {
            "path": "receipts/future/CANDIDATE_STAGED_PLAN.json",
            "role": "sealed protected_batch with Qwen-first run_order",
            "composed_as": "fallback disk authority when the Codex handoff is not in this sparse tree",
        },
        {
            "path": "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
            "role": "F002 READY_PROTECTED candidates idle on a GPU window the sidecar must not seize",
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "resident-eviction envelope around the landed 13-stage qualification sequencer: checkpoint, freeze, unload-plan, lease REQUEST, compose qp, receipts SPEC, restore, resume",
        "qualification value estimate is expected decisions changed; continuing dirty work is defined to change 0 decisions because DIAGNOSTIC_RELATIVE never promotes",
        "no self-interest: decide_eviction is the identity evict == window_justified; resident_convenience_may_veto is always false; convenience is recorded and discarded",
        "no seizure: acquire_lease/refuse_flock/execute raise rather than flock; lock probe is lsof on existing paths only",
        "interrupted window rolls back occupancy to RESUMED; restore_resident always runs after unload in a completed walk; rollback is idempotent and refuses to leave the resident unloaded",
        "batch order copied from Codex current_staged_protected_batch or CANDIDATE_STAGED_PLAN.protected_batch, never invented",
        "blocked physical work is emitted as SLEEPING WorkUnits that HCLI can wake; never a synthetic PROTECTED_ABSOLUTE result",
        "--dry-run walks all envelope stages against the real batch and real lock probe and reports the execution stop (lease on this host)",
    ]


def negative_findings() -> list[str]:
    return [
        "this sidecar has no GPU authority and cannot produce DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE",
        "exclusive flock is never used to inspect a lock; lsof is the only holder probe; unproven holders are present=false",
        "MetalContext / xcrun Metal compiler / HEAVY machine / Flash SCAFFOLD_ONLY / teacher 0/256 remain physical blockers and become SLEEPING units, not synthetic results",
        "qualification_pipeline.py is not modified (prohibited write); the eviction sequence wraps it",
        "concurrent wave modules (dirty_measure, resident_api, sandbox, wakeup, frontiers, succession, super_resident, workgraph) are not imported; local interfaces stand in and are named as integration points",
        "this sidecar never signals, pauses, or unloads a live resident process; occupancy mutation is a ledger for resume/rollback proofs",
        "Flash return batch stays closed while FLASH_NX_COMPLETENESS_AUDIT.seven_all_met is not true",
        "cannot update receipts/headless or take .hcli locks (Codex surface)",
    ]


def resident_callable(window: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = dict(window or {})
    return {
        "can_hcli_invoke": True,
        "entry_point": "python3 tools/future/protected_window.py --dry-run",
        "module": "tools.future.protected_window",
        "functions": {
            "run_window": "run_window(*, dry_run=True, live=False, ...) -> dict",
            "execute": "execute(*, explicit_execute=False, lease=None, contamination_class=None) -> None  # always raises WindowRefused",
            "estimate_qualification_value": "estimate_qualification_value(dirty_wins, batch) -> dict",
            "decide_eviction": "decide_eviction(value, resident=None) -> dict  # evict iff protected evidence requires",
            "rollback": "rollback(checkpoint) -> dict  # never leaves resident unloaded",
            "acquire_lease": "acquire_lease() -> None  # always raises lease_seizure",
        },
        "workunit_emitted": [
            "future.protected-window.plan",
            "future.protected-window.evict-resident",
            "future.protected-window.staged-qualification",
            "future.protected-window.restore-resident",
        ],
        "receipt_written": f"receipts/future/{RECEIPT}",
        "frontier_fed": "F002 — READY_PROTECTED candidates idle on a GPU window; window result is a SLEEPING refill until a real lease exists",
        "how_it_fails_closed": (
            "execute() raises without --execute, without a proven existing lease, "
            "when the machine is not QUIESCENT, and still raises gpu_authority when "
            "all three pass. acquire_lease/refuse_flock raise rather than flock. "
            "An interrupted unload rolls back occupancy. Dirty continuation cannot "
            "promote. Flash stays closed until NX is qualified."
        ),
        "sleeping_until": [
            "existing_hcli_lease",
            "machine_quiescence",
            "metal_capable_gpu",
            "metal_compiler",
        ],
        "evict_on_this_plan": result.get("evict"),
        "execution_stop": result.get("execution_stop"),
    }


def build(window: Mapping[str, Any] | None = None) -> Path:
    result = dict(window) if window is not None else run_window(dry_run=True, live=True)
    stop = result.get("execution_stop") or {}
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Protected evidence outranks resident convenience. Plan the sequence "
            "that checkpoints the resident, frees the GPU, runs Codex's staged "
            "qualification, and restores the mission — and keep that sequence "
            "incapable of seizing a lease it does not hold. FIVE ERAS, THREE "
            "ODYSSEYS. FPGA stays inside Accelerator / Physical Compiler / Fusion. "
            "DISK STATE IS AUTHORITY."
        ),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "no_era_vi": True,
        "no_odyssey_iv": True,
        "fpga": (
            "FPGA is part of Accelerator / Physical Compiler / Fusion. It is not "
            "its own civilization and this module does not build an FPGA backend."
        ),
        "vocabulary": {
            "DIAGNOSTIC_RELATIVE": "contaminated A/B on a busy machine. Guides. Never promotes.",
            "PROTECTED_ABSOLUTE": "measurement taken under a real protected GPU lease. Decides.",
            "STATIC_ONLY": "this sidecar. No GPU. Bench state UNKNOWN. Cannot promote.",
        },
        "stages_declared": list(WINDOW_STAGES),
        "n_stages": len(WINDOW_STAGES),
        "composed_qualification_stages": list(qp.STAGES),
        "authority_boundary": {
            "execute_requires": [
                "explicit --execute",
                "existing HCLI lease (read, never created, never flocked)",
                "machine QUIESCENT (assessed, never coerced)",
            ],
            "even_then": "sidecar has no GPU authority and still raises",
            "refuse_functions": [
                "refuse_start_benchmark",
                "refuse_create_lease",
                "refuse_signal_process",
                "refuse_quiesce_worker",
                "refuse_flock",
                "acquire_lease",
            ],
            "no_self_interest": (
                "decide_eviction.evict == window_justified; "
                "resident_convenience_may_veto is always false"
            ),
        },
        "queue_policy_binding": {
            "protected_start_requires_existing_hcli_lease": True,
            "protected_start_requires_machine_quiescence": True,
            "diagnostic_results_do_not_promote": True,
            "source": "qualification_pipeline queue_policy + Codex handoff preconditions",
        },
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "frontier_entry": _frontier_f002(),
        "window": result,
        "dry_run_stop": {
            "stage_index": stop.get("stage_index"),
            "stage_id": stop.get("stage_id"),
            "reason": stop.get("reason"),
            "honest_on_this_machine": (
                "stops at the lease request: no proven existing HCLI lease. "
                "later stages are still walked as STATIC_ONLY specs so restore "
                "is planned even when the lease is refused. flock is never taken."
            ),
        },
        "integration": {
            "run_window": (
                "run_window(*, dry_run=True, live=False, dirty_wins=None, batch=None, "
                "lease=None, interrupt_after=None, resume_from=None, "
                "mutate_occupancy=False) -> dict"
            ),
            "execute": (
                "execute(*, explicit_execute=False, lease=None, contamination_class=None) -> None  "
                "# always raises WindowRefused"
            ),
            "rollback": "rollback(checkpoint) -> dict  # occupancy never remains UNLOADED",
            "load_staged_batch": "load_staged_batch(injected=None) -> dict  # Codex order, not a guess",
        },
        "integration_points": {
            "dirty_measure.py": "DirtyWin schema (DIRTY_WIN_FIELDS); swap without changing the estimator",
            "resident_identity.py / resident_api.py": "actual checkpoint/unload of a live resident",
            "sandbox.py": "orchestrator sandbox that would freeze dirty work for real",
            "workgraph.py": "graph of GPU_DIRTY_OK units the freeze stage names",
            "wakeup.py": "HCLI wake of SLEEPING WorkUnits when hardware qualifies",
            "frontiers.py": "F002 refill after a real protected receipt exists",
            "succession.py": "restore/resume of resident identity after the window",
            "super_resident.py": "HCLI super-resident that invokes this entry point",
            "qualification_pipeline.py": "composed, not forked; 13 stages stay there",
        },
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "resident_callable": resident_callable(result),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


selftest = build


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Protected evidence", 1)[0])
    ap.add_argument("--dry-run", action="store_true", help="walk the envelope; report where execute() would stop")
    ap.add_argument("--build", action="store_true", help="emit the sealed receipt")
    ap.add_argument("--selftest", action="store_true", help="alias of --build")
    ap.add_argument("--execute", action="store_true", help="attempt execute(); must refuse on this sidecar")
    args = ap.parse_args()
    if args.execute:
        try:
            execute(explicit_execute=True)
        except WindowRefused as exc:
            print(f"execute refused [{exc.fault}]: {exc.reason}")
            return 2
        print("execute returned without raising — that is a campaign-level failure")
        return 1
    result = run_window(dry_run=True, live=True)
    out = build(window=result)
    summary = {
        "dry_run": True,
        "execution_stop": result.get("execution_stop"),
        "planning_walk_complete": result.get("planning_walk_complete"),
        "lease_present": result.get("lease_present"),
        "evict": result.get("evict"),
        "window_justified": result.get("window_justified"),
        "expected_decisions_changed_if_protected": result.get(
            "expected_decisions_changed_if_protected"
        ),
        "expected_decisions_changed_if_continue_dirty": result.get(
            "expected_decisions_changed_if_continue_dirty"
        ),
        "batch_source": result.get("batch_source"),
        "n_qwen_singletons": len(result.get("qwen_singleton_order") or []),
        "resident_left_unloaded": result.get("resident_left_unloaded"),
        "self_preference_path": result.get("self_preference_path"),
        "gpu_authority": False,
        "measurement_class": "STATIC_ONLY",
        "stages": [
            {
                "index": rec["index"],
                "name": rec["name"],
                "status": rec["status"],
                "execution_ok": rec.get("execution_ok"),
            }
            for rec in result.get("stages") or []
        ],
        "receipt": str(out),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    stop = result.get("execution_stop") or {}
    print(
        f"dry-run stop: stage {stop.get('stage_index')} {stop.get('stage_id')}: {stop.get('reason')}"
    )
    _ = args
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
