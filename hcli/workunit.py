from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .resources import (
    MutationLock,
    ResourceLimits,
    can_admit,
    next_class_slot,
    normalize_resource_class,
    occupancy_of,
)

DEFAULT_RETRY_BUDGET = 3

# Durable policy. Depth and per-root count live with the unit and the DAG
# document so a restart cannot reset them; they are not in-process scratch.
MAX_REPAIR_DEPTH = 3
MAX_REPAIRS_PER_ROOT = 6

# A killed running unit is re-run from the start. Resuming an inference
# mid-token is not something this system can do, and nothing here claims it.
RESUME_POLICY = "rerun"

CLASSIFICATION_INTERRUPTED = "INTERRUPTED"


@dataclass
class WorkUnit:
    id: str
    role: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    assigned_runtime: Optional[int] = None
    attempts: int = 0
    resource_class: str = "LIGHT_CONTROL"
    repairs: Optional[str] = None
    failure_context: Optional[Dict[str, Any]] = None
    preferred_backend: Optional[str] = None
    assigned_backend: Optional[str] = None
    backend_task_id: Optional[str] = None
    verifier: Optional[str] = None
    effect_class: Optional[str] = None
    workspace: Optional[str] = None
    verification: Optional[Dict[str, Any]] = None
    ready_at: Optional[float] = None
    running_at: Optional[float] = None
    finished_at: Optional[float] = None
    # Repair lineage, structured rather than parsed back out of the id. A
    # `a.repair.1.repair.2` string carries the same information only if every
    # reader agrees to parse it, and nothing that has to be re-derived from a
    # name is a reliable budget.
    repair_root: Optional[str] = None
    repair_depth: int = 0
    repair_reason: Optional[str] = None
    repair_exhausted: bool = False
    classification: Optional[str] = None

    def __post_init__(self) -> None:
        self.resource_class = normalize_resource_class(self.resource_class)

    def content_hash(self) -> str:
        return content_identity(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "status": self.status,
            "assigned_runtime": self.assigned_runtime,
            "attempts": self.attempts,
            "resource_class": self.resource_class,
            "repairs": self.repairs,
            "failure_context": self.failure_context,
            "preferred_backend": self.preferred_backend,
            "assigned_backend": self.assigned_backend,
            "backend_task_id": self.backend_task_id,
            "verifier": self.verifier,
            "effect_class": self.effect_class,
            "workspace": self.workspace,
            "verification": self.verification,
            "repair_root": self.repair_root,
            "repair_depth": self.repair_depth,
            "repair_reason": self.repair_reason,
            "repair_exhausted": self.repair_exhausted,
            "ready_at": self.ready_at,
            "running_at": self.running_at,
            "finished_at": self.finished_at,
            "classification": self.classification,
            "content_hash": content_identity(self),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkUnit":
        verification = data.get("verification")
        if not isinstance(verification, dict):
            verification = None
        return cls(
            id=data["id"],
            role=data["role"],
            description=data["description"],
            dependencies=list(data.get("dependencies") or []),
            status=data.get("status", "pending"),
            assigned_runtime=data.get("assigned_runtime"),
            attempts=int(data.get("attempts") or 0),
            resource_class=data.get("resource_class", "LIGHT_CONTROL"),
            repairs=data.get("repairs"),
            failure_context=data.get("failure_context"),
            preferred_backend=data.get("preferred_backend"),
            assigned_backend=data.get("assigned_backend"),
            backend_task_id=data.get("backend_task_id"),
            verifier=data.get("verifier"),
            effect_class=data.get("effect_class"),
            workspace=data.get("workspace"),
            verification=verification,
            repair_root=data.get("repair_root"),
            repair_depth=int(data.get("repair_depth") or 0),
            repair_reason=data.get("repair_reason"),
            repair_exhausted=bool(data.get("repair_exhausted")),
            ready_at=_opt_float(data.get("ready_at")),
            running_at=_opt_float(data.get("running_at")),
            finished_at=_opt_float(data.get("finished_at")),
            classification=data.get("classification"),
        )


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def content_identity(wu: WorkUnit) -> str:
    """Stable hash of the work, not of the id.

    Identity is (role, description, dependencies, verifier). The heading
    ordinal is not part of the work: two ids with this payload are the
    same unit, and one id with two payloads is a conflict.
    """
    verifier = getattr(wu, "verifier", None)
    payload = {
        "role": str(getattr(wu, "role", None) or ""),
        "description": str(getattr(wu, "description", None) or ""),
        "dependencies": [str(d) for d in (getattr(wu, "dependencies", None) or [])],
        "verifier": str(verifier) if verifier else "",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class IdentityConflict(ValueError):
    """Same id, different content. Must be visible; never a silent overwrite."""

    def __init__(self, unit_id: str, existing: WorkUnit, incoming: WorkUnit) -> None:
        self.unit_id = unit_id
        self.existing = existing
        self.incoming = incoming
        self.existing_hash = content_identity(existing)
        self.incoming_hash = content_identity(incoming)
        super().__init__(
            f"WorkUnit id {unit_id!r} already exists with different content "
            f"(existing={self.existing_hash[:16]} incoming={self.incoming_hash[:16]})"
        )


@dataclass
class AdmitOutcome:
    unit: WorkUnit
    kind: str  # "inserted" | "idempotent"


def admit_unit(units: Dict[str, WorkUnit], wu: WorkUnit) -> AdmitOutcome:
    """Admit ``wu`` into ``units``.

    Same content (any id) is idempotent. Same id with different content
    is an IdentityConflict. The existing unit wins so runtime state is
    not clobbered by a replan of the same work.
    """
    incoming_hash = content_identity(wu)
    existing = units.get(wu.id)
    if existing is not None:
        if content_identity(existing) == incoming_hash:
            return AdmitOutcome(unit=existing, kind="idempotent")
        raise IdentityConflict(wu.id, existing, wu)
    for other in units.values():
        if content_identity(other) == incoming_hash:
            return AdmitOutcome(unit=other, kind="idempotent")
    units[wu.id] = wu
    return AdmitOutcome(unit=wu, kind="inserted")


def failure_signature(
    wu: WorkUnit,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Stable hash of what actually went wrong, for repair-cycle detection."""
    ctx = context or {}
    payload = {
        "resource_class": getattr(wu, "resource_class", None),
        "verifier": getattr(wu, "verifier", None),
        "reason": ctx.get("reason"),
        "error": str(ctx.get("error"))[:200],
        "validation": json.dumps(ctx.get("validation"), sort_keys=True, default=str)[:400],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def lineage_root(wu: WorkUnit, units: Dict[str, WorkUnit]) -> str:
    stored = getattr(wu, "repair_root", None)
    if stored:
        return str(stored)
    seen = set()
    cur = wu
    while getattr(cur, "repairs", None):
        parent_id = cur.repairs
        if parent_id in seen:
            break
        seen.add(parent_id)
        parent = units.get(parent_id)
        if parent is None:
            return str(parent_id)
        cur = parent
    return str(getattr(cur, "repair_root", None) or cur.id)


def _parse_persisted_budget(persisted: Any) -> tuple:
    empty: Dict[str, int] = {}
    empty_sigs: Dict[str, set] = {}
    if not isinstance(persisted, dict) or not persisted:
        return empty, empty_sigs
    if "repair_budget" in persisted and isinstance(persisted.get("repair_budget"), dict):
        return _parse_persisted_budget(persisted["repair_budget"])
    if "counts" in persisted or "signatures" in persisted:
        counts_src = persisted.get("counts") or {}
        sigs_src = persisted.get("signatures") or {}
    elif "repair_counts" in persisted or "repair_signatures" in persisted:
        counts_src = persisted.get("repair_counts") or {}
        sigs_src = persisted.get("repair_signatures") or {}
    elif all(isinstance(v, int) and not isinstance(v, bool) for v in persisted.values()):
        counts_src = persisted
        sigs_src = {}
    else:
        return empty, empty_sigs
    counts: Dict[str, int] = {}
    if isinstance(counts_src, dict):
        for key, value in counts_src.items():
            try:
                counts[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
    signatures: Dict[str, set] = {}
    if isinstance(sigs_src, dict):
        for key, value in sigs_src.items():
            if isinstance(value, (list, tuple, set)):
                signatures[str(key)] = {str(item) for item in value}
            elif value:
                signatures[str(key)] = {str(value)}
    return counts, signatures


def rebuild_repair_budget(
    units: Dict[str, WorkUnit],
    persisted: Any = None,
) -> Dict[str, Any]:
    """Reconstruct per-root counts and signatures from units plus disk.

    The in-process maps reset on death. Units and the DAG document do not.
    Persisted counts are a floor: dropping repair units cannot reset the budget.
    """
    counts: Dict[str, int] = {}
    signatures: Dict[str, set] = {}
    for wu in units.values():
        if not getattr(wu, "repairs", None):
            continue
        root = lineage_root(wu, units)
        counts[root] = counts.get(root, 0) + 1
        ctx = getattr(wu, "failure_context", None) or {}
        sig = ctx.get("failure_signature") if isinstance(ctx, dict) else None
        if sig:
            signatures.setdefault(root, set()).add(str(sig))
    persisted_counts, persisted_sigs = _parse_persisted_budget(persisted)
    for root, n in persisted_counts.items():
        counts[root] = max(counts.get(root, 0), int(n))
    for root, sigs in persisted_sigs.items():
        signatures.setdefault(root, set()).update(sigs)
    return {"counts": counts, "signatures": signatures}


def serialize_repair_budget(budget: Dict[str, Any]) -> Dict[str, Any]:
    counts = {str(k): int(v) for k, v in (budget.get("counts") or {}).items()}
    signatures = {
        str(k): sorted(str(s) for s in (v or []))
        for k, v in (budget.get("signatures") or {}).items()
    }
    return {"counts": counts, "signatures": signatures}


def _next_repair_id(units: Dict[str, WorkUnit], original_id: str) -> str:
    n = 1
    while True:
        candidate = f"{original_id}.repair.{n}"
        if candidate not in units:
            return candidate
        n += 1


def _mark_exhausted(wu: WorkUnit, root: str, reason: str) -> None:
    wu.repair_exhausted = True
    wu.repair_root = wu.repair_root or root
    wu.repair_reason = reason


def emit_repair(
    units: Dict[str, WorkUnit],
    wu: WorkUnit,
    context: Optional[Dict[str, Any]] = None,
    budget: Optional[Dict[str, Any]] = None,
) -> Optional[WorkUnit]:
    """Create one repair unit, or refuse and mark the lineage exhausted.

    Interrupted units are re-run, not repaired. A crash is not a verifier
    failure and must not grow the repair tree.
    """
    if wu.status == "interrupted" or getattr(wu, "classification", None) == CLASSIFICATION_INTERRUPTED:
        return None
    if budget is None:
        budget = rebuild_repair_budget(units)
    budget.setdefault("counts", {})
    budget.setdefault("signatures", {})
    root = lineage_root(wu, units)
    depth = int(getattr(wu, "repair_depth", 0) or 0) + 1
    if depth > MAX_REPAIR_DEPTH:
        _mark_exhausted(
            wu,
            root,
            f"repair budget exhausted at depth {MAX_REPAIR_DEPTH} for root {root}",
        )
        return None

    emitted = int(budget["counts"].get(root, 0) or 0)
    if emitted >= MAX_REPAIRS_PER_ROOT:
        _mark_exhausted(
            wu,
            root,
            f"repair budget exhausted: {emitted} repairs already emitted for root {root}",
        )
        return None

    signature = failure_signature(wu, context)
    seen = budget["signatures"].setdefault(root, set())
    if not isinstance(seen, set):
        seen = set(seen)
        budget["signatures"][root] = seen
    if signature in seen:
        _mark_exhausted(
            wu,
            root,
            f"repair cycle: failure signature already seen in lineage {root}",
        )
        return None
    seen.add(signature)

    failure_context: Dict[str, Any] = {
        "failed_id": wu.id,
        "attempts": wu.attempts,
        "status": wu.status,
        "description": wu.description,
        "failure_signature": signature,
    }
    if context:
        failure_context.update(context)
        failure_context["failure_signature"] = signature
    repair = WorkUnit(
        id=_next_repair_id(units, wu.id),
        role=wu.role,
        description=f"repair of {wu.id}: {wu.description}",
        dependencies=[wu.id],
        resource_class=wu.resource_class,
        repairs=wu.id,
        failure_context=failure_context,
        verifier=getattr(wu, "verifier", None),
        preferred_backend=getattr(wu, "preferred_backend", None),
        repair_root=root,
        repair_depth=depth,
    )
    units[repair.id] = repair
    budget["counts"][root] = emitted + 1
    return repair


def mark_interrupted(wu: WorkUnit) -> bool:
    """Recover a unit left running after process death.

    Classification is INTERRUPTED, distinct from a verifier failure.
    ``attempts`` is not changed: the crash does not consume a retry.
    The unit will be re-run from the start; it is not resumed mid-token.
    """
    if wu.status != "running":
        return False
    if not transition_status(wu, "interrupted"):
        wu.status = "interrupted"
    wu.classification = CLASSIFICATION_INTERRUPTED
    wu.assigned_runtime = None
    ctx = dict(wu.failure_context or {})
    ctx["classification"] = CLASSIFICATION_INTERRUPTED
    ctx["reason"] = "process_death"
    ctx["resume_policy"] = RESUME_POLICY
    wu.failure_context = ctx
    return True


def transition_status(wu: WorkUnit, new_status: str) -> bool:
    valid = {
        "pending": {"ready"},
        "ready": {"running"},
        "running": {"completed", "failed", "interrupted"},
        "failed": {"ready"},
        "interrupted": {"ready"},
        "completed": set(),
    }
    if new_status in valid.get(wu.status, set()):
        wu.status = new_status
        return True
    return False


def _deps_satisfied(wu: WorkUnit, all_units: Dict[str, WorkUnit]) -> bool:
    for dep_id in wu.dependencies:
        dep = all_units.get(dep_id)
        if dep is None:
            return False
        if dep.status == "completed":
            continue
        # A repair unit may proceed from the failure it is repairing.
        if wu.repairs == dep_id and dep.status == "failed":
            continue
        return False
    return True


def _has_live_repair(wu: WorkUnit, all_units: Dict[str, WorkUnit]) -> bool:
    for other in all_units.values():
        if other.repairs == wu.id and other.status not in ("failed", "completed"):
            return True
    return False


def is_ready(wu: WorkUnit, all_units: Dict[str, WorkUnit]) -> bool:
    if wu.status not in ("pending", "failed", "interrupted"):
        return False
    if wu.repair_exhausted:
        # Repair budget spent. Terminal by policy, not by transient failure.
        return False
    # Interrupted is not failed: a crash does not consume the retry budget.
    if wu.status == "failed" and wu.attempts >= DEFAULT_RETRY_BUDGET:
        return False
    # Repair is the preferred path after failure. The retry budget is a
    # backstop used only once every outstanding repair of this unit has
    # itself failed.
    if wu.status == "failed" and _has_live_repair(wu, all_units):
        return False
    if wu.status == "failed":
        repairs = [u for u in all_units.values() if u.repairs == wu.id]
        if repairs and not all(r.status == "failed" for r in repairs):
            return False
    return _deps_satisfied(wu, all_units)


def identify_ready(units: Dict[str, WorkUnit]) -> List[WorkUnit]:
    ready = []
    # Offset stamps in this pass so two newly-ready units cannot share a clock
    # tick; a tied ready_at would sort by id and invert identification order.
    now = time.time()
    seq = 0
    for wu in units.values():
        if wu.status == "ready":
            ready.append(wu)
            continue
        if is_ready(wu, units):
            if transition_status(wu, "ready"):
                wu.ready_at = now + seq * 1e-6
                seq += 1
            ready.append(wu)
    return ready


def assign_ready(
    ready_units: List[WorkUnit],
    runtime_count: int,
    all_units: Optional[Dict[str, WorkUnit]] = None,
    limits: Optional[ResourceLimits] = None,
    mutation_lock: Optional[MutationLock] = None,
) -> List[tuple]:
    """Admit ready units under per-class concurrency limits.

    ``runtime_count`` is the resident-runtime count. It is intentionally
    NOT the GPU_DECODE cap; decode capacity comes from ACTIVE_DECODE_LIMIT
    on ``limits``. Classes that are idle are left idle — this function
    never invents work.

    An interrupted unit is re-run from the start. That re-run does not
    increment ``attempts``: the crash did not consume the retry. Nothing
    here resumes an inference mid-token.
    """
    del runtime_count  # not the decode limit; kept for call-site compatibility
    if limits is None:
        limits = ResourceLimits.resolve()
    pool = all_units if all_units is not None else {wu.id: wu for wu in ready_units}
    occupied = occupancy_of(pool.values())
    used_slots: Dict[str, set] = {}
    for wu in pool.values():
        if wu.status != "running":
            continue
        rc = normalize_resource_class(wu.resource_class)
        used_slots.setdefault(rc, set())
        if wu.assigned_runtime is not None:
            used_slots[rc].add(wu.assigned_runtime)

    assignments = []
    for wu in ready_units:
        if wu.status != "ready":
            continue
        rc = normalize_resource_class(wu.resource_class)
        if not can_admit(rc, occupied, limits):
            continue
        if rc == "MUTATION" and mutation_lock is not None:
            if not mutation_lock.acquire(wu.id):
                continue
        slot = next_class_slot(rc, used_slots, limits)
        if slot is None:
            if rc == "MUTATION" and mutation_lock is not None:
                mutation_lock.release(wu.id)
            continue
        if not transition_status(wu, "running"):
            if rc == "MUTATION" and mutation_lock is not None:
                mutation_lock.release(wu.id)
            continue
        wu.running_at = time.time()
        wu.assigned_runtime = slot
        if getattr(wu, "classification", None) == CLASSIFICATION_INTERRUPTED:
            wu.classification = None
            if wu.failure_context:
                ctx = dict(wu.failure_context)
                ctx.pop("classification", None)
                wu.failure_context = ctx or None
        else:
            wu.attempts += 1
        occupied[rc] += 1
        used_slots.setdefault(rc, set()).add(slot)
        assignments.append((wu, slot))
    return assignments
