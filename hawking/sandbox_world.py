"""Persistent S0 problem-world state for the Hawking Sandbox.

The Sandbox is a durable problem world, not another scheduler or public command.
This module owns only the restart-readable S0 record: branches are disposable,
findings/candidates remain claims until an independent evaluator and the
protected controller act, and a failed child never closes the root or siblings.
Execution policy remains in :mod:`hawking.execution_sandbox`; Option-C remains
the candidate/reviewer/controller lifecycle owner.
"""
from __future__ import annotations

import re
import threading


def _text(value: object) -> str:
    """Coerce a bounded world field to text, failing closed on non-strings."""
    if isinstance(value, str):
        return value
    raise SandboxWorldError("sandbox world field must be text", status=400)


def _bounded_text(value: object, field: str, limit: int) -> str:
    """Coerce a world field to text and enforce a hard length bound.

    Fails closed on non-strings and on empty or over-long values so a
    restart-readable S0 record can never carry an unbounded field.
    """
    text = _text(value)
    if not text:
        raise SandboxWorldError(
            f"sandbox world field {field} must be non-empty", status=400
        )
    if len(text) > limit:
        raise SandboxWorldError(
            f"sandbox world field {field} exceeds {limit} characters", status=400
        )
    return text


def _bounded_identifier(value: object, field: str, pattern: str) -> str:
    """Coerce a world identifier to text and enforce its declared shape.

    Identifiers are the restart-readable join keys of the S0 record, so a
    malformed value must fail closed at the boundary rather than be persisted
    and only discovered when a later reader cannot resolve the reference.
    """
    text = _bounded_text(value, field, 96)
    if re.fullmatch(pattern, text) is None:
        raise SandboxWorldError(
            f"sandbox world field {field} is not a valid identifier", status=400
        )
    return text


def _bounded_identifier_list(
    value: object, field: str, pattern: str, limit: int = 64
) -> tuple[str, ...]:
    """Coerce a bounded list of world identifiers, failing closed on any bad row.

    A restart-readable S0 record may carry a collection of join keys (for
    example the branch ids a finding cites). Each row must satisfy the same
    declared identifier shape as a scalar identifier, the collection must be
    non-empty, and the collection must not exceed ``limit`` rows so a persisted
    record can never carry an unbounded or partially-valid reference set. The
    result is a tuple so the persisted value is immutable and hashable.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise SandboxWorldError(
            f"sandbox world field {field} must be a list of identifiers", status=400
        )
    if not value:
        raise SandboxWorldError(
            f"sandbox world field {field} must be non-empty", status=400
        )
    if len(value) > limit:
        raise SandboxWorldError(
            f"sandbox world field {field} exceeds {limit} identifiers", status=400
        )
    return tuple(
        _bounded_identifier(row, f"{field}[{index}]", pattern)
        for index, row in enumerate(value)
    )


WORLD_SCHEMA = "hawking.sandbox.world.v1"
BRANCH_SCHEMA = "hawking.sandbox.branch.v1"
FINDING_SCHEMA = "hawking.sandbox.finding.v1"
CANDIDATE_SCHEMA = "hawking.sandbox.candidate.v1"
EVALUATOR_SCHEMA = "hawking.sandbox.evaluator.v1"
ARTIFACT_SCHEMA = "hawking.sandbox.artifact.v1"
EVENT_SCHEMA = "hawking.sandbox.event.v1"
DEFAULT_CAPABILITY_CATALOG = (
    "git", "python", "rust", "c_cpp", "go", "node", "browser",
    "symbolic_math", "formal_proof", "source_index", "debugger", "fuzzer",
    "metal", "cuda", "fpga", "gravity", "nr", "darkmatter", "nx",
    "odyssey", "tunnel", "notifications",
)
EVALUATOR_KINDS = {
    "software", "optimization", "mathematics", "research", "security",
}
EVALUATOR_KIND_ALIASES = {
    "math": "mathematics",
    "maths": "mathematics",
    "optimisation": "optimization",
    "sec": "security",
    "swe": "software",
}
WORLD_ID_RE = r"^S0-[A-Za-z0-9_-]{4,80}$"
BRANCH_ID_RE = r"^branch-[A-Za-z0-9_-]{3,80}$"
FINDING_ID_RE = r"^finding-[A-Za-z0-9_-]{3,80}$"
CANDIDATE_ID_RE = r"^candidate-[A-Za-z0-9_-]{3,80}$"

_LOCK = threading.RLock()


def _require_lock_held() -> None:
    """Fail closed when a mutating world operation runs without the S0 lock."""
    if not _LOCK._is_owned():
        raise SandboxWorldError("sandbox world mutation requires the S0 lock", status=500)


def _require_lock_held_for_read() -> None:
    """Fail closed when a read-only world operation runs without the S0 lock."""
    if not _LOCK._is_owned():
        raise SandboxWorldError("sandbox world read requires the S0 lock", status=500)


class SandboxWorldError(RuntimeError):
    """A bounded, fail-closed S0 world operation failure."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = int(status)














def _safe(value: Any, depth: int = 0) -> Any:
    if depth > 3:
        return "[omitted]"
    if isinstance(value, str):
        return _text(value, 1200)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, list):
        return [_safe(item, depth + 1) for item in value[:64]]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:80]:
            name = str(key)
            if any(marker in name.casefold() for marker in (
                "secret", "token", "cookie", "authorization", "api_key",
                "credential", "environment",
            )):
                continue
            result[name[:120]] = _safe(item, depth + 1)
        return result
    return _text(value, 500)


def _load(workspace: str | os.PathLike[str], world_id: str) -> dict[str, Any]:
    value = read_json_object_or_none(_world_path(workspace, world_id))
    if not value or value.get("schema") != WORLD_SCHEMA:
        raise SandboxWorldError("Sandbox world not found", 404)
    if str(value.get("world_id") or "") != str(world_id):
        raise SandboxWorldError("Sandbox world identity mismatch", 409)
    return value


def _save(workspace: str | os.PathLike[str], world: Mapping[str, Any]) -> None:
    atomic_write_json(
        _world_path(workspace, str(world.get("world_id") or "")),
        dict(world),
    )


def _branch(world: Mapping[str, Any], branch_id: str) -> dict[str, Any]:
    branches = world.get("branches")
    if not isinstance(branches, Mapping):
        raise SandboxWorldError("Sandbox branch index is malformed", 500)
    branch = branches.get(branch_id)
    if not isinstance(branch, Mapping):
        raise SandboxWorldError("Sandbox branch not found", 404)
    return dict(branch)


def _require_active_branch(world: Mapping[str, Any], branch_id: str) -> dict[str, Any]:
    branch = _branch(world, branch_id)
    if str(branch.get("state") or "") != "ACTIVE":
        raise SandboxWorldError(
            f"Sandbox branch is not active: {branch.get('state')}", 409
        )
    return branch


def _event(world: dict[str, Any], kind: str, **fields: Any) -> dict[str, Any]:
    sequence = int(world.get("next_sequence") or 0) + 1
    event = {
        "schema": EVENT_SCHEMA,
        "sequence": sequence,
        "kind": str(kind),
        "at": time.time(),
        **{str(key): _safe(value) for key, value in fields.items()},
    }
    world["next_sequence"] = sequence
    events = list(world.get("events") or [])
    events.append(event)
    world["events"] = events[-256:]
    world["updated_at"] = time.time()
    return event


def _append_unique(items: list[Any], value: Any, *, limit: int = 256) -> None:
    if value not in items:
        items.append(value)
    del items[:-limit]


def create_world(
    workspace: str | os.PathLike[str],
    *,
    objective: str,
    world_id: Optional[str] = None,
    capability_catalog: Iterable[Any] | None = None,
    budget_usd: float = 0.0,
    resource_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one persistent S0 root; no child worker is launched."""
    import re

    objective_text = _text(objective)
    if not objective_text:
        raise SandboxWorldError("Sandbox objective is required", 400)
    wid = _text(world_id, 100) if world_id else _id("S0")
    if not re.fullmatch(WORLD_ID_RE, wid):
        raise SandboxWorldError("world_id is invalid", 400)
    try:
        budget = max(0.0, float(budget_usd))
    except (TypeError, ValueError) as exc:
        raise SandboxWorldError("budget_usd must be numeric", 400) from exc
    now = time.time()
    catalog = [
        _text(item, 100) for item in (
            capability_catalog if capability_catalog is not None
            else DEFAULT_CAPABILITY_CATALOG
        ) if _text(item, 100)
    ]
    world: dict[str, Any] = {
        "schema": WORLD_SCHEMA,
        "world_id": wid,
        "objective": objective_text,
        "workspace": str(Path(workspace).expanduser().resolve()),
        "state": "ACTIVE",
        "root_persists": True,
        "created_at": now,
        "updated_at": now,
        "next_sequence": 0,
        "branches": {},
        "findings": [],
        "candidates": [],
        "evaluators": {},
        "artifacts": [],
        "child_failures": [],
        "events": [],
        "capability_catalog": sorted(set(catalog)),
        "auto_allocator": {
            "owner": "hawking.auto",
            "mode": "allocation_only",
            "claim_boundary": "Auto may choose resources; it does not certify findings",
        },
        "execution_policy": {
            "schema": "hawking.lab.execution_sandbox_policy.v1",
            "owner": "hawking.execution_sandbox.ExecutionSandboxPolicy",
            "deny_by_default": True,
        },
        "candidate_lifecycle": {
            "schema": "hawking.worker.option_c.v1",
            "owner": "hawking.worker.option_c",
            "promotion_authority": "protected_controller",
        },
        "resource_policy": _safe(dict(resource_policy or {})),
        "budget": {
            "authorized_usd": budget,
            "reserved_usd": 0.0,
            "spent_usd": 0.0,
            "available_usd": budget,
        },
    }
    _event(world, "world.created", objective=objective_text)
    with _LOCK:
        path = _world_path(workspace, wid)
        if path.exists():
            raise SandboxWorldError("Sandbox world already exists", 409)
        _save(workspace, world)
    return dict(world)


def load_world(workspace: str | os.PathLike[str], world_id: str) -> dict[str, Any]:
    with _LOCK:
        return _load(workspace, _text(world_id, 100))


def list_worlds(workspace: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with _LOCK:
        rows = []
        for path in sorted(_world_dir(workspace).glob("S0-*.json")):
            value = read_json_object_or_none(path)
            if not value or value.get("schema") != WORLD_SCHEMA:
                continue
            rows.append({
                "world_id": value.get("world_id"),
                "objective": value.get("objective"),
                "state": value.get("state"),
                "updated_at": value.get("updated_at"),
                "branch_count": len(value.get("branches") or {}),
            })
        return rows


def create_branch(
    workspace: str | os.PathLike[str],
    world_id: str,
    *,
    branch_id: Optional[str] = None,
    parent_branch_id: Optional[str] = None,
    worktree_ref: str,
    source_revision: str = "",
    budget_usd: float = 0.0,
) -> dict[str, Any]:
    import re

    bid = _text(branch_id, 100) if branch_id else _id("branch")
    if not re.fullmatch(BRANCH_ID_RE, bid):
        raise SandboxWorldError("branch_id is invalid", 400)
    ref = _text(worktree_ref, 400)
    if not ref:
        raise SandboxWorldError("worktree_ref is required", 400)
    try:
        budget = max(0.0, float(budget_usd))
    except (TypeError, ValueError) as exc:
        raise SandboxWorldError("branch budget_usd must be numeric", 400) from exc
    with _LOCK:
        world = _load(workspace, world_id)
        if bid in world["branches"]:
            raise SandboxWorldError("Sandbox branch already exists", 409)
        if parent_branch_id is not None:
            _branch(world, _text(parent_branch_id, 100))
        branch = {
            "schema": BRANCH_SCHEMA,
            "branch_id": bid,
            "parent_branch_id": _text(parent_branch_id, 100) if parent_branch_id else None,
            "worktree_ref": ref,
            "source_revision": _text(source_revision, 200),
            "state": "ACTIVE",
            "disposable": True,
            "created_at": time.time(),
            "updated_at": time.time(),
            "budget": {
                "authorized_usd": budget,
                "reserved_usd": 0.0,
                "spent_usd": 0.0,
            },
            "finding_ids": [],
            "candidate_ids": [],
            "artifact_ids": [],
        }
        world["branches"][bid] = branch
        _event(world, "branch.created", branch_id=bid, parent_branch_id=parent_branch_id)
        _save(workspace, world)
        return dict(branch)


def register_evaluator(
    workspace: str | os.PathLike[str],
    world_id: str,
    *,
    evaluator_id: Optional[str] = None,
    kind: str,
    owner: str = "protected_controller",
    independent: bool = True,
    selector: str = "",
) -> dict[str, Any]:
    eid = _text(evaluator_id, 100) if evaluator_id else _id("eval")
    kind_text = _text(kind, 40).casefold()
    if kind_text not in EVALUATOR_KINDS:
        raise SandboxWorldError("unsupported evaluator kind", 400)
    if not independent:
        raise SandboxWorldError("Sandbox evaluator must be independent", 403)
    if _text(owner, 100).casefold() in {"sandbox_model", "executor", "reviewer"}:
        raise SandboxWorldError("Sandbox model cannot own an evaluator", 403)
    evaluator = {
        "schema": EVALUATOR_SCHEMA,
        "evaluator_id": eid,
        "kind": kind_text,
        "owner": _text(owner, 100),
        "independent": True,
        "selector": _text(selector, 400),
        "created_at": time.time(),
    }
    with _LOCK:
        world = _load(workspace, world_id)
        if eid in world["evaluators"]:
            raise SandboxWorldError("evaluator already exists", 409)
        world["evaluators"][eid] = evaluator
        _event(world, "evaluator.registered", evaluator_id=eid, evaluator_kind=kind_text)
        _save(workspace, world)
    return dict(evaluator)


def reserve_budget(
    workspace: str | os.PathLike[str],
    world_id: str,
    *,
    amount_usd: float,
    branch_id: Optional[str] = None,
    reason: str = "",
) -> dict[str, Any]:
    """Reserve S0 budget without launching a worker or charging a provider."""
    try:
        amount = float(amount_usd)
    except (TypeError, ValueError) as exc:
        raise SandboxWorldError("amount_usd must be numeric", 400) from exc
    if amount <= 0:
        raise SandboxWorldError("amount_usd must be positive", 400)
    with _LOCK:
        world = _load(workspace, world_id)
        branch = _require_active_branch(world, branch_id) if branch_id else None
        budget = dict(world.get("budget") or {})
        authorized = float(budget.get("authorized_usd") or 0.0)
        spent = float(budget.get("spent_usd") or 0.0)
        reserved = float(budget.get("reserved_usd") or 0.0)
        available = authorized - spent - reserved
        if amount > available + 1e-12:
            raise SandboxWorldError("Sandbox budget reservation exceeds available budget", 409)
        budget["reserved_usd"] = reserved + amount
        budget["available_usd"] = authorized - spent - budget["reserved_usd"]
        world["budget"] = budget
        if branch is not None:
            branch_budget = dict(branch.get("budget") or {})
            branch_budget["reserved_usd"] = float(branch_budget.get("reserved_usd") or 0.0) + amount
            branch["budget"] = branch_budget
            world["branches"][branch_id] = branch
        _event(world, "budget.reserved", amount_usd=amount, branch_id=branch_id, reason=reason)
        _save(workspace, world)
        return dict(budget)


def settle_budget(
    workspace: str | os.PathLike[str],
    world_id: str,
    *,
    amount_usd: float,
    reserved_usd: float = 0.0,
    branch_id: Optional[str] = None,
    reason: str = "",
) -> dict[str, Any]:
    """Record observed spend against the S0 ledger, never beyond authorization."""
    try:
        amount = float(amount_usd)
        release = max(0.0, float(reserved_usd))
    except (TypeError, ValueError) as exc:
        raise SandboxWorldError("budget amounts must be numeric", 400) from exc
    if amount < 0 or release < 0:
        raise SandboxWorldError("budget amounts cannot be negative", 400)
    with _LOCK:
        world = _load(workspace, world_id)
        branch = _require_active_branch(world, branch_id) if branch_id else None
        budget = dict(world.get("budget") or {})
        authorized = float(budget.get("authorized_usd") or 0.0)
        spent = float(budget.get("spent_usd") or 0.0)
        reserved = float(budget.get("reserved_usd") or 0.0)
        if release > reserved + 1e-12 or spent + amount > authorized + 1e-12:
            raise SandboxWorldError("Sandbox spend exceeds authorized budget", 409)
        budget["reserved_usd"] = reserved - release
        budget["spent_usd"] = spent + amount
        budget["available_usd"] = authorized - budget["spent_usd"] - budget["reserved_usd"]
        world["budget"] = budget
        if branch is not None:
            branch_budget = dict(branch.get("budget") or {})
            branch_budget["reserved_usd"] = max(
                0.0, float(branch_budget.get("reserved_usd") or 0.0) - release
            )
            branch_budget["spent_usd"] = float(branch_budget.get("spent_usd") or 0.0) + amount
            branch["budget"] = branch_budget
            world["branches"][branch_id] = branch
        _event(world, "budget.settled", amount_usd=amount, reserved_usd=release, branch_id=branch_id, reason=reason)
        _save(workspace, world)
        return dict(budget)


def record_finding(
    workspace: str | os.PathLike[str],
    world_id: str,
    *,
    branch_id: str,
    finding: str,
    falsifier: str,
    action: str,
    evidence: Iterable[Any] = (),
    classification: str = "CANDIDATE",
) -> dict[str, Any]:
    finding_text = _text(finding)
    if not finding_text or not _text(falsifier) or not _text(action):
        raise SandboxWorldError("finding, falsifier, and action are required", 400)
    with _LOCK:
        world = _load(workspace, world_id)
        _require_active_branch(world, branch_id)
        item = {
            "schema": FINDING_SCHEMA,
            "finding_id": _id("finding"),
            "world_id": world_id,
            "branch_id": branch_id,
            "finding": finding_text,
            "falsifier": _text(falsifier),
            "action": _text(action),
            "evidence": [_safe(value) for value in list(evidence)[:32]],
            "classification": _text(classification, 80).upper() or "CANDIDATE",
            "authority_level": "candidate",
            "assimilated": False,
            "created_at": time.time(),
        }
        world["findings"].append(item)
        world["branches"][branch_id]["finding_ids"].append(item["finding_id"])
        _event(world, "finding.recorded", finding_id=item["finding_id"], branch_id=branch_id)
        _save(workspace, world)
        return dict(item)


def register_candidate(
    workspace: str | os.PathLike[str],
    world_id: str,
    *,
    branch_id: str,
    category: str,
    summary: str,
    evaluator_ids: Iterable[Any],
    artifact_ids: Iterable[Any] = (),
) -> dict[str, Any]:
    category_text = _text(category, 100)
    summary_text = _text(summary)
    if not category_text or not summary_text:
        raise SandboxWorldError("candidate category and summary are required", 400)
    evaluators = [_text(item, 100) for item in list(evaluator_ids)[:16] if _text(item, 100)]
    artifacts = [_text(item, 100) for item in list(artifact_ids)[:32] if _text(item, 100)]
    with _LOCK:
        world = _load(workspace, world_id)
        _require_active_branch(world, branch_id)
        missing = [eid for eid in evaluators if eid not in world["evaluators"]]
        if missing:
            raise SandboxWorldError(f"unknown evaluator(s): {missing}", 409)
        if not evaluators:
            raise SandboxWorldError("candidate requires an independent evaluator", 403)
        candidate = {
            "schema": CANDIDATE_SCHEMA,
            "candidate_id": _id("candidate"),
            "world_id": world_id,
            "branch_id": branch_id,
            "category": category_text,
            "summary": summary_text,
            "evaluator_ids": evaluators,
            "artifact_ids": artifacts,
            "state": "CANDIDATE",
            "promotion_authority": "protected_controller",
            "created_at": time.time(),
            "claim_boundary": "candidate is not an accepted result",
        }
        world["candidates"].append(candidate)
        world["branches"][branch_id]["candidate_ids"].append(candidate["candidate_id"])
        _event(world, "candidate.registered", candidate_id=candidate["candidate_id"], branch_id=branch_id)
        _save(workspace, world)
        return dict(candidate)


def record_artifact(
    workspace: str | os.PathLike[str],
    world_id: str,
    *,
    branch_id: str,
    path: str,
    digest: str,
    kind: str = "artifact",
) -> dict[str, Any]:
    path_text = _text(path, 500)
    digest_text = _text(digest, 128)
    if not path_text or not digest_text:
        raise SandboxWorldError("artifact path and digest are required", 400)
    with _LOCK:
        world = _load(workspace, world_id)
        _require_active_branch(world, branch_id)
        artifact = {
            "schema": ARTIFACT_SCHEMA,
            "artifact_id": _id("artifact"),
            "world_id": world_id,
            "branch_id": branch_id,
            "path": path_text,
            "sha256": digest_text,
            "kind": _text(kind, 80) or "artifact",
            "created_at": time.time(),
            "claim_boundary": "artifact provenance only; evaluator/controller still required",
        }
        world["artifacts"].append(artifact)
        world["branches"][branch_id]["artifact_ids"].append(artifact["artifact_id"])
        _event(world, "artifact.recorded", artifact_id=artifact["artifact_id"], branch_id=branch_id)
        _save(workspace, world)
        return dict(artifact)


def assimilate_finding(
    workspace: str | os.PathLike[str],
    world_id: str,
    finding_id: str,
    *,
    controller_id: str = "protected_controller",
) -> dict[str, Any]:
    if _text(controller_id, 100).casefold() in {"sandbox_model", "executor", "reviewer"}:
        raise SandboxWorldError("Sandbox model cannot assimilate authoritative state", 403)
    with _LOCK:
        world = _load(workspace, world_id)
        finding = next(
            (item for item in world["findings"]
             if isinstance(item, Mapping) and item.get("finding_id") == finding_id),
            None,
        )
        if finding is None:
            raise SandboxWorldError("Sandbox finding not found", 404)
        updated = dict(finding)
        updated["assimilated"] = True
        updated["assimilated_by"] = _text(controller_id, 100)
        updated["assimilated_at"] = time.time()
        world["findings"] = [
            updated if item.get("finding_id") == finding_id else item
            for item in world["findings"]
        ]
        _event(world, "finding.assimilated", finding_id=finding_id, controller_id=controller_id)
        _save(workspace, world)
        return updated


def fail_branch(
    workspace: str | os.PathLike[str],
    world_id: str,
    branch_id: str,
    *,
    reason: str,
    failure_class: str = "CHILD_FAILURE",
) -> dict[str, Any]:
    with _LOCK:
        world = _load(workspace, world_id)
        branch = _branch(world, branch_id)
        branch["state"] = "FAILED"
        branch["updated_at"] = time.time()
        branch["failure"] = {
            "class": _text(failure_class, 100) or "CHILD_FAILURE",
            "reason": _text(reason),
            "at": time.time(),
        }
        world["branches"][branch_id] = branch
        failure = {
            "branch_id": branch_id,
            "class": branch["failure"]["class"],
            "reason": branch["failure"]["reason"],
            "at": branch["failure"]["at"],
            "root_continues": True,
        }
        world["child_failures"].append(failure)
        _event(world, "branch.failed", branch_id=branch_id, failure_class=failure["class"])
        # The S0 root remains active; siblings are intentionally untouched.
        world["state"] = "ACTIVE"
        _save(workspace, world)
        return dict(failure)


def dispose_branch(
    workspace: str | os.PathLike[str],
    world_id: str,
    branch_id: str,
    *,
    reason: str = "branch complete",
) -> dict[str, Any]:
    with _LOCK:
        world = _load(workspace, world_id)
        branch = _branch(world, branch_id)
        branch["state"] = "DISPOSED"
        branch["disposed_at"] = time.time()
        branch["dispose_reason"] = _text(reason)
        world["branches"][branch_id] = branch
        _event(world, "branch.disposed", branch_id=branch_id, reason=reason)
        _save(workspace, world)
        return dict(branch)


def snapshot_world(
    workspace: str | os.PathLike[str],
    world_id: str,
    *,
    after_sequence: int = 0,
) -> dict[str, Any]:
    try:
        after = max(0, int(after_sequence))
    except (TypeError, ValueError) as exc:
        raise SandboxWorldError("after_sequence must be an integer", 400) from exc
    with _LOCK:
        world = _load(workspace, world_id)
        branches = [
            {
                "branch_id": branch.get("branch_id"),
                "parent_branch_id": branch.get("parent_branch_id"),
                "state": branch.get("state"),
                "worktree_ref": branch.get("worktree_ref"),
                "finding_count": len(branch.get("finding_ids") or []),
                "candidate_count": len(branch.get("candidate_ids") or []),
            }
            for branch in world["branches"].values()
            if isinstance(branch, Mapping)
        ]
        events = [
            dict(event) for event in world.get("events", [])
            if isinstance(event, Mapping) and int(event.get("sequence") or 0) > after
        ]
        return {
            "schema": WORLD_SCHEMA,
            "world_id": world["world_id"],
            "objective": world["objective"],
            "state": world["state"],
            "root_persists": True,
            "branches": branches,
            "finding_count": len(world.get("findings") or []),
            "candidate_count": len(world.get("candidates") or []),
            "evaluator_count": len(world.get("evaluators") or {}),
            "artifact_count": len(world.get("artifacts") or []),
            "child_failure_count": len(world.get("child_failures") or []),
            "budget": dict(world.get("budget") or {}),
            "events": events,
            "through_sequence": int(world.get("next_sequence") or 0),
            "claim_boundary": (
                "S0 state projection; candidates/findings are not accepted "
                "results and branches remain disposable"
            ),
        }


__all__ = [
    "ARTIFACT_SCHEMA", "BRANCH_SCHEMA", "CANDIDATE_SCHEMA", "EVALUATOR_SCHEMA",
    "FINDING_SCHEMA", "SandboxWorldError", "WORLD_SCHEMA", "assimilate_finding",
    "create_branch", "create_world", "dispose_branch", "fail_branch",
    "list_worlds", "load_world", "record_artifact", "record_finding",
    "register_candidate", "register_evaluator", "reserve_budget", "settle_budget",
    "snapshot_world",
]
