"""Typed execution bindings for bounded Hawking work.

This module is deliberately a read-mostly contract, not a second scheduler or
state store.  A binding snapshots the authority that an operation was admitted
against: the canonical root, worktree, capability, resource reservation and
verification obligations.  The existing Goal/DAG, resource and receipt owners
remain authoritative; this document is the compact memory image passed between
those owners and a worker.

The candidate manifest is immutable once published.  It is useful at a crash
boundary because a later process can tell which root, Git revision and resource
policy an attempt was actually bound to instead of inferring them from a live
working directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

from .persist import atomic_write_json


BINDING_SCHEMA = "hawking.execution.binding.v1"
CANDIDATE_MANIFEST_SCHEMA = "hawking.execution.candidate_manifest.v1"


def _string(value: Any, default: str = "") -> str:
    text = str(value if value is not None else default).strip()
    if not text:
        return default
    return text


def _safe_identifier(value: Any, prefix: str) -> str:
    text = _string(value)
    if not text:
        return f"{prefix}-{uuid.uuid4().hex}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", text):
        raise ValueError(f"invalid execution identifier: {text!r}")
    return text


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _paths(
    values: Iterable[Any],
    *,
    root: Optional[Union[str, os.PathLike[str]]] = None,
) -> Tuple[str, ...]:
    result = []
    for value in values or ():
        text = _string(value)
        if root is not None and text:
            # Keep the binding's scope subject to the same path normalizer as
            # typed mutations without importing that module at import time.
            # The lazy import avoids creating a new module cycle while making
            # an A0 manifest fail closed on traversal, absolute escape, or a
            # protected state path.
            from .mutation import normalize_mutation_path

            text = normalize_mutation_path(text, root=root)
        if text and text not in result:
            result.append(text)
    return tuple(sorted(result))


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _git_snapshot(root: Path) -> Dict[str, Any]:
    """Read a bounded Git identity without changing the worktree."""
    result: Dict[str, Any] = {
        "root": None,
        "revision": None,
        "status_sha256": None,
        "status_state": "unavailable",
    }
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if top.returncode != 0 or not top.stdout.strip():
            return result
        result["root"] = str(Path(top.stdout.strip()).resolve())
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if head.returncode == 0 and head.stdout.strip():
            result["revision"] = head.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=False,
            timeout=10,
            check=False,
        )
        if status.returncode == 0:
            result["status_sha256"] = hashlib.sha256(status.stdout).hexdigest()
            result["status_state"] = "clean" if not status.stdout else "dirty"
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _resource_snapshot(root: Path) -> Dict[str, Any]:
    """Return the current resource policy without duplicating its authority."""
    try:
        from .resources import ResourceLimits

        limits = ResourceLimits.resolve(repo_root=root)
        return {
            "gpu_decode": int(limits.gpu_decode),
            "gpu_decode_source": str(limits.gpu_decode_source),
            "gpu_exclusive": int(limits.gpu_exclusive),
            "mutation": int(limits.mutation),
            "cpu_heavy": int(limits.cpu_heavy),
            "compile": int(limits.compile),
            "test": int(limits.test),
            "test_authoring": int(limits.test_authoring),
            "static_analysis": int(limits.static_analysis),
            "memory_heavy": int(limits.memory_heavy),
            "io_heavy": int(limits.io_heavy),
            "tool_wait": int(limits.tool_wait),
            "light_control": int(limits.light_control),
            "hawking": (int(limits.hawking) if limits.hawking is not None else None),
            "hawking_source": str(limits.hawking_source),
        }
    except Exception as exc:  # pragma: no cover - optional native policy
        return {"status": "unavailable", "reason": type(exc).__name__}


@dataclass(frozen=True)
class ExecutionBinding:
    """The immutable authority envelope for one worker attempt."""

    request_id: str
    goal_id: str
    workunit_id: str
    worker_attempt_id: str
    capability_id: str
    capability_schema: str
    worktree_id: str
    canonical_root: str
    allowed_paths: Tuple[str, ...] = field(default_factory=tuple)
    verification_obligations: Mapping[str, Any] = field(default_factory=dict)
    budget: Mapping[str, Any] = field(default_factory=dict)
    resource_reservation: Mapping[str, Any] = field(default_factory=dict)
    lease_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _safe_identifier(self.request_id, "REQ"))
        object.__setattr__(self, "goal_id", _string(self.goal_id))
        object.__setattr__(self, "workunit_id", _string(self.workunit_id))
        object.__setattr__(self, "worker_attempt_id", _safe_identifier(self.worker_attempt_id, "ATTEMPT"))
        object.__setattr__(self, "capability_id", _string(self.capability_id, "hawking.unknown"))
        object.__setattr__(self, "capability_schema", _string(self.capability_schema, BINDING_SCHEMA))
        object.__setattr__(self, "worktree_id", _string(self.worktree_id, "worktree-unknown"))
        object.__setattr__(
            self,
            "canonical_root",
            str(Path(self.canonical_root).expanduser().resolve()),
        )
        object.__setattr__(
            self,
            "allowed_paths",
            _paths(self.allowed_paths, root=Path(self.canonical_root)),
        )
        object.__setattr__(self, "verification_obligations", _json_copy(_mapping(self.verification_obligations)))
        object.__setattr__(self, "budget", _json_copy(_mapping(self.budget)))
        object.__setattr__(self, "resource_reservation", _json_copy(_mapping(self.resource_reservation)))
        if self.lease_id is not None:
            object.__setattr__(self, "lease_id", _safe_identifier(self.lease_id, "LEASE"))

    @classmethod
    def create(
        cls,
        root: Union[str, os.PathLike[str]],
        *,
        goal_id: Any = "",
        workunit_id: Any = "",
        worker_attempt_id: Any = None,
        capability_id: str = "hawking.mutation",
        capability_schema: str = BINDING_SCHEMA,
        worktree_id: Any = None,
        allowed_paths: Iterable[Any] = (),
        verification_obligations: Optional[Mapping[str, Any]] = None,
        budget: Optional[Mapping[str, Any]] = None,
        resource_reservation: Optional[Mapping[str, Any]] = None,
        lease_id: Any = None,
    ) -> "ExecutionBinding":
        canonical = Path(root).expanduser().resolve()
        worktree = _string(worktree_id) or f"worktree-{hashlib.sha256(str(canonical).encode()).hexdigest()[:16]}"
        return cls(
            request_id=f"REQ-{uuid.uuid4().hex}",
            goal_id=_string(goal_id),
            workunit_id=_string(workunit_id),
            worker_attempt_id=_safe_identifier(worker_attempt_id, "ATTEMPT"),
            capability_id=capability_id,
            capability_schema=capability_schema,
            worktree_id=worktree,
            canonical_root=str(canonical),
            allowed_paths=_paths(allowed_paths),
            verification_obligations=_mapping(verification_obligations),
            budget=_mapping(budget),
            resource_reservation=_mapping(resource_reservation),
            lease_id=lease_id,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionBinding":
        if not isinstance(value, Mapping):
            raise TypeError("execution binding must be an object")
        schema = value.get("schema")
        if schema not in (None, BINDING_SCHEMA):
            raise ValueError(f"unsupported execution binding schema: {schema}")
        return cls(
            request_id=value.get("request_id") or f"REQ-{uuid.uuid4().hex}",
            goal_id=value.get("goal_id") or "",
            workunit_id=value.get("workunit_id") or "",
            worker_attempt_id=value.get("worker_attempt_id") or f"ATTEMPT-{uuid.uuid4().hex}",
            capability_id=value.get("capability_id") or "hawking.unknown",
            capability_schema=value.get("capability_schema") or BINDING_SCHEMA,
            worktree_id=value.get("worktree_id") or "worktree-unknown",
            canonical_root=value.get("canonical_root") or ".",
            allowed_paths=value.get("allowed_paths") or (),
            verification_obligations=value.get("verification_obligations") or {},
            budget=value.get("budget") or {},
            resource_reservation=value.get("resource_reservation") or {},
            lease_id=value.get("lease_id"),
            created_at=float(value.get("created_at") or time.time()),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": BINDING_SCHEMA,
            "request_id": self.request_id,
            "goal_id": self.goal_id,
            "workunit_id": self.workunit_id,
            "worker_attempt_id": self.worker_attempt_id,
            "capability_id": self.capability_id,
            "capability_schema": self.capability_schema,
            "worktree_id": self.worktree_id,
            "canonical_root": self.canonical_root,
            "allowed_paths": list(self.allowed_paths),
            "verification_obligations": _json_copy(self.verification_obligations),
            "budget": _json_copy(self.budget),
            "resource_reservation": _json_copy(self.resource_reservation),
            "lease_id": self.lease_id,
            "created_at": self.created_at,
        }

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateManifest:
    """A durable, content-addressed admission snapshot."""

    manifest_id: str
    binding: ExecutionBinding
    git: Mapping[str, Any]
    resource_policy: Mapping[str, Any]
    capabilities: Tuple[str, ...] = field(default_factory=tuple)
    protected_paths: Tuple[str, ...] = (".git", ".hawking")
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": CANDIDATE_MANIFEST_SCHEMA,
            "manifest_id": self.manifest_id,
            "created_at": self.created_at,
            "binding": self.binding.to_dict(),
            "binding_sha256": self.binding.digest(),
            "git": _json_copy(self.git),
            "resource_policy": _json_copy(self.resource_policy),
            "capabilities": list(self.capabilities),
            "protected_paths": list(self.protected_paths),
        }

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_candidate_manifest(
    root: Union[str, os.PathLike[str]],
    *,
    goal_id: Any = "",
    workunit_id: Any = "",
    worker_attempt_id: Any = None,
    capability_id: str = "hawking.execution",
    allowed_paths: Iterable[Any] = (),
    verification_obligations: Optional[Mapping[str, Any]] = None,
    budget: Optional[Mapping[str, Any]] = None,
    resource_reservation: Optional[Mapping[str, Any]] = None,
    capabilities: Iterable[Any] = (),
    lease_id: Any = None,
) -> CandidateManifest:
    canonical = Path(root).expanduser().resolve()
    binding = ExecutionBinding.create(
        canonical,
        goal_id=goal_id,
        workunit_id=workunit_id,
        worker_attempt_id=worker_attempt_id,
        capability_id=capability_id,
        allowed_paths=allowed_paths,
        verification_obligations=verification_obligations,
        budget=budget,
        resource_reservation=resource_reservation,
        lease_id=lease_id,
    )
    git = _git_snapshot(canonical)
    policy = _resource_snapshot(canonical)
    capability_names = _paths(capabilities)
    identity = {
        "root": str(canonical),
        "binding": binding.digest(),
        "git": git,
        "resource_policy": policy,
        "capabilities": capability_names,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return CandidateManifest(
        manifest_id=f"CANDIDATE-{digest[:24]}",
        binding=binding,
        git=git,
        resource_policy=policy,
        capabilities=capability_names,
    )


def persist_candidate_manifest(root: Union[str, os.PathLike[str]], manifest: CandidateManifest) -> Path:
    """Publish once; a different document may not replace an admitted one."""
    destination = Path(root).expanduser().resolve() / ".hawking" / "manifests" / "candidates" / f"{manifest.manifest_id}.json"
    payload = manifest.to_dict()
    if destination.is_file():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"candidate manifest is unreadable: {destination}") from exc
        if existing != payload:
            raise ValueError(f"candidate manifest is immutable: {destination}")
        return destination
    atomic_write_json(destination, payload)
    return destination


__all__ = [
    "BINDING_SCHEMA",
    "CANDIDATE_MANIFEST_SCHEMA",
    "ExecutionBinding",
    "CandidateManifest",
    "build_candidate_manifest",
    "persist_candidate_manifest",
]
