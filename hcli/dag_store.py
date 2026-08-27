"""Durable DAG store. Disk is authority.

Writes go to ``<workspace>/.hcli/dag.json`` via a temp file and ``os.replace``.
A unit left ``running`` across a crash comes back as ``interrupted`` — its own
reason, distinct from a verifier failure — and is re-run from the start.
``attempts`` is not changed by the crash. The unit is never silently
``completed``, and nothing here resumes an inference mid-token.

A Grok unit that is still alive is the exception: when a liveness callable is
supplied and ``GrokBridge.status`` reports the task ``running``, the unit is
kept ``running`` and recorded for polling. A grok task that has already failed
is ``failed``. ``stale-running``, a nonzero-unrelated process death, and an
unobservable checker (``None`` or a raised/unknown result) interrupt, not
fail-and-retry. This module does not import ``grok_bridge``.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .persist import atomic_write_json
from .workunit import (
    IdentityConflict,
    WorkUnit,
    admit_unit,
    content_identity,
    mark_interrupted,
    rebuild_repair_budget,
    serialize_repair_budget,
    transition_status,
)

DAG_FILENAME = "dag.json"
DAG_VERSION = 1

# Fields the parent WorkUnit already serializes; kept as a belt so a
# snapshot whose to_dict dropped them still round-trips through disk.
_PERSISTED_EXTRAS = (
    "backend_task_id",
    "assigned_backend",
    "preferred_backend",
    "verifier",
    "verification",
    "classification",
    "provider",
)

GrokLiveness = Callable[[str], Any]


class DagCorruptError(ValueError):
    """Raised when dag.json exists but is not a valid DAG document."""


def _unit_to_disk(wu: WorkUnit) -> Dict[str, Any]:
    data = wu.to_dict()
    for key in _PERSISTED_EXTRAS:
        val = getattr(wu, key, None)
        if val is not None:
            data[key] = val
    return data


def _apply_disk_extras(wu: WorkUnit, payload: Dict[str, Any]) -> None:
    for key in _PERSISTED_EXTRAS:
        if key in payload:
            setattr(wu, key, payload[key])


def _is_grok_dispatched(wu: WorkUnit) -> bool:
    backend = str(
        getattr(wu, "assigned_backend", None)
        or getattr(wu, "preferred_backend", None)
        or ""
    ).strip().lower()
    if backend == "grok":
        return True
    return str(getattr(wu, "resource_class", "") or "") == "GROK"


def _persistable_identity(old: WorkUnit, incoming: WorkUnit) -> bool:
    """True if ``incoming`` is the same work as ``old``, or an annotation of it.

    Admission (``submit`` / ``admit_unit``) still treats any identity-tuple
    change as a conflict. Checkpoint ``save`` is the live graph: steering
    appends a note to a future unit's description and then persists. That is
    the same work evolving, not a silent overwrite of a different plan.
    A wholesale replacement of role, dependencies, verifier, or description
    is still IdentityConflict.
    """
    if content_identity(old) == content_identity(incoming):
        return True
    old_deps = [str(d) for d in (getattr(old, "dependencies", None) or [])]
    new_deps = [str(d) for d in (getattr(incoming, "dependencies", None) or [])]
    if str(getattr(old, "role", None) or "") != str(getattr(incoming, "role", None) or ""):
        return False
    if old_deps != new_deps:
        return False
    if str(getattr(old, "verifier", None) or "") != str(getattr(incoming, "verifier", None) or ""):
        return False
    old_desc = str(getattr(old, "description", None) or "")
    new_desc = str(getattr(incoming, "description", None) or "")
    return new_desc.startswith(old_desc)


def _backend_task_id(wu: WorkUnit) -> Optional[str]:
    tid = getattr(wu, "backend_task_id", None)
    if tid is None:
        return None
    text = str(tid).strip()
    return text or None


def _grok_recovery_decision(
    wu: WorkUnit,
    grok_liveness: Optional[GrokLiveness],
) -> tuple:
    """Return ``(adopt|failed|interrupted, info_or_none)``.

    ``adopt`` keeps a live Grok process running. ``failed`` is a terminal
    grok result (the verifier/process actually finished wrong). Everything
    else is process death / unobservable — interrupt and re-run, do not
    pretend the verifier failed.
    """
    task_id = _backend_task_id(wu)
    if not task_id:
        return "interrupted", None
    if grok_liveness is None:
        if wu.failure_context is None:
            wu.failure_context = {
                "reason": "grok_liveness_unobservable",
                "backend_task_id": task_id,
            }
        return "interrupted", None
    try:
        info = grok_liveness(task_id)
    except Exception as exc:
        wu.failure_context = {
            "reason": "grok_liveness_unobservable",
            "backend_task_id": task_id,
            "error": str(exc),
        }
        return "interrupted", None
    if not isinstance(info, dict):
        return "interrupted", None
    state = str(info.get("state") or "").strip().lower()
    if state == "running":
        return "adopt", info
    if state in {"failed", "error", "errored"}:
        wu.failure_context = {
            "reason": "grok_task_terminal",
            "backend_task_id": task_id,
            "grok_state": state,
            "exit_code": info.get("exit_code"),
        }
        return "failed", info
    if state == "done":
        if info.get("successful") is False:
            wu.failure_context = {
                "reason": "grok_task_terminal",
                "backend_task_id": task_id,
                "grok_state": state,
                "exit_code": info.get("exit_code"),
            }
            return "failed", info
        exit_code = info.get("exit_code")
        if exit_code is None:
            return "adopt", info
        try:
            if int(exit_code) != 0:
                wu.failure_context = {
                    "reason": "grok_task_terminal",
                    "backend_task_id": task_id,
                    "grok_state": state,
                    "exit_code": exit_code,
                }
                return "failed", info
            return "adopt", info
        except (TypeError, ValueError):
            return "failed", info
    # stale-running and anything unknown: process death, not a red test.
    return "interrupted", info


class DagStore:
    def __init__(self, workspace: Union[str, Path]) -> None:
        self.workspace = Path(workspace)
        self.dir = self.workspace / ".hcli"
        self.path = self.dir / DAG_FILENAME
        self.last_meta: Dict[str, Any] = {}
        self.adopted_running: List[Dict[str, Any]] = []
        self.repair_budget: Dict[str, Any] = {"counts": {}, "signatures": {}}

    def exists(self) -> bool:
        return self.path.is_file()

    def _read_document(self) -> Dict[str, Any]:
        if not self.path.is_file():
            raise FileNotFoundError(str(self.path))
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DagCorruptError(f"dag.json unreadable: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DagCorruptError(f"dag.json is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise DagCorruptError("dag.json root is not an object")
        return data

    def _units_from_document(self, data: Dict[str, Any]) -> Dict[str, WorkUnit]:
        units_blob = data.get("units")
        if not isinstance(units_blob, dict):
            raise DagCorruptError("dag.json missing units object")
        units: Dict[str, WorkUnit] = {}
        for uid, payload in units_blob.items():
            if not isinstance(payload, dict):
                raise DagCorruptError(f"unit {uid!r} is not an object")
            try:
                wu = WorkUnit.from_dict(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise DagCorruptError(f"unit {uid!r} is not a valid WorkUnit: {exc}") from exc
            if not wu.id:
                wu.id = str(uid)
            _apply_disk_extras(wu, payload)
            units[str(uid)] = wu
        return units

    def _previous_document(self) -> Optional[Dict[str, Any]]:
        if not self.path.is_file():
            return None
        try:
            return self._read_document()
        except (DagCorruptError, FileNotFoundError, OSError):
            return None

    def _reject_content_conflicts(
        self,
        units: Dict[str, WorkUnit],
        previous: Optional[Dict[str, WorkUnit]],
    ) -> None:
        if not previous:
            return
        for uid, wu in units.items():
            old = previous.get(uid)
            if old is None:
                continue
            if _persistable_identity(old, wu):
                continue
            raise IdentityConflict(uid, old, wu)

    def save(
        self,
        units: Dict[str, WorkUnit],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        previous_doc = self._previous_document()
        previous_units = None
        if previous_doc is not None:
            try:
                previous_units = self._units_from_document(previous_doc)
            except DagCorruptError:
                previous_units = None
        self._reject_content_conflicts(units, previous_units)
        layers: List[Any] = []
        if previous_doc:
            layers.append(
                previous_doc.get("repair_budget")
                or {
                    "counts": previous_doc.get("repair_counts") or {},
                    "signatures": previous_doc.get("repair_signatures") or {},
                }
            )
        if extra:
            extra_budget = extra.get("repair_budget")
            if extra_budget is None and (
                extra.get("repair_counts") is not None
                or extra.get("repair_signatures") is not None
            ):
                extra_budget = {
                    "counts": extra.get("repair_counts") or {},
                    "signatures": extra.get("repair_signatures") or {},
                }
            if extra_budget is not None:
                layers.append(extra_budget)
        floor: Dict[str, Any] = {"counts": {}, "signatures": {}}
        for layer in layers:
            step = rebuild_repair_budget(units, layer)
            for root, n in (step.get("counts") or {}).items():
                floor["counts"][root] = max(int(floor["counts"].get(root, 0)), int(n))
            for root, sigs in (step.get("signatures") or {}).items():
                floor["signatures"].setdefault(root, set()).update(sigs)
        budget = rebuild_repair_budget(units, floor)
        serialized = serialize_repair_budget(budget)
        document: Dict[str, Any] = {
            "version": DAG_VERSION,
            "units": {uid: _unit_to_disk(wu) for uid, wu in units.items()},
            "repair_budget": serialized,
            "repair_counts": serialized["counts"],
            "repair_signatures": serialized["signatures"],
        }
        if extra:
            for key, value in extra.items():
                if key in (
                    "version",
                    "units",
                    "repair_budget",
                    "repair_counts",
                    "repair_signatures",
                ):
                    continue
                document[key] = value
        atomic_write_json(self.path, document)
        self.repair_budget = budget
        self.last_meta = {k: v for k, v in document.items() if k != "units"}

    def load(
        self,
        recover_running: bool = True,
        *,
        grok_liveness: Optional[GrokLiveness] = None,
    ) -> Dict[str, WorkUnit]:
        data = self._read_document()
        units = self._units_from_document(data)
        self.adopted_running = []
        if recover_running:
            for wu in units.values():
                if wu.status != "running":
                    continue
                if _is_grok_dispatched(wu):
                    decision, info = _grok_recovery_decision(wu, grok_liveness)
                    if decision == "adopt":
                        task_id = _backend_task_id(wu)
                        state = info.get("state") if isinstance(info, dict) else "running"
                        self.adopted_running.append(
                            {
                                "unit_id": wu.id,
                                "backend_task_id": task_id,
                                "grok_state": state,
                            }
                        )
                        continue
                    if decision == "failed":
                        transition_status(wu, "failed")
                        wu.assigned_runtime = None
                        continue
                mark_interrupted(wu)
        budget = rebuild_repair_budget(
            units,
            {
                "repair_budget": data.get("repair_budget"),
                "repair_counts": data.get("repair_counts"),
                "repair_signatures": data.get("repair_signatures"),
            },
        )
        self.repair_budget = budget
        self.last_meta = {k: v for k, v in data.items() if k != "units"}
        self.last_meta["repair_budget"] = serialize_repair_budget(budget)
        self.last_meta["repair_counts"] = self.last_meta["repair_budget"]["counts"]
        self.last_meta["repair_signatures"] = self.last_meta["repair_budget"]["signatures"]
        return units

    def submit(self, wu: WorkUnit) -> Any:
        """Admit one unit into the durable DAG.

        Same content is idempotent. Same id with different content raises
        IdentityConflict and does not write.
        """
        units: Dict[str, WorkUnit] = {}
        if self.exists():
            units = self.load(recover_running=False)
        outcome = admit_unit(units, wu)
        self.save(units)
        return outcome
