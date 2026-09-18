"""Hawking-owned macOS WorldState and one-shot semantic action lease.

The Swift helper is mechanics only.  This module owns durable observations,
lease admission, preemption/release, post-action verification, and receipts.
There is deliberately no coordinate mouse, keyboard, CGEvent, shell, or
arbitrary accessibility escape here: the only admitted action is one exact
semantic ``AXPress`` inside an explicitly named application scope.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .native_owner import NativeOwnerError, inspect_code_identity, resolve_native_helper
from .persist import atomic_write_json


MACOS_WORLD_SCHEMA = "hawking.macos.world_state.v1"
MACOS_HELPER_SCHEMA = "hawking.macos.helper.v1"
MACOS_LEASE_SCHEMA = "hawking.macos.action_lease.v1"
MACOS_ACTION_SCHEMA = "hawking.macos.action_receipt.v1"
MAX_HISTORY = 128
MAX_LEASE_TTL_S = 120.0


class MacOSWorldError(RuntimeError):
    """A stable, non-authoritative native-helper failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


def _safe_target(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only stable semantic target identity; never persist arbitrary input."""
    result: Dict[str, Any] = {}
    for key in ("application", "bundle_id", "pid", "role", "title", "label", "identifier", "value"):
        if key not in value or value.get(key) in (None, ""):
            continue
        item = value.get(key)
        if key == "pid":
            try:
                item = int(item)
            except (TypeError, ValueError):
                continue
        else:
            item = str(item).strip()[:240]
            if not item:
                continue
        result[key] = item
    return result


def _now() -> float:
    return time.time()


def _root(workspace: Path) -> Path:
    value = workspace.resolve() / ".hawking" / "macos-world"
    value.mkdir(parents=True, exist_ok=True)
    return value


def _identity(row: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    return ":".join(str(row.get(key) or "") for key in keys)


class MacOSWorldStore:
    """One durable desktop WorldState per Hawking workspace."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = _root(self.workspace)
        self.path = self.root / "world.json"
        self._lock = threading.RLock()

    @staticmethod
    def _empty() -> Dict[str, Any]:
        now = _now()
        return {
            "schema": MACOS_WORLD_SCHEMA,
            "revision": 0,
            "created_at": now,
            "updated_at": now,
            "snapshot": {},
            "history": [],
            "helper": {"schema": MACOS_HELPER_SCHEMA, "state": "UNKNOWN"},
        }

    def load(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return self._empty()
        return value if isinstance(value, dict) and value.get("schema") == MACOS_WORLD_SCHEMA else self._empty()

    def save(self, document: Mapping[str, Any]) -> None:
        atomic_write_json(self.path, dict(document))

    @staticmethod
    def _changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
        def changed(kind: str, keys: tuple[str, ...]) -> list[Dict[str, str]]:
            old_rows = before.get(kind) if isinstance(before.get(kind), list) else []
            new_rows = after.get(kind) if isinstance(after.get(kind), list) else []
            old = {_identity(row, keys): row for row in old_rows if isinstance(row, Mapping)}
            new = {_identity(row, keys): row for row in new_rows if isinstance(row, Mapping)}
            rows: list[Dict[str, str]] = []
            for key in sorted(set(old) | set(new)):
                if old.get(key) == new.get(key):
                    continue
                rows.append({"identity": key, "kind": "added" if key not in old else "removed" if key not in new else "changed"})
            return rows[:64]

        return {
            "applications": changed("applications", ("pid",)),
            "windows": changed("windows", ("window_id",)),
            "focus_changed": before.get("focused_app") != after.get("focused_app")
            or before.get("focused_element") != after.get("focused_element"),
        }

    def record(self, snapshot: Mapping[str, Any], helper: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            document = self.load()
            previous = document.get("snapshot") if isinstance(document.get("snapshot"), Mapping) else {}
            current = dict(snapshot)
            revision = int(document.get("revision") or 0) + 1
            changes = self._changes(previous, current)
            event = {
                "revision": revision,
                "at": _now(),
                "operation": "observe",
                "changes": changes,
                "freshness": "CURRENT",
            }
            document.update({
                "revision": revision,
                "updated_at": event["at"],
                "snapshot": current,
                "helper": dict(helper),
                "history": [*(document.get("history") or []), event][-MAX_HISTORY:],
            })
            self.save(document)
            return {
                "schema": MACOS_WORLD_SCHEMA,
                "world_revision": revision,
                "changes": changes,
                "snapshot": current,
                "freshness": "CURRENT",
            }

    def changed(self, since_revision: Any) -> Dict[str, Any]:
        try:
            since = max(0, int(since_revision))
        except (TypeError, ValueError):
            raise MacOSWorldError("INVALID_ARGUMENTS", "since_revision must be an integer")
        with self._lock:
            document = self.load()
            current = int(document.get("revision") or 0)
            history = [row for row in document.get("history") or [] if isinstance(row, Mapping)]
            oldest = int(history[0].get("revision") or current) if history else current
            gap = bool(history and since < oldest - 1)
            return {
                "schema": MACOS_WORLD_SCHEMA,
                "since_revision": since,
                "world_revision": current,
                "gap": gap,
                "snapshot_required": gap,
                "events": [dict(row) for row in history if int(row.get("revision") or 0) > since][-64:],
            }


class MacOSInputLeaseStore:
    """Durable single-owner lease for one short semantic desktop action."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = _root(self.workspace)
        self.path = self.root / "action-lease.json"
        self.receipts = self.workspace / "receipts" / "future" / "macos"
        self._lock = threading.RLock()

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {"schema": MACOS_LEASE_SCHEMA, "state": "NONE", "lease_id": None}

    def load(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return self._empty()
        return value if isinstance(value, dict) and value.get("schema") == MACOS_LEASE_SCHEMA else self._empty()

    def _save(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        document = dict(value)
        document["schema"] = MACOS_LEASE_SCHEMA
        atomic_write_json(self.path, document)
        return document

    def _receipt(self, *, operation: str, lease: Mapping[str, Any], outcome: str,
                 reason: str = "", details: Optional[Mapping[str, Any]] = None) -> str:
        self.receipts.mkdir(parents=True, exist_ok=True)
        lease_id = str(lease.get("lease_id") or "none")
        safe = "".join(ch for ch in lease_id if ch.isalnum() or ch in "-_")[:80] or "none"
        path = self.receipts / f"{safe}-{operation}-{uuid.uuid4().hex[:10]}.json"
        document = {
            "schema": MACOS_ACTION_SCHEMA,
            "operation": str(operation),
            "outcome": str(outcome),
            "at": _now(),
            "lease_id": lease.get("lease_id"),
            "goal_id": lease.get("goal_id"),
            "workunit_id": lease.get("workunit_id"),
            "worker_id": lease.get("worker_id"),
            "target": dict(lease.get("target") or {}),
            "state": lease.get("state"),
            "reason": str(reason or "")[:400],
            "details": dict(details or {}),
            "claim_boundary": (
                "Hawking admitted one scoped semantic AXPress; the native helper "
                "performed mechanics and did not own authority."
            ),
        }
        atomic_write_json(path, document)
        return str(path)

    @staticmethod
    def _expired(lease: Mapping[str, Any]) -> bool:
        try:
            return float(lease.get("expires_at") or 0.0) <= _now()
        except (TypeError, ValueError):
            return True

    def _expire_if_needed(self, lease: Dict[str, Any]) -> Dict[str, Any]:
        if lease.get("state") == "ACTIVE" and self._expired(lease):
            lease["state"] = "EXPIRED"
            lease["released_at"] = _now()
            lease["release_reason"] = "lease_ttl_expired"
            lease["receipt"] = self._receipt(
                operation="expire", lease=lease, outcome="EXPIRED", reason="lease_ttl_expired"
            )
            self._save(lease)
        return lease

    def acquire(
        self, *, goal_id: str, workunit_id: str, worker_id: str,
        target: Mapping[str, Any], ttl_s: Any = 30.0, reason: str = "",
    ) -> Dict[str, Any]:
        clean_target = _safe_target(target)
        if not clean_target.get("application") and not clean_target.get("bundle_id"):
            raise MacOSWorldError(
                "INVALID_TARGET", "a lease target must name an exact application or bundle_id"
            )
        try:
            ttl = min(MAX_LEASE_TTL_S, max(1.0, float(ttl_s)))
        except (TypeError, ValueError):
            raise MacOSWorldError("INVALID_ARGUMENTS", "lease ttl must be numeric")
        owner_goal = str(goal_id or "").strip()
        owner_workunit = str(workunit_id or "").strip()
        if not owner_goal or not owner_workunit:
            raise MacOSWorldError("AUTHORITY_REQUIRED", "Goal and WorkUnit identity are required for an action lease")
        with self._lock:
            current = self._expire_if_needed(self.load())
            if current.get("state") == "ACTIVE":
                if (
                    current.get("goal_id") == owner_goal
                    and current.get("workunit_id") == owner_workunit
                    and current.get("target") == clean_target
                ):
                    return {**current, "reused": True}
                raise MacOSWorldError("LEASE_BUSY", "another Goal owns the active macOS action lease")
            lease = {
                "schema": MACOS_LEASE_SCHEMA,
                "lease_id": f"MACOS-LEASE-{uuid.uuid4().hex[:16].upper()}",
                "state": "ACTIVE",
                "goal_id": owner_goal,
                "workunit_id": owner_workunit,
                "worker_id": str(worker_id or "").strip(),
                "target": clean_target,
                "acquired_at": _now(),
                "expires_at": _now() + ttl,
                "ttl_s": ttl,
                "preemptible": True,
                "semantic_action": "AXPress",
                "acquire_reason": str(reason or "")[:400],
            }
            lease["receipt"] = self._receipt(
                operation="acquire", lease=lease, outcome="ADMITTED", reason=reason
            )
            self._save(lease)
            return dict(lease)

    def require_active(
        self, lease_id: str, *, goal_id: str, workunit_id: str,
        target: Mapping[str, Any],
    ) -> Dict[str, Any]:
        with self._lock:
            lease = self._expire_if_needed(self.load())
            if lease.get("state") != "ACTIVE":
                raise MacOSWorldError("LEASE_REQUIRED", "an active Hawking action lease is required")
            if str(lease.get("lease_id") or "") != str(lease_id or ""):
                raise MacOSWorldError("LEASE_MISMATCH", "the action lease does not match the request")
            if lease.get("goal_id") != str(goal_id or "") or lease.get("workunit_id") != str(workunit_id or ""):
                raise MacOSWorldError("LEASE_OWNER_MISMATCH", "the action lease belongs to another Goal or WorkUnit")
            requested = _safe_target(target)
            admitted = _safe_target(lease.get("target") or {})
            for key in ("application", "bundle_id", "pid"):
                if admitted.get(key) is not None and requested.get(key) not in (None, admitted.get(key)):
                    raise MacOSWorldError("TARGET_OUTSIDE_LEASE", "the semantic target is outside the admitted lease")
            return dict(lease)

    def _transition(self, lease_id: str, *, state: str, goal_id: str, workunit_id: str,
                    reason: str) -> Dict[str, Any]:
        with self._lock:
            lease = self.load()
            if str(lease.get("lease_id") or "") != str(lease_id or ""):
                raise MacOSWorldError("LEASE_MISMATCH", "unknown macOS action lease")
            if lease.get("goal_id") != str(goal_id or "") or lease.get("workunit_id") != str(workunit_id or ""):
                raise MacOSWorldError("LEASE_OWNER_MISMATCH", "the lease belongs to another Goal or WorkUnit")
            if lease.get("state") == "ACTIVE":
                lease["state"] = state
                lease["released_at"] = _now()
                lease["release_reason"] = str(reason or "")[:400]
                lease["receipt"] = self._receipt(
                    operation="release" if state == "RELEASED" else "preempt",
                    lease=lease, outcome=state, reason=reason,
                )
                self._save(lease)
            return dict(lease)

    def release(self, lease_id: str, *, goal_id: str, workunit_id: str, reason: str = "") -> Dict[str, Any]:
        return self._transition(lease_id, state="RELEASED", goal_id=goal_id, workunit_id=workunit_id, reason=reason or "owner_release")

    def preempt(self, lease_id: str, *, goal_id: str, workunit_id: str, reason: str = "") -> Dict[str, Any]:
        return self._transition(lease_id, state="PREEMPTED", goal_id=goal_id, workunit_id=workunit_id, reason=reason or "user_input_preemption")


class MacOSRuntime:
    """Invoke the installed stable native owner; it never owns Goal state.

    Older development code compiled ``native/hawking-macos`` into the current
    workspace the first time a desktop observation was requested.  That made
    the executable identity depend on a checkout and an ad-hoc build.  The
    runtime now resolves only the native-owner release store (or an explicit,
    visible fixture override) and never compiles on a Goal's critical path.
    """

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = _root(self.workspace)
        self.project_root = self._project_root()
        self._lock = threading.RLock()

    def _project_root(self) -> Path:
        explicit = os.environ.get("HAWKING_MACOS_NATIVE_ROOT")
        candidates = [Path(explicit).expanduser()] if explicit else []
        candidates.extend((self.workspace, *self.workspace.parents, Path(__file__).resolve().parents[1]))
        for candidate in candidates:
            if (candidate / "native" / "hawking-macos" / "hawking_macos.swift").is_file():
                return candidate.resolve()
        return self.workspace

    @property
    def source(self) -> Path:
        return self.project_root / "native" / "hawking-macos" / "hawking_macos.swift"

    @property
    def binary(self) -> Path:
        try:
            path, _identity = resolve_native_helper(self.workspace)
            return path
        except NativeOwnerError:
            # Keep a deterministic diagnostic path for callers that display the
            # missing owner; this property does not make the path runnable.
            return self.root / "bin" / "hawking-native-helper"

    def _ensure_binary(self) -> Path:
        if sys.platform != "darwin":
            raise MacOSWorldError("UNSUPPORTED_OS", "Hawking macOS WorldState requires macOS")
        try:
            binary, identity = resolve_native_helper(self.workspace)
        except NativeOwnerError as exc:
            raise MacOSWorldError("MISSING_DEPENDENCY", str(exc)) from exc
        # An explicitly selected fixture is useful for tests and development,
        # but the native owner is not allowed to silently become an unsigned
        # production authority.  The opt-in is deliberately named and is
        # never set by the normal launcher.
        if not identity.get("stable_owner") and not (
            identity.get("owner_mode") == "explicit_override"
            and os.environ.get("HAWKING_ALLOW_EPHEMERAL_NATIVE_HELPER") == "1"
        ):
            raise MacOSWorldError(
                "CODE_IDENTITY_REQUIRED",
                "native helper is not a verified stable Hawking permission owner",
            )
        return binary

    def call(self, operation: str, arguments: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        with self._lock:
            binary = self._ensure_binary()
            request = json.dumps({"id": f"macos-{uuid.uuid4().hex}", "operation": operation, **dict(arguments or {})}) + "\n"
            try:
                completed = subprocess.run([str(binary)], input=request, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30.0, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise MacOSWorldError("HELPER_UNAVAILABLE", f"macOS helper did not respond: {type(exc).__name__}") from exc
        line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        try:
            response = json.loads(line)
        except (ValueError, TypeError):
            raise MacOSWorldError("HELPER_PROTOCOL", "macOS helper returned invalid JSON")
        if not isinstance(response, Mapping):
            raise MacOSWorldError("HELPER_PROTOCOL", "macOS helper response was not an object")
        if not response.get("ok"):
            raise MacOSWorldError("HELPER_REJECTED", str(response.get("error") or "macOS helper rejected request"))
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise MacOSWorldError("HELPER_PROTOCOL", "macOS helper omitted result")
        return dict(result)

    def health(self) -> Dict[str, Any]:
        try:
            binary = self._ensure_binary()
            identity = inspect_code_identity(binary)
            return {
                "schema": MACOS_HELPER_SCHEMA,
                "health": "READY",
                "binary": str(binary),
                "code_identity": identity,
                **self.call("health"),
            }
        except MacOSWorldError as exc:
            return {
                "schema": MACOS_HELPER_SCHEMA,
                "health": "UNAVAILABLE",
                "failure_class": exc.code,
                "message": str(exc),
            }


class MacOSWorld:
    """Typed native facade with an explicit, one-shot semantic action door."""

    def __init__(self, workspace: str | os.PathLike[str], *, runtime: MacOSRuntime | Any | None = None) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.store = MacOSWorldStore(self.workspace)
        self.leases = MacOSInputLeaseStore(self.workspace)
        self.runtime = runtime if runtime is not None else MacOSRuntime(self.workspace)

    def health(self) -> Dict[str, Any]:
        return self.runtime.health()

    def observe(self) -> Dict[str, Any]:
        snapshot = self.runtime.call("observe")
        helper = self.runtime.health()
        return self.store.record(snapshot, helper)

    @staticmethod
    def _identity(authority: Optional[Mapping[str, Any]], arguments: Mapping[str, Any]) -> tuple[str, str, str]:
        authority = authority if isinstance(authority, Mapping) else {}
        return (
            str(authority.get("goal_id") or arguments.get("goal_id") or "").strip(),
            str(authority.get("workunit_id") or arguments.get("workunit_id") or "").strip(),
            str(authority.get("worker_id") or arguments.get("worker_id") or "").strip(),
        )

    @staticmethod
    def _require_action_authority(authority: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not isinstance(authority, Mapping) or authority.get("actions") is not True:
            raise MacOSWorldError("AUTHORITY_REQUIRED", "semantic macOS actions are not admitted for this Goal")
        return dict(authority)

    @staticmethod
    def _application_target(arguments: Mapping[str, Any]) -> Dict[str, Any]:
        value = arguments.get("target")
        target = dict(value) if isinstance(value, Mapping) else {}
        for key in ("application", "bundle_id", "pid"):
            if arguments.get(key) not in (None, "") and key not in target:
                target[key] = arguments[key]
        return _safe_target(target)

    @staticmethod
    def _authority_allows_app(authority: Mapping[str, Any], app: Mapping[str, Any]) -> bool:
        applications = {str(item) for item in authority.get("applications") or []}
        bundles = {str(item) for item in authority.get("bundle_ids") or []}
        return (
            (not applications and not bundles)
            or str(app.get("name") or "") in applications
            or str(app.get("bundle_id") or "") in bundles
        )

    def _resolve_app(self, snapshot: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Any]:
        rows = snapshot.get("applications") if isinstance(snapshot.get("applications"), list) else []
        matches = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if target.get("application") and str(row.get("name") or "") != str(target["application"]):
                continue
            if target.get("bundle_id") and str(row.get("bundle_id") or "") != str(target["bundle_id"]):
                continue
            if target.get("pid") is not None and int(row.get("pid") or 0) != int(target["pid"]):
                continue
            matches.append(dict(row))
        if not matches:
            raise MacOSWorldError("TARGET_NOT_FOUND", "the exact semantic application was not observed")
        if len(matches) > 1:
            raise MacOSWorldError("TARGET_AMBIGUOUS", "the semantic application resolved to more than one process")
        return matches[0]

    def _find(self, snapshot: Mapping[str, Any], target: Mapping[str, Any], max_results: Any = 8) -> Dict[str, Any]:
        app = self._resolve_app(snapshot, target)
        try:
            limit = min(16, max(1, int(max_results)))
        except (TypeError, ValueError):
            limit = 8
        element_target = {
            key: value for key, value in target.items()
            if key not in {"application", "bundle_id", "pid"}
        }
        found = self.runtime.call("find", {
            "pid": int(app.get("pid") or 0),
            "target": element_target,
            "max_results": limit,
        })
        return {
            "world_revision": int(self.store.load().get("revision") or 0),
            "application": app,
            "target": element_target,
            "matches": list(found.get("matches") or [])[:limit],
            "match_count": len(list(found.get("matches") or [])),
        }

    def _failure_receipt(self, operation: str, *, authority: Optional[Mapping[str, Any]], target: Mapping[str, Any], error: MacOSWorldError) -> None:
        goal_id, workunit_id, worker_id = self._identity(authority, {})
        lease = {
            "lease_id": None,
            "goal_id": goal_id,
            "workunit_id": workunit_id,
            "worker_id": worker_id,
            "target": _safe_target(target),
            "state": "REFUSED",
        }
        try:
            self.leases._receipt(
                operation=operation, lease=lease, outcome="REFUSED",
                reason=f"{error.code}: {error}",
            )
        except Exception:
            pass

    def _press(self, arguments: Mapping[str, Any], authority: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        admitted = self._require_action_authority(authority)
        goal_id, workunit_id, worker_id = self._identity(admitted, arguments)
        target = self._application_target(arguments)
        lease_id = str(arguments.get("lease_id") or "").strip()
        lease = self.leases.require_active(
            lease_id, goal_id=goal_id, workunit_id=workunit_id, target=target,
        )
        try:
            snapshot_before = self.observe()
            app = self._resolve_app(snapshot_before["snapshot"], target)
            if not self._authority_allows_app(admitted, app):
                raise MacOSWorldError("TARGET_OUTSIDE_AUTHORITY", "the application is outside the Goal's desktop authority")
            focused = snapshot_before["snapshot"].get("focused_app")
            focus_matches = bool(
                isinstance(focused, Mapping)
                and int(focused.get("pid") or 0) == int(app.get("pid") or 0)
                and (
                    not app.get("name")
                    or str(focused.get("name") or "") == str(app.get("name") or "")
                )
                and (
                    not app.get("bundle_id")
                    or str(focused.get("bundle_id") or "") == str(app.get("bundle_id") or "")
                )
            )
            if not focus_matches:
                preempted = self.leases.preempt(
                    lease_id, goal_id=goal_id, workunit_id=workunit_id,
                    reason="user_input_preemption_focus_changed",
                )
                raise MacOSWorldError("PREEMPTED", f"user focus preempted the semantic action lease ({preempted.get('state')})")
            before = self._find(snapshot_before["snapshot"], target)
            if before["match_count"] != 1:
                raise MacOSWorldError("TARGET_NOT_UNIQUE", "semantic action requires exactly one AX target")
            self.runtime.call("press", {
                "pid": int(app.get("pid") or 0),
                "target": before["target"],
            })
            snapshot_after = self.observe()
            verification_target = arguments.get("verify")
            if not isinstance(verification_target, Mapping):
                raise MacOSWorldError("VERIFICATION_REQUIRED", "press requires a semantic post-action verification target")
            after = self._find(snapshot_after["snapshot"], self._application_target({**dict(arguments), "target": verification_target}))
            verified = after["match_count"] == 1
            if not verified:
                raise MacOSWorldError("VERIFICATION_FAILED", "the intended semantic state was not observed after AXPress")
            released = self.leases.release(
                lease_id, goal_id=goal_id, workunit_id=workunit_id,
                reason="post_action_observation_complete",
            )
            action_receipt = self.leases._receipt(
                operation="press",
                lease=lease,
                outcome="VERIFIED",
                reason="semantic AXPress completed with post-action observation",
                details={
                    "target": before,
                    "verification": after,
                    "before_revision": snapshot_before["world_revision"],
                    "after_revision": snapshot_after["world_revision"],
                    "revision_delta": snapshot_after["world_revision"] - snapshot_before["world_revision"],
                    "verified": True,
                },
            )
            return {
                "schema": MACOS_ACTION_SCHEMA,
                "operation": "press",
                "ok": True,
                "lease_id": lease_id,
                "lease_state": released.get("state"),
                "target": before,
                "verification": after,
                "verified": True,
                "before_revision": snapshot_before["world_revision"],
                "after_revision": snapshot_after["world_revision"],
                "revision_delta": snapshot_after["world_revision"] - snapshot_before["world_revision"],
                "receipt": action_receipt,
                "lease_receipt": released.get("receipt"),
                "worker_id": worker_id,
                "goal_id": goal_id,
                "workunit_id": workunit_id,
            }
        except MacOSWorldError as exc:
            try:
                current = self.leases.load()
                if current.get("state") == "ACTIVE" and current.get("lease_id") == lease_id:
                    self.leases.release(
                        lease_id, goal_id=goal_id, workunit_id=workunit_id,
                        reason=f"action_refused:{exc.code}",
                    )
            except Exception:
                pass
            self._failure_receipt("press", authority=admitted, target=target, error=exc)
            raise

    def invoke(
        self, operation: str, arguments: Mapping[str, Any],
        *, authority: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if operation == "health":
            return self.health()
        if operation == "observe":
            return self.observe()
        if operation == "world.changed":
            return self.store.changed(arguments.get("since_revision"))
        if operation == "find":
            snapshot = self.observe()
            target = self._application_target(arguments)
            result = self._find(snapshot["snapshot"], target, arguments.get("max_results", 8))
            return {"schema": MACOS_ACTION_SCHEMA, "operation": "find", "verified": result["match_count"] == 1, **result}
        if operation == "verify":
            snapshot = self.observe()
            target = self._application_target(arguments)
            result = self._find(snapshot["snapshot"], target, arguments.get("max_results", 8))
            return {"schema": MACOS_ACTION_SCHEMA, "operation": "verify", "verified": result["match_count"] == 1, **result}
        if operation == "lease.acquire":
            try:
                admitted = self._require_action_authority(authority)
                goal_id, workunit_id, worker_id = self._identity(admitted, arguments)
                target = self._application_target(arguments)
                app_scope = str(target.get("application") or "")
                bundle_scope = str(target.get("bundle_id") or "")
                allowed_apps = {str(item) for item in admitted.get("applications") or []}
                allowed_bundles = {str(item) for item in admitted.get("bundle_ids") or []}
                if (allowed_apps or allowed_bundles) and app_scope not in allowed_apps and bundle_scope not in allowed_bundles:
                    raise MacOSWorldError("TARGET_OUTSIDE_AUTHORITY", "lease target is outside the Goal's desktop authority")
                return self.leases.acquire(
                    goal_id=goal_id, workunit_id=workunit_id, worker_id=worker_id,
                    target=target, ttl_s=arguments.get("ttl_s", admitted.get("lease_ttl_s", 30.0)),
                    reason=str(arguments.get("reason") or "")[:400],
                )
            except MacOSWorldError as exc:
                self._failure_receipt("acquire", authority=authority, target=self._application_target(arguments), error=exc)
                raise
        if operation in {"lease.release", "lease.preempt"}:
            admitted = self._require_action_authority(authority)
            goal_id, workunit_id, _worker_id = self._identity(admitted, arguments)
            try:
                transition = (
                    self.leases.preempt if operation == "lease.preempt" else self.leases.release
                )
                return transition(
                    str(arguments.get("lease_id") or ""),
                    goal_id=goal_id, workunit_id=workunit_id,
                    reason=str(arguments.get("reason") or "")[:400],
                )
            except MacOSWorldError as exc:
                self._failure_receipt(operation, authority=admitted, target={}, error=exc)
                raise
        if operation == "press":
            try:
                return self._press(arguments, authority)
            except MacOSWorldError:
                raise
        raise MacOSWorldError("UNSUPPORTED_OPERATION", f"macOS WorldState has no {operation!r} operation")


def macos_tool(
    workspace: str | os.PathLike[str], operation: str, arguments: Mapping[str, Any],
    *, authority: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return MacOSWorld(workspace).invoke(operation, arguments, authority=authority)


__all__ = [
    "MACOS_ACTION_SCHEMA", "MACOS_HELPER_SCHEMA", "MACOS_LEASE_SCHEMA",
    "MACOS_WORLD_SCHEMA", "MacOSInputLeaseStore", "MacOSRuntime", "MacOSWorld",
    "MacOSWorldError", "MacOSWorldStore", "macos_tool",
]


def action_receipt_revision_delta(receipt: Mapping[str, Any]) -> int:
    """Return the verified revision delta recorded by a semantic action."""
    details = receipt.get("details") if isinstance(receipt, Mapping) else None
    value = details.get("revision_delta") if isinstance(details, Mapping) else 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
