"""Hawking-owned Playwright session and Browser WorldState boundary.

Playwright is deliberately only a mechanics sidecar.  This module owns the
canonical session identifier, local helper lifecycle, durable observations,
revision history, artifacts, and the projection returned to workers.  It never
creates Goals, grants permissions, or interprets model output as authority.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .persist import atomic_write_json

try:  # Unix is the supported local production host for the sidecar.
    import fcntl
except ImportError:  # pragma: no cover - a clear health result is preferable.
    fcntl = None  # type: ignore[assignment]


BROWSER_WORLD_SCHEMA = "hawking.browser.world_state.v1"
BROWSER_HELPER_SCHEMA = "hawking.browser.helper.v1"
MAX_HISTORY = 128
MAX_EVENTS = 64
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,120}$")
_RUNTIMES: Dict[str, "BrowserRuntime"] = {}
_RUNTIMES_LOCK = threading.Lock()


class BrowserWorldError(RuntimeError):
    """A local browser capability failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


def _safe_id(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not _ID_RE.fullmatch(text):
        raise BrowserWorldError("INVALID_ARGUMENTS", f"{label} must be a safe identifier")
    return text


def _now() -> float:
    return time.time()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _world_root(workspace: Path) -> Path:
    root = workspace.resolve() / ".hawking" / "browser-world"
    root.mkdir(parents=True, exist_ok=True)
    return root


class BrowserWorldStore:
    """Durable per-session WorldState; helper process memory is never truth."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = _world_root(self.workspace)
        self.sessions = self.root / "sessions"
        self.sessions.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, session_id: str) -> Path:
        return self.sessions / f"{_safe_id(session_id, 'browser_session_id')}.json"

    @staticmethod
    def _empty(session_id: str) -> Dict[str, Any]:
        now = _now()
        return {
            "schema": BROWSER_WORLD_SCHEMA,
            "browser_session_id": session_id,
            "revision": 0,
            "created_at": now,
            "updated_at": now,
            "pages": {},
            "history": [],
            "last_action": None,
            "helper": {"schema": BROWSER_HELPER_SCHEMA, "state": "UNKNOWN"},
        }

    def load(self, session_id: str) -> Dict[str, Any]:
        session_id = _safe_id(session_id, "browser_session_id")
        path = self._path(session_id)
        if not path.is_file():
            return self._empty(session_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty(session_id)
        if not isinstance(value, dict) or value.get("schema") != BROWSER_WORLD_SCHEMA:
            return self._empty(session_id)
        return value

    def save(self, document: Mapping[str, Any]) -> None:
        session_id = _safe_id(document.get("browser_session_id"), "browser_session_id")
        atomic_write_json(self._path(session_id), dict(document))

    def record(
        self,
        session_id: str,
        *,
        operation: str,
        payload: Mapping[str, Any],
        helper: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Commit a helper fact and return the current, revisioned projection."""
        session_id = _safe_id(session_id, "browser_session_id")
        with self._lock:
            document = self.load(session_id)
            revision = int(document.get("revision") or 0) + 1
            pages = document.get("pages") if isinstance(document.get("pages"), dict) else {}
            page = payload.get("page") if isinstance(payload.get("page"), Mapping) else payload
            page_id = str(page.get("page_id") or "").strip() if isinstance(page, Mapping) else ""
            changed_nodes: list[dict[str, Any]] = []
            if page_id and isinstance(page, Mapping):
                old_page = pages.get(page_id) if isinstance(pages.get(page_id), Mapping) else {}
                if "nodes" in page:
                    prior_nodes = {
                        str(item.get("node_id")): item
                        for item in (old_page.get("nodes") or [])
                        if isinstance(item, Mapping) and item.get("node_id")
                    }
                    current_nodes = {
                        str(item.get("node_id")): item
                        for item in (page.get("nodes") or [])
                        if isinstance(item, Mapping) and item.get("node_id")
                    }
                    for node_id in sorted(set(prior_nodes) | set(current_nodes)):
                        before = prior_nodes.get(node_id)
                        after = current_nodes.get(node_id)
                        if before == after:
                            continue
                        changed_nodes.append({
                            "node_id": node_id,
                            "kind": "added" if before is None else "removed" if after is None else "changed",
                        })
                stored_page = {**dict(old_page), **dict(page)}
                stored_page["observed_revision"] = revision
                pages[page_id] = stored_page
            document["pages"] = pages
            document["revision"] = revision
            document["updated_at"] = _now()
            document["helper"] = dict(helper)
            event = {
                "revision": revision,
                "operation": str(operation),
                "at": document["updated_at"],
                "page_id": page_id or None,
                "changed_nodes": changed_nodes[:MAX_EVENTS],
                "payload_summary": {
                    "url": page.get("url") if isinstance(page, Mapping) else None,
                    "title": page.get("title") if isinstance(page, Mapping) else None,
                    "verified": payload.get("verified"),
                },
            }
            history = list(document.get("history") or [])
            history.append(event)
            document["history"] = history[-MAX_HISTORY:]
            if operation not in {"observe", "pages", "health", "world.changed"}:
                document["last_action"] = event
            self.save(document)
            return {
                "schema": BROWSER_WORLD_SCHEMA,
                "browser_session_id": session_id,
                "world_revision": revision,
                "changed_nodes": changed_nodes[:MAX_EVENTS],
                "state": document,
            }

    def changed(self, session_id: str, since_revision: Any) -> Dict[str, Any]:
        session_id = _safe_id(session_id, "browser_session_id")
        try:
            since = max(0, int(since_revision))
        except (TypeError, ValueError):
            raise BrowserWorldError("INVALID_ARGUMENTS", "since_revision must be an integer")
        with self._lock:
            document = self.load(session_id)
            current = int(document.get("revision") or 0)
            history = [item for item in document.get("history") or [] if isinstance(item, Mapping)]
            oldest = int(history[0].get("revision") or current) if history else current
            gap = bool(history and since < oldest - 1)
            events = [dict(item) for item in history if int(item.get("revision") or 0) > since]
            return {
                "schema": BROWSER_WORLD_SCHEMA,
                "browser_session_id": session_id,
                "since_revision": since,
                "world_revision": current,
                "gap": gap,
                "snapshot_required": gap,
                "events": events[-MAX_EVENTS:],
            }


@contextmanager
def _sidecar_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class BrowserRuntime:
    """One local Unix-socket sidecar shared by Hawking processes for a root."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = _world_root(self.workspace)
        self.project_root = self._project_root()
        self.socket_path = self.root / "helper.sock"
        self.lock_path = self.root / "helper.lock"
        self.info_path = self.root / "helper.json"
        self.stderr_path = self.root / "helper.stderr.log"
        self._lock = threading.RLock()

    @property
    def helper_path(self) -> Path:
        return self.project_root / "workspace" / "ops" / "hawking_browser_helper.mjs"

    @property
    def ops_dir(self) -> Path:
        return self.project_root / "workspace" / "ops"

    def _project_root(self) -> Path:
        """Find the installed Hawking runtime, including from a linked worktree."""
        explicit = os.environ.get("HAWKING_BROWSER_OPS_ROOT")
        candidates = []
        if explicit:
            candidates.append(Path(explicit).expanduser())
        candidates.extend((self.workspace, *self.workspace.parents, Path(__file__).resolve().parents[1]))
        for candidate in candidates:
            root = candidate.resolve()
            if (root / "workspace" / "ops" / "hawking_browser_helper.mjs").is_file():
                return root
        # Keep the error deterministic and point at the primary workspace.
        return self.workspace

    def _request(self, payload: Mapping[str, Any], *, timeout_s: float = 30.0) -> Dict[str, Any]:
        if not hasattr(socket, "AF_UNIX"):
            raise BrowserWorldError("UNSUPPORTED_OS", "Hawking browser sidecar requires Unix sockets")
        encoded = (json.dumps(dict(payload), separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(max(0.1, min(120.0, float(timeout_s))))
                client.connect(str(self.socket_path))
                client.sendall(encoded)
                parts: list[bytes] = []
                while True:
                    block = client.recv(65536)
                    if not block:
                        break
                    parts.append(block)
                    if b"\n" in block:
                        break
        except (OSError, TimeoutError) as exc:
            raise BrowserWorldError("HELPER_UNAVAILABLE", str(exc)) from exc
        raw = b"".join(parts).split(b"\n", 1)[0]
        try:
            response = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BrowserWorldError("HELPER_PROTOCOL", "browser helper returned invalid JSON") from exc
        if not isinstance(response, dict) or response.get("schema") != BROWSER_HELPER_SCHEMA:
            raise BrowserWorldError("HELPER_PROTOCOL", "browser helper returned an unknown protocol")
        if response.get("ok") is not True:
            error = response.get("error") if isinstance(response.get("error"), Mapping) else {}
            raise BrowserWorldError(str(error.get("code") or "HELPER_ERROR"), str(error.get("message") or "browser helper failed"))
        result = response.get("result")
        if not isinstance(result, dict):
            raise BrowserWorldError("HELPER_PROTOCOL", "browser helper result was not an object")
        return result

    def _helper_ready(self) -> bool:
        try:
            self._request({"id": f"health-{uuid.uuid4().hex}", "op": "health"}, timeout_s=1.0)
            return True
        except BrowserWorldError:
            return False

    def _start(self) -> None:
        if not self.helper_path.is_file():
            raise BrowserWorldError("MISSING_DEPENDENCY", f"browser helper is missing: {self.helper_path}")
        node = os.environ.get("HAWKING_NODE") or shutil.which("node")
        if not node:
            raise BrowserWorldError("MISSING_DEPENDENCY", "Node.js is not installed")
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError as exc:
                raise BrowserWorldError("HELPER_UNAVAILABLE", f"stale helper socket cannot be removed: {exc}") from exc
        environment = os.environ.copy()
        environment["PLAYWRIGHT_BROWSERS_PATH"] = str(self.project_root / ".hawking" / "browser-runtime")
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with self.stderr_path.open("ab") as stderr:
            process = subprocess.Popen(
                [node, str(self.helper_path), "--serve", str(self.socket_path)],
                cwd=str(self.ops_dir),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                start_new_session=True,
            )
        atomic_write_json(self.info_path, {
            "schema": BROWSER_HELPER_SCHEMA,
            "pid": process.pid,
            "started_at": _now(),
            "socket": str(self.socket_path),
            "helper": str(self.helper_path),
            "node": node,
        })
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._helper_ready():
                return
            if process.poll() is not None:
                break
            time.sleep(0.05)
        detail = "browser sidecar did not become ready"
        try:
            tail = self.stderr_path.read_text(encoding="utf-8", errors="replace")[-1200:]
            if tail:
                detail = f"{detail}: {tail}"
        except OSError:
            pass
        raise BrowserWorldError("MISSING_DEPENDENCY", detail)

    def call(self, operation: str, arguments: Mapping[str, Any], *, timeout_s: float = 30.0) -> Dict[str, Any]:
        payload = {"id": f"browser-{uuid.uuid4().hex}", "op": str(operation), **dict(arguments)}
        with self._lock:
            if not self._helper_ready():
                with _sidecar_lock(self.lock_path):
                    if not self._helper_ready():
                        self._start()
            return self._request(payload, timeout_s=timeout_s)

    def health(self, *, probe: bool = False) -> Dict[str, Any]:
        try:
            result = self.call("health", {"probe": bool(probe)}, timeout_s=5.0)
        except BrowserWorldError as exc:
            return {
                "schema": BROWSER_HELPER_SCHEMA,
                "health": "MISSING_DEPENDENCY" if exc.code == "MISSING_DEPENDENCY" else "BROKEN",
                "failure_class": exc.code,
                "message": str(exc),
            }
        return {"schema": BROWSER_HELPER_SCHEMA, **result}

    def shutdown(self) -> bool:
        try:
            self._request({"id": f"shutdown-{uuid.uuid4().hex}", "op": "shutdown"}, timeout_s=5.0)
            return True
        except BrowserWorldError:
            return False


def browser_runtime(workspace: str | os.PathLike[str]) -> BrowserRuntime:
    root = str(Path(workspace).expanduser().resolve())
    with _RUNTIMES_LOCK:
        current = _RUNTIMES.get(root)
        if current is None:
            current = BrowserRuntime(root)
            _RUNTIMES[root] = current
        return current


class BrowserWorld:
    """Typed Hawking operations over the shared browser sidecar."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.store = BrowserWorldStore(self.workspace)
        self.runtime = browser_runtime(self.workspace)

    @staticmethod
    def _session_id(arguments: Mapping[str, Any], *, optional: bool = False) -> str:
        value = arguments.get("browser_session_id")
        if value is None and optional:
            return f"BROWSER-{uuid.uuid4().hex[:16].upper()}"
        return _safe_id(value, "browser_session_id")

    def _commit(self, session_id: str, operation: str, result: Mapping[str, Any]) -> Dict[str, Any]:
        helper = self.runtime.health()
        projection = self.store.record(session_id, operation=operation, payload=result, helper=helper)
        return {
            **dict(result),
            "browser_session_id": session_id,
            "world_revision": projection["world_revision"],
            "changed_nodes": projection["changed_nodes"],
        }

    def health(self) -> Dict[str, Any]:
        # The public tool is an active health probe; WorldState commits below
        # deliberately use the passive Runtime.health() default instead.
        return self.runtime.health(probe=True)

    def open(self, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        session_id = self._session_id(arguments, optional=True)
        result = self.runtime.call("session.open", {**dict(arguments), "browser_session_id": session_id}, timeout_s=float(arguments.get("timeout_s") or 30.0))
        return self._commit(session_id, "open", result)

    def close(self, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        session_id = self._session_id(arguments)
        result = self.runtime.call("session.close", {"browser_session_id": session_id}, timeout_s=15.0)
        return self._commit(session_id, "close", result)

    def changed(self, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        return self.store.changed(self._session_id(arguments), arguments.get("since_revision"))

    def invoke(self, operation: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        if operation == "health":
            return self.health()
        if operation == "open":
            return self.open(arguments)
        if operation == "close":
            return self.close(arguments)
        if operation == "world.changed":
            return self.changed(arguments)
        session_id = self._session_id(arguments)
        helper_operation = operation
        supplied = dict(arguments)
        if operation == "screenshot":
            page_id = str(supplied.get("page_id") or "default")
            safe_page = re.sub(r"[^A-Za-z0-9_-]", "_", page_id)[:120] or "default"
            output = self.store.root / "artifacts" / session_id / safe_page / f"{int(time.time() * 1000)}.png"
            supplied["output_path"] = str(output)
        result = self.runtime.call(helper_operation, supplied, timeout_s=float(supplied.get("timeout_s") or 30.0))
        if operation == "screenshot":
            artifact = result.get("artifact") if isinstance(result.get("artifact"), Mapping) else {}
            candidate = Path(str(artifact.get("path") or ""))
            if candidate.is_file():
                result = dict(result)
                result["artifact"] = {
                    **dict(artifact),
                    "sha256": _sha256_path(candidate),
                    "retention_class": "DERIVED",
                }
        page = result.get("page") if isinstance(result.get("page"), Mapping) else result
        if isinstance(page, Mapping) and page.get("page_id"):
            result = {**dict(result), "page": dict(page)}
        return self._commit(session_id, operation, result)


def browser_tool(workspace: str | os.PathLike[str], operation: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Registry adapter; the ToolRegistry remains the permission authority."""
    return BrowserWorld(workspace).invoke(operation, arguments)


__all__ = [
    "BROWSER_HELPER_SCHEMA",
    "BROWSER_WORLD_SCHEMA",
    "BrowserRuntime",
    "BrowserWorld",
    "BrowserWorldError",
    "BrowserWorldStore",
    "browser_runtime",
    "browser_tool",
]
