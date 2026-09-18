"""Canonical machine-capability and macOS permission broker.

This is the machine side of Hawking's existing capability registry.  It owns
observations and Goal admission, not user-facing Goal state: the operating
system is the authority for TCC, while the workspace and Goal contracts are
the narrower logical ceilings.  The JSON file written here is a diagnostic
cache of the last observation, never a claim that a permission is granted.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .native_owner import (
    STABLE_OWNER_IDENTIFIER,
    inspect_code_identity,
    resolve_native_helper,
)
from .persist import atomic_write_json


PERMISSIONS_SCHEMA = "hawking.permissions.v1"
PERMISSION_CACHE_SCHEMA = "hawking.permission.observation.v1"
PERMISSION_BLOCKED = "PERMISSION_BLOCKED"

TCC_CAPABILITIES = (
    "FULL_DISK_ACCESS",
    "ACCESSIBILITY",
    "SCREEN_CAPTURE",
    "INPUT_MONITORING",
    "AUTOMATION",
    "MICROPHONE",
    "SPEECH_RECOGNITION",
)
FULL_DISK_ACCESS = "FULL_DISK_ACCESS"
ACCESSIBILITY = "ACCESSIBILITY"
SCREEN_CAPTURE = "SCREEN_CAPTURE"
INPUT_MONITORING = "INPUT_MONITORING"
AUTOMATION = "AUTOMATION"
MICROPHONE = "MICROPHONE"
SPEECH_RECOGNITION = "SPEECH_RECOGNITION"
NON_TCC_CAPABILITIES = (
    "NETWORK",
    "WORKSPACE_READ",
    "WORKSPACE_WRITE",
    "EXTERNAL_DRIVE_READ",
    "EXTERNAL_DRIVE_WRITE",
    "PROCESS_EXECUTION",
    "DEVELOPER_TOOLS",
    "BROWSER",
)
NETWORK = "NETWORK"
WORKSPACE_READ = "WORKSPACE_READ"
WORKSPACE_WRITE = "WORKSPACE_WRITE"
EXTERNAL_DRIVE_READ = "EXTERNAL_DRIVE_READ"
EXTERNAL_DRIVE_WRITE = "EXTERNAL_DRIVE_WRITE"
PROCESS_EXECUTION = "PROCESS_EXECUTION"
DEVELOPER_TOOLS = "DEVELOPER_TOOLS"
BROWSER = "BROWSER"
MACHINE_CAPABILITIES = TCC_CAPABILITIES + NON_TCC_CAPABILITIES

READY = "READY"
MISSING = "MISSING"
NOT_REQUESTED = "NOT_REQUESTED"
NOT_REQUIRED = "NOT_REQUIRED"
NOT_APPLICABLE = "NOT_APPLICABLE"
NOT_PRESENT = "NOT_PRESENT"
UNKNOWN = "UNKNOWN"
UNAVAILABLE = "UNAVAILABLE"
PARTIAL = "PARTIAL"

_SETTINGS_PANES = {
    "FULL_DISK_ACCESS": "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles",
    "ACCESSIBILITY": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "SCREEN_CAPTURE": "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
    "INPUT_MONITORING": "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
    "AUTOMATION": "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation",
    "MICROPHONE": "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    "SPEECH_RECOGNITION": "x-apple.systempreferences:com.apple.preference.security?Privacy_SpeechRecognition",
}


def normalize_capability(value: Any) -> str:
    """Normalize a Goal capability without erasing target-specific suffixes."""
    text = str(value or "").strip()
    if not text:
        return ""
    if ":" in text:
        head, tail = text.split(":", 1)
        return f"{head.strip().upper()}:{tail.strip()}"
    return text.upper().replace("-", "_").replace(" ", "_")


def normalize_capabilities(values: Iterable[Any] | None) -> list[str]:
    result: list[str] = []
    for value in values or ():
        name = normalize_capability(value)
        if name and name not in result:
            result.append(name)
    return result


def _cache_path(workspace: Path) -> Path:
    return workspace / ".hawking" / "permissions" / "state.json"


def _safe_path(value: Any) -> Optional[Path]:
    if value in (None, ""):
        return None
    try:
        return Path(str(value)).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _path_readable(path: Path) -> tuple[bool, str]:
    try:
        if not path.exists():
            return False, "path is not present"
        if path.is_dir():
            next(os.scandir(path), None)
        else:
            with path.open("rb") as handle:
                handle.read(1)
        return True, "read probe succeeded"
    except (OSError, PermissionError) as exc:
        return False, f"read probe failed: {type(exc).__name__}"


def _path_writable(path: Path) -> tuple[bool, str]:
    try:
        if not path.exists() or not path.is_dir():
            return False, "directory is not present"
        return bool(os.access(path, os.W_OK)), (
            "directory write access reported by the OS"
            if os.access(path, os.W_OK) else "directory is not writable"
        )
    except OSError as exc:
        return False, f"write probe failed: {type(exc).__name__}"


def _workspace_capability(workspace: Path, name: str) -> Dict[str, Any]:
    if name == "WORKSPACE_READ":
        ready, reason = _path_readable(workspace)
    else:
        ready, reason = _path_writable(workspace)
    return {
        "state": READY if ready else MISSING,
        "kind": "logical_workspace",
        "owner": "hawking.permissions.WorkspaceBoundary",
        "evidence": {"path": str(workspace), "reason": reason},
    }


def _external_roots(explicit: Sequence[Any] | None) -> list[Path]:
    roots: list[Path] = []
    for item in explicit or ():
        path = _safe_path(item)
        if path and path not in roots:
            roots.append(path)
    if roots:
        return roots
    volumes = Path("/Volumes")
    if volumes.is_dir():
        try:
            for child in sorted(volumes.iterdir()):
                if child.name != "Macintosh HD" and child.is_dir():
                    roots.append(child.resolve())
        except OSError:
            pass
    return roots[:32]


def _external_capability(name: str, roots: list[Path], *, explicitly_required: bool) -> Dict[str, Any]:
    if not roots:
        return {
            "state": MISSING if explicitly_required else NOT_PRESENT,
            "kind": "logical_external_drive",
            "owner": "hawking.permissions.ExternalDriveBoundary",
            "evidence": {"paths": [], "reason": "no external volume is mounted"},
        }
    checks = []
    for root in roots:
        if name.endswith("READ"):
            ok, reason = _path_readable(root)
        else:
            ok, reason = _path_writable(root)
        checks.append({"path": str(root), "ready": ok, "reason": reason})
    ready = any(item["ready"] for item in checks)
    return {
        "state": READY if ready else MISSING,
        "kind": "logical_external_drive",
        "owner": "hawking.permissions.ExternalDriveBoundary",
        "evidence": {"paths": checks, "reason": "at least one mounted volume passed" if ready else "no mounted volume passed"},
    }


def _non_tcc_capability(workspace: Path, name: str, external_roots: list[Path], *, explicitly_required: bool) -> Dict[str, Any]:
    if name in {"WORKSPACE_READ", "WORKSPACE_WRITE"}:
        return _workspace_capability(workspace, name)
    if name in {"EXTERNAL_DRIVE_READ", "EXTERNAL_DRIVE_WRITE"}:
        return _external_capability(name, external_roots, explicitly_required=explicitly_required)
    if name == "NETWORK":
        # Network has no TCC prompt.  A loopback socket proves that the process
        # has a usable network stack without contacting an external service.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
            return {"state": READY, "kind": "logical_non_tcc", "owner": "hawking.network", "evidence": {"probe": "loopback bind"}}
        except OSError as exc:
            return {"state": MISSING, "kind": "logical_non_tcc", "owner": "hawking.network", "evidence": {"reason": f"loopback probe failed: {type(exc).__name__}"}}
    if name == "PROCESS_EXECUTION":
        executable = Path(sys.executable)
        ready = executable.is_file() and os.access(executable, os.X_OK)
        return {"state": READY if ready else MISSING, "kind": "logical_non_tcc", "owner": "hawking.processes", "evidence": {"executable": str(executable)}}
    if name == "DEVELOPER_TOOLS":
        tools = ["/usr/bin/xcrun", "/usr/bin/swiftc", "/usr/bin/codesign"]
        present = [path for path in tools if Path(path).is_file()]
        ready = sys.platform != "darwin" or len(present) >= 2
        return {"state": READY if ready else MISSING, "kind": "logical_non_tcc", "owner": "hawking.native_owner", "evidence": {"tools": present}}
    if name == "BROWSER":
        try:
            from .browser_world import browser_tool  # noqa: F401
            ready = True
        except Exception:
            ready = False
        return {"state": READY if ready else UNAVAILABLE, "kind": "logical_non_tcc", "owner": "hawking.browser_world", "evidence": {"module": ready}}
    return {"state": UNKNOWN, "kind": "logical_non_tcc", "owner": "hawking.permissions", "evidence": {"reason": "no probe is implemented"}}


def _default_observation(name: str, *, required: bool, reason: str) -> Dict[str, Any]:
    return {
        "state": MISSING if required else NOT_REQUESTED,
        "kind": "tcc",
        "owner": STABLE_OWNER_IDENTIFIER,
        "evidence": {"reason": reason},
    }


class PermissionRequired(RuntimeError):
    """Fresh OS evidence says a required capability cannot be admitted."""

    code = PERMISSION_BLOCKED

    def __init__(self, missing: Sequence[Mapping[str, Any]], *, required: Sequence[str], snapshot: Mapping[str, Any]):
        self.missing = [dict(item) for item in missing]
        self.required = list(required)
        self.snapshot = dict(snapshot)
        names = ", ".join(str(item.get("capability") or "") for item in self.missing)
        super().__init__(f"{PERMISSION_BLOCKED}: required capability unavailable: {names}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "failure_class": self.code,
            "required": list(self.required),
            "missing": list(self.missing),
            "next_action": "grant the named capability in System Settings, then recheck",
            "snapshot_checked_at": self.snapshot.get("checked_at"),
        }


class PermissionsService:
    """One canonical status/require/explain service for machine readiness."""

    def __init__(self, workspace: str | os.PathLike[str] | None = None, *, runtime: Any = None) -> None:
        self.workspace = Path(workspace or os.getcwd()).expanduser().resolve()
        self.runtime = runtime

    @property
    def cache_path(self) -> Path:
        return _cache_path(self.workspace)

    def _load_cache(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        return value if isinstance(value, dict) and value.get("schema") == PERMISSION_CACHE_SCHEMA else {}

    def _native_status(self, requested: Sequence[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Return (native result, identity), preserving unavailable as evidence."""
        identity: Dict[str, Any] = {
            "schema": "hawking.native.code_identity.v1",
            "stable_identifier": STABLE_OWNER_IDENTIFIER,
            "state": MISSING,
            "stable_owner": False,
        }
        runtime = self.runtime
        if runtime is None:
            try:
                _helper_path, identity = resolve_native_helper(self.workspace)
            except Exception as exc:
                return {
                    "available": False,
                    "state": UNAVAILABLE,
                    "reason": str(exc)[:400],
                    "capabilities": {},
                }, identity
            try:
                from .macos_world import MacOSRuntime
                runtime = MacOSRuntime(self.workspace)
            except Exception as exc:
                return {"available": False, "state": UNAVAILABLE, "reason": str(exc)[:400], "capabilities": {}}, identity
        else:
            # Runtime injection is an explicit seam for deterministic tests and
            # an eventual in-process native bridge.  Production construction
            # still takes the stable-owner path above; an injected runtime is
            # never presented as a signed permission owner.
            identity = {
                **identity,
                "state": "EXPLICIT_RUNTIME",
                "owner_mode": "injected_runtime",
            }
        try:
            result = runtime.call("permissions.status", {"capabilities": list(requested)})
            return {"available": True, **dict(result)}, identity
        except Exception as exc:
            return {"available": False, "state": UNAVAILABLE, "reason": f"native helper permission probe failed: {type(exc).__name__}", "capabilities": {}}, identity

    def _observe(self, required: Sequence[str], *, external_roots: Sequence[Any] | None = None) -> Dict[str, Any]:
        requested = normalize_capabilities(required)
        all_names = list(dict.fromkeys([*MACHINE_CAPABILITIES, *requested]))
        explicit_external = [item for item in external_roots or () if str(item or "").strip()]
        roots = _external_roots(explicit_external)
        native, identity = self._native_status([name for name in all_names if name.split(":", 1)[0] in TCC_CAPABILITIES])
        native_caps = native.get("capabilities") if isinstance(native.get("capabilities"), Mapping) else {}
        capabilities: Dict[str, Any] = {}
        for name in all_names:
            base_name = name.split(":", 1)[0]
            if base_name in TCC_CAPABILITIES:
                if base_name == "FULL_DISK_ACCESS":
                    # macOS has no public FDA preflight boolean.  The native
                    # helper reports UNKNOWN; a real protected-path probe is
                    # the stronger evidence accepted by Hawking.
                    capabilities[name] = self._full_disk_access_probe(required=name in requested)
                    continue
                row = native_caps.get(name) or native_caps.get(base_name)
                if isinstance(row, Mapping):
                    row = dict(row)
                    row.setdefault("owner", STABLE_OWNER_IDENTIFIER)
                    capabilities[name] = row
                elif native.get("available") is True:
                    capabilities[name] = _default_observation(name, required=name in requested, reason="native helper returned no observation")
                else:
                    capabilities[name] = _default_observation(name, required=name in requested, reason=str(native.get("reason") or "stable native helper unavailable"))
            else:
                capabilities[name] = _non_tcc_capability(self.workspace, base_name, roots, explicitly_required=name in requested)
        checked_at = time.time()
        cache = {
            "schema": PERMISSION_CACHE_SCHEMA,
            "owner": "hawking.capabilities",
            "last_observed_state": capabilities,
            "code_identity": identity,
            "os_version": platform.platform(),
            "helper_revision": native.get("helper_version") or native.get("helper_revision"),
            "checked_at": checked_at,
            "workspace": str(self.workspace),
            "external_roots": [str(path) for path in roots],
            "native": {key: value for key, value in native.items() if key != "capabilities"},
        }
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.cache_path, cache)
        except OSError:
            # A read-only workspace must not turn an accurate OS observation
            # into a false permission block.  The response remains current.
            pass
        return {
            "schema": PERMISSIONS_SCHEMA,
            "checked_at": checked_at,
            "workspace": str(self.workspace),
            "code_identity": identity,
            "helper_revision": cache["helper_revision"],
            "capabilities": capabilities,
            "required": list(requested),
            "native": cache["native"],
            "cache_path": str(self.cache_path),
        }

    def _full_disk_access_probe(self, *, required: bool) -> Dict[str, Any]:
        raw = os.environ.get("HAWKING_FDA_PROBE_PATHS", "")
        paths = [_safe_path(item) for item in raw.split(os.pathsep) if item.strip()]
        paths = [path for path in paths if path is not None]
        if not paths:
            paths = [
                Path.home() / "Library" / "Mail",
                Path.home() / "Library" / "Messages",
                Path.home() / "Library" / "Safari",
            ]
        existing = [path for path in paths if path.exists()]
        if not existing:
            return {
                "state": UNKNOWN if required else NOT_REQUESTED,
                "kind": "tcc",
                "owner": STABLE_OWNER_IDENTIFIER,
                "evidence": {"probe": "protected user-library paths", "paths": [], "reason": "no configured FDA probe path exists; no claim made"},
            }
        checks = [{"path": str(path), "ready": _path_readable(path)[0], "reason": _path_readable(path)[1]} for path in existing]
        ready = any(item["ready"] for item in checks)
        return {
            "state": READY if ready else MISSING,
            "kind": "tcc",
            "owner": STABLE_OWNER_IDENTIFIER,
            "evidence": {"probe": "protected user-library paths", "paths": checks, "reason": "protected path read succeeded" if ready else "protected paths were denied"},
        }

    def status(self, required: Iterable[Any] | None = None, *, refresh: bool = True, external_roots: Sequence[Any] | None = None) -> Dict[str, Any]:
        requested = normalize_capabilities(required)
        if not refresh:
            cache = self._load_cache()
            if cache:
                return {
                    "schema": PERMISSIONS_SCHEMA,
                    "checked_at": cache.get("checked_at"),
                    "workspace": str(self.workspace),
                    "code_identity": cache.get("code_identity") or {},
                    "helper_revision": cache.get("helper_revision"),
                    "capabilities": cache.get("last_observed_state") or {},
                    "required": requested,
                    "native": cache.get("native") or {},
                    "cache_path": str(self.cache_path),
                    "fresh": False,
                }
        return {**self._observe(requested, external_roots=external_roots), "fresh": True}

    def require(self, required: Iterable[Any], *, external_roots: Sequence[Any] | None = None) -> Dict[str, Any]:
        names = normalize_capabilities(required)
        snapshot = self.status(names, refresh=True, external_roots=external_roots)
        missing: list[Dict[str, Any]] = []
        capabilities = snapshot.get("capabilities") if isinstance(snapshot.get("capabilities"), Mapping) else {}
        for name in names:
            row = capabilities.get(name) if isinstance(capabilities, Mapping) else None
            state = str(row.get("state") or UNKNOWN) if isinstance(row, Mapping) else UNKNOWN
            if state != READY:
                missing.append({
                    "capability": name,
                    "state": state,
                    "explanation": self.explain(name),
                })
        if missing:
            raise PermissionRequired(missing, required=names, snapshot=snapshot)
        return {"schema": PERMISSIONS_SCHEMA, "ready": True, "required": names, "snapshot": snapshot}

    def explain(self, capability: Any) -> Dict[str, Any]:
        name = normalize_capability(capability)
        base = name.split(":", 1)[0]
        target = name.split(":", 1)[1] if ":" in name else None
        messages = {
            "FULL_DISK_ACCESS": "Hawking needs Full Disk Access to read the requested protected local paths.",
            "ACCESSIBILITY": "Hawking needs Accessibility to observe and operate authorized desktop applications through its native helper.",
            "SCREEN_CAPTURE": "Hawking needs Screen & System Audio Recording permission for visual observation.",
            "INPUT_MONITORING": "Hawking needs Input Monitoring only when the Goal requires global input observation; Accessibility is not a substitute.",
            "AUTOMATION": f"Hawking needs Apple Events automation permission for {target or 'the named target application'}.",
            "MICROPHONE": "Hawking needs Microphone access for an authorized voice Goal.",
            "SPEECH_RECOGNITION": "Hawking needs Speech Recognition access when Apple speech APIs are used.",
            "WORKSPACE_READ": "The Goal must be authorized to read its logical workspace.",
            "WORKSPACE_WRITE": "The Goal must be authorized to mutate its logical workspace.",
            "EXTERNAL_DRIVE_READ": "The Goal must name a mounted external volume it may read.",
            "EXTERNAL_DRIVE_WRITE": "The Goal must name a mounted external volume it may write.",
            "NETWORK": "Hawking needs a usable network stack for this Goal.",
            "PROCESS_EXECUTION": "Hawking needs process execution for the requested build or test operation.",
            "DEVELOPER_TOOLS": "Hawking needs the installed developer tools required by this Goal.",
            "BROWSER": "Hawking needs its managed browser capability for this Goal.",
        }
        return {
            "capability": name,
            "kind": "tcc" if base in TCC_CAPABILITIES else "logical",
            "message": messages.get(base, f"Hawking requires {name} for this Goal."),
            "settings_url": _SETTINGS_PANES.get(base),
            "operator_action": "Open System Settings and grant the named capability, then press Recheck." if base in TCC_CAPABILITIES else "Confirm the logical Goal/workspace authority and press Recheck.",
            "target": target,
        }

    def machine_ready(self, *, broad: bool = False, external_roots: Sequence[Any] | None = None) -> Dict[str, Any]:
        required = ["WORKSPACE_READ", "WORKSPACE_WRITE", "NETWORK", "PROCESS_EXECUTION"]
        if broad:
            required.extend(TCC_CAPABILITIES)
            required.extend(("DEVELOPER_TOOLS", "BROWSER"))
            if external_roots:
                required.extend(("EXTERNAL_DRIVE_READ", "EXTERNAL_DRIVE_WRITE"))
        snapshot = self.status(required, refresh=True, external_roots=external_roots)
        capabilities = snapshot.get("capabilities") or {}
        # Return the complete fresh observation for the operator surface, but
        # calculate ``machine_ready`` only from the scoped required set. This
        # lets H-Web show specialist capabilities as NOT_REQUESTED/MISSING
        # without making a source-only Goal wait for them.
        rows = {
            str(name): {**dict(row), "capability": str(name)}
            for name, row in capabilities.items()
            if isinstance(row, Mapping)
        }
        for name in required:
            rows.setdefault(name, {"capability": name, "state": UNKNOWN})
        ready = all(rows[name].get("state") == READY for name in required)
        return {
            "schema": "hawking.machine.authority.v1",
            "machine_ready": ready,
            "broad": broad,
            "checked_at": snapshot.get("checked_at"),
            "required": required,
            "code_identity": snapshot.get("code_identity"),
            "capabilities": rows,
            "snapshot": snapshot,
        }


def permissions_status(workspace: str | os.PathLike[str] | None = None, **kwargs: Any) -> Dict[str, Any]:
    return PermissionsService(workspace).status(**kwargs)


def permissions_require(workspace: str | os.PathLike[str], required: Iterable[Any], **kwargs: Any) -> Dict[str, Any]:
    return PermissionsService(workspace).require(required, **kwargs)


def permissions_explain(capability: Any) -> Dict[str, Any]:
    return PermissionsService().explain(capability)


__all__ = [
    "ACCESSIBILITY", "AUTOMATION", "BROWSER", "DEVELOPER_TOOLS",
    "EXTERNAL_DRIVE_READ", "EXTERNAL_DRIVE_WRITE", "FULL_DISK_ACCESS",
    "INPUT_MONITORING", "MACHINE_CAPABILITIES", "MICROPHONE", "MISSING",
    "NETWORK", "NON_TCC_CAPABILITIES",
    "NOT_APPLICABLE", "NOT_PRESENT", "NOT_REQUESTED", "NOT_REQUIRED", "PARTIAL",
    "PERMISSION_BLOCKED", "PERMISSIONS_SCHEMA", "PROCESS_EXECUTION",
    "PermissionRequired", "PermissionsService", "READY", "SCREEN_CAPTURE",
    "SPEECH_RECOGNITION", "TCC_CAPABILITIES", "UNKNOWN", "UNAVAILABLE",
    "WORKSPACE_READ", "WORKSPACE_WRITE", "normalize_capabilities", "normalize_capability",
    "permissions_explain", "permissions_require", "permissions_status",
]
