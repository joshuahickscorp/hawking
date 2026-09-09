"""The nine tools hawking actually allowlists, implemented natively.

hcli/vmcp_adapter.py has always refused everything outside VMCP_READ_ONLY_TOOLS,
so the ~294 other tools the foreign package ships were never reachable and are
not owed. These nine are.

Two departures from the foreign implementation, both deliberate:

  observe   is built on file_eye's magic-byte classifier, which covers 12
            formats with NO Pillow and NO OpenCV. The foreign path imports
            numpy and PIL at module scope and calls cv2 unconditionally --
            despite its own manifest claiming no core path reaches them. That
            would have added two dependencies, one a native binary wheel, for
            capability hawking already has.

  verify    binds evidence to a SUBJECT PATH, not only to a digest. Comparing
            digests alone accepts a replay of identical bytes from a different
            file. That check came out of the retired hcli_integration prototype
            and is the one idea in it worth keeping.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

from .host import Host
from .store import ProjectStore, sha256_file, utc_now

ADAPTERS = {"file.local"}
ADAPTER_VERSION = "1"
TOOL_NAMES = (
    "system.doctor", "project.status", "vision.capabilities", "vision.observe",
    "vision.query", "vision.verify", "vision.progress", "vision.list_artifacts",
    "vision.get_artifact",
)


def _classify(path: Path) -> dict[str, Any]:
    """Magic-byte classification with no image stack. Falls back honestly."""
    try:
        from hcli.agentos.vmcp.file_eye import classify_bytes
    except Exception as exc:  # pragma: no cover - environment, not logic
        return {"kind": "unclassified", "why": f"file_eye unavailable: {exc}"}
    return classify_bytes(path.read_bytes()[:8_000_000])


def _open(projects_root: Path, project_path: str) -> ProjectStore:
    """Resolve a project path, refusing anything that escapes the root."""
    root = Path(projects_root).expanduser().resolve()
    p = Path(project_path).expanduser()
    p = (root / p).resolve() if not p.is_absolute() else p.resolve()
    if root not in p.parents and p != root:
        raise ValueError(f"project path escapes projects root: {p}")
    return ProjectStore.open(p) if (p / "project.db").is_file() else ProjectStore.create(p, p.name)


def register(host: Host) -> Host:
    root = Path(host.projects_root)

    @host.tool("system.doctor")
    def system_doctor() -> dict[str, Any]:
        """Offline health report. No network, no optional dependency."""
        return {
            "ok": True, "offline": True, "network_forbidden": True,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "projects_root": str(root),
            "projects_root_writable": os.access(root.parent if not root.exists() else root, os.W_OK),
            "adapters": sorted(ADAPTERS),
            "tools": list(TOOL_NAMES),
            "implementation": "hawking-native (hcli.perception)",
            "checked_at": utc_now(),
        }

    @host.tool("vision.capabilities")
    def vision_capabilities() -> dict[str, Any]:
        """What this host can do, and what it deliberately does not."""
        return {
            "profile": host.profile,
            "tools": list(TOOL_NAMES),
            "adapters": sorted(ADAPTERS),
            "implementation": "hawking-native (hcli.perception)",
            "not_implemented": {
                "vision.verify(receipt_path)":
                    "the acceptance-receipt audit trail is not ported; verify binds a "
                    "capture_id to its subject instead",
                "browser/3d/worldir adapters":
                    "never in the allowlist, never reachable through vmcp_adapter",
            },
            "dependencies": {"pillow": False, "opencv": False, "numpy": False},
        }

    @host.tool("project.status")
    def project_status(project_path: str) -> dict[str, Any]:
        """Project metadata plus table counts and directory health."""
        return _open(root, project_path).status()

    @host.tool("vision.list_artifacts")
    def vision_list_artifacts(project_path: str, limit: int = 100) -> dict[str, Any]:
        """List content-addressed artifacts recorded in the project."""
        rows = _open(root, project_path).list_artifacts(limit)
        return {"artifacts": rows, "count": len(rows), "limit": max(1, min(int(limit), 10_000))}

    @host.tool("vision.get_artifact")
    def vision_get_artifact(project_path: str, digest: str,
                            include_path: bool = True) -> dict[str, Any]:
        """Resolve a content-addressed artifact by SHA-256 digest."""
        store = _open(root, digest and project_path)
        path = store.artifact_path(digest)
        if not path.is_file():
            raise FileNotFoundError(f"artifact not found for digest {digest}")
        out: dict[str, Any] = {"digest": digest, "exists": True,
                               "size": path.stat().st_size, "record": store.artifact(digest)}
        if include_path:
            out["path"] = str(path)
        return out

    @host.tool("vision.observe")
    def vision_observe(project_path: str, rights_decision: str,
                       adapter: str = "file.local",
                       target: dict[str, Any] | None = None,
                       configuration: dict[str, Any] | None = None,
                       source_id: str | None = None) -> dict[str, Any]:
        """Capture a local target into durable OBSERVED evidence."""
        if not str(rights_decision or "").strip():
            raise ValueError("rights_decision is required")
        if adapter not in ADAPTERS:
            raise ValueError(f"adapter {adapter!r} is not available; have {sorted(ADAPTERS)}")
        if not target or not target.get("path"):
            raise ValueError(f"{adapter} requires a typed target with a path")
        subject = Path(str(target["path"])).expanduser().resolve()
        if not subject.is_file():
            raise FileNotFoundError(f"observe target does not exist: {subject}")
        store = _open(root, project_path)
        kind = _classify(subject)
        rec = store.ingest_file(subject, media_type=kind.get("media_type") or "application/octet-stream")
        request = {"adapter": adapter, "target": {"path": str(subject)},
                   "configuration": dict(configuration or {}),
                   "rights_decision": rights_decision.strip(), "source_id": source_id}
        cid = store.record_observation(
            adapter=adapter, adapter_version=ADAPTER_VERSION, status="OBSERVED",
            authority="local", manifest_digest=rec["digest"],
            subject_path=str(subject), request=request)
        return {"capture_id": cid, "status": "OBSERVED", "adapter": adapter,
                "adapter_version": ADAPTER_VERSION, "manifest_digest": rec["digest"],
                "artifact": rec, "classification": kind, "request": request}

    @host.tool("vision.query")
    def vision_query(project_path: str, limit: int = 100) -> dict[str, Any]:
        """Observations recorded in this project, most recent first."""
        rows = _open(root, project_path).observations(limit)
        return {"observations": rows, "count": len(rows)}

    @host.tool("vision.progress")
    def vision_progress(project_path: str) -> dict[str, Any]:
        """Overview: how much evidence exists and of what kind."""
        store = _open(root, project_path)
        obs = store.observations(10_000)
        by_status: dict[str, int] = {}
        by_adapter: dict[str, int] = {}
        for o in obs:
            by_status[o["status"]] = by_status.get(o["status"], 0) + 1
            by_adapter[o["adapter"]] = by_adapter.get(o["adapter"], 0) + 1
        return {"project": store.project(), "observations": len(obs),
                "artifacts": len(store.list_artifacts(10_000)),
                "by_status": by_status, "by_adapter": by_adapter}

    @host.tool("vision.verify")
    def vision_verify(project_path: str, capture_id: str) -> dict[str, Any]:
        """Re-check a capture against its subject. Binds to the PATH, not just bytes."""
        store = _open(root, project_path)
        obs = store.observation(capture_id)
        if obs is None:
            raise LookupError(f"no capture {capture_id}")
        failures: list[str] = []
        subject = Path(obs["subject_path"]) if obs.get("subject_path") else None
        stored = store.artifact_path(obs["manifest_digest"]) if obs.get("manifest_digest") else None

        if stored is None or not stored.is_file():
            failures.append("artifact_missing")
            live_digest = None
        else:
            live_digest = sha256_file(stored)
            if live_digest != obs["manifest_digest"]:
                failures.append("artifact_tampered")

        subject_present = bool(subject and subject.is_file())
        subject_digest = sha256_file(subject) if subject_present else None
        if not subject_present:
            failures.append("subject_absent")
        elif subject_digest != obs["manifest_digest"]:
            # Identical bytes at a DIFFERENT path would still pass a digest-only
            # check. Binding to the recorded path is what makes replay visible.
            failures.append("subject_changed")

        return {"capture_id": capture_id, "ok": not failures, "failures": failures,
                "manifest_digest": obs.get("manifest_digest"),
                "artifact_digest": live_digest, "subject_path": obs.get("subject_path"),
                "subject_digest": subject_digest, "status": obs.get("status"),
                "verified_at": utc_now()}

    return host


def create_server(projects_root, profile: str = "core") -> Host:
    """The one entry point hcli/vmcp_adapter.py needs."""
    return register(Host(Path(projects_root), profile))


def list_profiles(include_internal: bool = False) -> list[dict[str, Any]]:
    return [{"name": "core", "tools": list(TOOL_NAMES),
             "description": "hawking-native perception, local files only"}]


def public_api_versions() -> dict[str, Any]:
    return {"perception": "1", "implementation": "hcli.perception"}
