"""Small, local-only blue/green asset release owner for H-Web."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict

from .persist import atomic_write_bytes, atomic_write_json


class WebReleaseStore:
    """Tracks current/candidate/previous static assets without a second server."""

    def __init__(self, workspace: str | Path, source: str | Path) -> None:
        self.root = Path(workspace).resolve() / ".hawking" / "web-releases"
        self.source = Path(source).resolve()
        self.manifest_path = self.root / "manifest.json"

    def _read(self) -> Dict[str, Any]:
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            value = {}
        return dict(value) if isinstance(value, dict) else {}

    def _asset(self, payload: bytes, role: str) -> Dict[str, Any]:
        digest = hashlib.sha256(payload).hexdigest()
        release_id = f"web-{digest[:12]}"
        path = self.root / f"{release_id}.html"
        if not path.exists() or path.read_bytes() != payload:
            atomic_write_bytes(path, payload)
        return {"release_id": release_id, "asset": path.name, "sha256": digest, "role": role, "created_at": time.time(), "contract": "hawking.web.contract.v1"}

    def bootstrap(self) -> Dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self._read()
        if not manifest.get("current"):
            manifest = {"schema": "hawking.web.release.v1", "current": self._asset(self.source.read_bytes(), "current"), "candidate": None, "previous": None}
            atomic_write_json(self.manifest_path, manifest)
        return manifest

    def stage(self) -> Dict[str, Any]:
        manifest = self.bootstrap()
        manifest["candidate"] = self._asset(self.source.read_bytes(), "candidate")
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def activate(self) -> Dict[str, Any]:
        manifest = self.bootstrap()
        candidate = manifest.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError("no H-Web candidate release is staged")
        asset = self.root / str(candidate.get("asset") or "")
        if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != candidate.get("sha256"):
            raise ValueError("candidate H-Web asset integrity check failed")
        manifest["previous"] = manifest.get("current")
        candidate["role"] = "current"
        manifest["current"] = candidate
        manifest["candidate"] = None
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def rollback(self) -> Dict[str, Any]:
        manifest = self.bootstrap()
        previous = manifest.get("previous")
        if not isinstance(previous, dict):
            raise ValueError("no previous H-Web release is available")
        manifest["previous"], manifest["current"] = manifest.get("current"), previous
        manifest["current"]["role"] = "current"
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def current_bytes(self) -> tuple[bytes, Dict[str, Any]]:
        manifest = self.bootstrap()
        current = manifest.get("current")
        if not isinstance(current, dict):
            raise ValueError("H-Web has no current release")
        asset = self.root / str(current.get("asset") or "")
        return asset.read_bytes(), manifest
