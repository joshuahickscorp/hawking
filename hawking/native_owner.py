"""Stable native helper ownership for macOS-sensitive Hawking capabilities.

The native helper is an implementation owner, not a Goal or permission
database.  This module gives it one durable install identity and the same
current/candidate/previous promotion shape used by the other Hawking release
owners.  A source-checkout Swift build is never selected implicitly: a
workspace-local binary can be useful as an explicitly requested fixture, but
it is not a production permission owner.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .persist import atomic_write_json


NATIVE_OWNER_SCHEMA = "hawking.native.permission_owner.v1"
NATIVE_RELEASE_SCHEMA = "hawking.native.permission_release.v1"
STABLE_OWNER_IDENTIFIER = "com.hawking.native-helper"
STABLE_OWNER_NAME = "hawking-native-helper"
OWNER_ENV = "HAWKING_NATIVE_OWNER_ROOT"
HELPER_ENV = "HAWKING_NATIVE_HELPER_BIN"


class NativeOwnerError(RuntimeError):
    """A native owner cannot be admitted or resolved."""


def owner_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    raw = explicit or os.environ.get(OWNER_ENV)
    if raw:
        return Path(raw).expanduser().resolve()
    # Keep this outside a repository checkout.  TCC identifies the executable,
    # while the release record belongs to the installed Hawking owner.
    return (Path.home() / "Library" / "Application Support" / "Hawking" / "native-owner").resolve()


def _safe_name(value: Any) -> str:
    text = str(value or "").strip()
    return "".join(ch for ch in text if ch.isalnum() or ch in "._-")[:160] or "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _codesign_path() -> str:
    configured = os.environ.get("HAWKING_CODESIGN_BIN")
    if configured:
        return configured
    return "/usr/bin/codesign" if Path("/usr/bin/codesign").is_file() else "codesign"


def inspect_code_identity(path: str | os.PathLike[str] | None) -> Dict[str, Any]:
    """Inspect the actual code signature; never infer authority from a filename.

    ``codesign -d -r-`` prints the designated requirement on stderr on macOS.
    The complete requirement is retained as a diagnostic string, while the
    stable identifier and signature verification are separate facts.
    """
    candidate = Path(path).expanduser().resolve() if path else None
    base: Dict[str, Any] = {
        "schema": "hawking.native.code_identity.v1",
        "path": str(candidate) if candidate else None,
        "platform": sys.platform,
        "stable_identifier": STABLE_OWNER_IDENTIFIER,
        "signed": False,
        "verified": False,
        "stable_owner": False,
        "identifier": None,
        "team_identifier": None,
        "designated_requirement": None,
        "file_sha256": _sha256(candidate) if candidate and candidate.is_file() else "",
    }
    if candidate is None or not candidate.is_file():
        base.update({"state": "MISSING", "reason": "native helper binary is not present"})
        return base
    if sys.platform != "darwin":
        base.update({
            "state": "UNAVAILABLE",
            "reason": "macOS code identity is only inspectable on macOS",
        })
        return base
    codesign = _codesign_path()
    try:
        verified = subprocess.run(
            [codesign, "--verify", "--strict", str(candidate)],
            capture_output=True, text=True, timeout=8.0, check=False,
        )
        details = subprocess.run(
            [codesign, "-d", "-r-", "--verbose=4", str(candidate)],
            capture_output=True, text=True, timeout=8.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        base.update({"state": "UNKNOWN", "reason": f"codesign unavailable: {type(exc).__name__}"})
        return base
    text = "\n".join(item for item in (details.stdout, details.stderr) if item)
    identifier = None
    team = None
    requirement = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Identifier="):
            identifier = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("TeamIdentifier="):
            team = stripped.split("=", 1)[1].strip()
        elif stripped.lower().startswith("designated requirement"):
            requirement = stripped.split(":", 1)[1].strip() if ":" in stripped else stripped
        elif not requirement and "designated =>" in stripped.lower():
            requirement = stripped.split("=>", 1)[1].strip()
    # On some Xcode versions -r- emits the requirement without a label.
    if not requirement:
        for line in text.splitlines():
            if "designated =>" in line.lower():
                requirement = line.split("=>", 1)[1].strip()
                break
    is_verified = verified.returncode == 0
    stable = bool(is_verified and identifier == STABLE_OWNER_IDENTIFIER and requirement)
    base.update({
        "signed": bool(identifier),
        "verified": is_verified,
        "stable_owner": stable,
        "identifier": identifier,
        "team_identifier": team,
        "designated_requirement": requirement,
        "state": "READY" if stable else "MISSING",
        "reason": (
            "stable designated requirement verified"
            if stable else
            (verified.stderr or details.stderr or "binary is not a verified stable Hawking owner").strip()[-600:]
        ),
        "codesign_returncode": verified.returncode,
    })
    return base


def _release_path(root: Path, slot: str) -> Path:
    return root / slot / STABLE_OWNER_NAME


def stable_helper_candidates(workspace: str | os.PathLike[str] | None = None) -> list[tuple[Path, str]]:
    """Return ordered stable candidates, never an implicit source build."""
    candidates: list[tuple[Path, str]] = []
    for variable in (HELPER_ENV, "HAWKING_MACOS_HELPER_BIN"):
        raw = os.environ.get(variable)
        if raw:
            candidates.append((Path(raw).expanduser().resolve(), f"env:{variable}"))
    root = owner_root()
    candidates.extend([
        (_release_path(root, "current"), "native-owner.current"),
        (Path.home() / ".local" / "share" / "hawking" / "current" / STABLE_OWNER_NAME, "hawking-install.current"),
    ])
    # A caller may explicitly point at a staged install root, but a normal
    # workspace is intentionally not searched for .hawking/.../bin builds.
    if workspace:
        configured = Path(workspace).expanduser().resolve() / ".hawking" / "native-owner" / "current" / STABLE_OWNER_NAME
        candidates.append((configured, "workspace-native-owner.current"))
    seen: set[Path] = set()
    return [(path, source) for path, source in candidates if not (path in seen or seen.add(path))]


def resolve_native_helper(workspace: str | os.PathLike[str] | None = None) -> tuple[Path, Dict[str, Any]]:
    """Resolve one executable and its provenance, or fail closed."""
    for path, source in stable_helper_candidates(workspace):
        if path.is_file() and os.access(path, os.X_OK):
            identity = inspect_code_identity(path)
            identity["resolution"] = source
            if source.startswith("env:") and not identity.get("stable_owner"):
                # Explicit overrides remain useful for deterministic fixtures,
                # but their non-production identity is visible to diagnostics.
                identity["owner_mode"] = "explicit_override"
            else:
                identity["owner_mode"] = "stable_release"
            return path, identity
    raise NativeOwnerError(
        "no stable Hawking native helper is installed; build/sign "
        f"{STABLE_OWNER_NAME} and promote it to {owner_root() / 'current'}"
    )


class NativeHelperReleaseStore:
    """Current/candidate/previous promotion for the permission-owning helper."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = owner_root(root)
        self.manifest_path = self.root / "manifest.json"

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {
            "schema": NATIVE_RELEASE_SCHEMA,
            "owner": STABLE_OWNER_IDENTIFIER,
            "current": None,
            "candidate": None,
            "previous": None,
            "updated_at": None,
        }

    def load(self) -> Dict[str, Any]:
        if not self.manifest_path.is_file():
            return self._empty()
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return self._empty()
        if not isinstance(value, dict) or value.get("schema") != NATIVE_RELEASE_SCHEMA:
            return self._empty()
        return value

    def _save(self, document: Mapping[str, Any]) -> Dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        value = dict(document)
        value["schema"] = NATIVE_RELEASE_SCHEMA
        value["owner"] = STABLE_OWNER_IDENTIFIER
        value["updated_at"] = time.time()
        atomic_write_json(self.manifest_path, value)
        return value

    def stage(self, source: str | os.PathLike[str]) -> Dict[str, Any]:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file() or not os.access(source_path, os.X_OK):
            raise NativeOwnerError(f"candidate helper is not an executable file: {source_path}")
        target = _release_path(self.root, "candidate")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_path != target:
            shutil.copy2(source_path, target)
            target.chmod(target.stat().st_mode | 0o111)
        identity = inspect_code_identity(target)
        manifest = self.load()
        manifest["candidate"] = {
            "path": str(target),
            "staged_from": str(source_path),
            "staged_at": time.time(),
            "code_identity": identity,
        }
        return self._save(manifest)

    def activate(self) -> Dict[str, Any]:
        manifest = self.load()
        candidate = manifest.get("candidate") if isinstance(manifest.get("candidate"), Mapping) else {}
        candidate_path = Path(str(candidate.get("path") or _release_path(self.root, "candidate"))).resolve()
        if not candidate_path.is_file():
            raise NativeOwnerError("no staged native helper candidate")
        identity = inspect_code_identity(candidate_path)
        if not identity.get("stable_owner"):
            raise NativeOwnerError(
                "refusing to promote native helper: its verified designated requirement "
                f"is not {STABLE_OWNER_IDENTIFIER!r}"
            )
        current = manifest.get("current") if isinstance(manifest.get("current"), Mapping) else {}
        current_req = ((current.get("code_identity") or {}).get("designated_requirement")
                       if isinstance(current.get("code_identity"), Mapping) else None)
        if current_req and current_req != identity.get("designated_requirement"):
            raise NativeOwnerError(
                "refusing to promote native helper: designated requirement changed; "
                "reauthorization risk must be reviewed"
            )
        current_path = _release_path(self.root, "current")
        previous_path = _release_path(self.root, "previous")
        if current_path.is_file():
            previous_path.parent.mkdir(parents=True, exist_ok=True)
            if previous_path.exists():
                previous_path.unlink()
            shutil.copy2(current_path, previous_path)
            previous_path.chmod(previous_path.stat().st_mode | 0o111)
            manifest["previous"] = {
                "path": str(previous_path),
                "activated_at": time.time(),
                "code_identity": inspect_code_identity(previous_path),
            }
        current_path.parent.mkdir(parents=True, exist_ok=True)
        if current_path.exists():
            current_path.unlink()
        shutil.copy2(candidate_path, current_path)
        current_path.chmod(current_path.stat().st_mode | 0o111)
        manifest["current"] = {
            "path": str(current_path),
            "activated_at": time.time(),
            "code_identity": inspect_code_identity(current_path),
        }
        manifest["candidate"] = None
        return self._save(manifest)

    def rollback(self) -> Dict[str, Any]:
        manifest = self.load()
        previous = manifest.get("previous") if isinstance(manifest.get("previous"), Mapping) else {}
        previous_path = Path(str(previous.get("path") or _release_path(self.root, "previous"))).resolve()
        if not previous_path.is_file():
            raise NativeOwnerError("no previous native helper release is available")
        identity = inspect_code_identity(previous_path)
        if not identity.get("stable_owner"):
            raise NativeOwnerError("refusing rollback: previous release is not a verified stable owner")
        current_path = _release_path(self.root, "current")
        if current_path.is_file():
            candidate_path = _release_path(self.root, "candidate")
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current_path, candidate_path)
            candidate_path.chmod(candidate_path.stat().st_mode | 0o111)
            manifest["candidate"] = {
                "path": str(candidate_path),
                "staged_at": time.time(),
                "code_identity": inspect_code_identity(candidate_path),
            }
        shutil.copy2(previous_path, current_path)
        current_path.chmod(current_path.stat().st_mode | 0o111)
        manifest["current"] = {
            "path": str(current_path),
            "activated_at": time.time(),
            "code_identity": inspect_code_identity(current_path),
        }
        manifest["previous"] = None
        return self._save(manifest)

    def status(self) -> Dict[str, Any]:
        manifest = self.load()
        current = manifest.get("current") if isinstance(manifest.get("current"), Mapping) else {}
        path = current.get("path") if current else str(_release_path(self.root, "current"))
        identity = inspect_code_identity(path)
        return {
            "schema": NATIVE_OWNER_SCHEMA,
            "owner": STABLE_OWNER_IDENTIFIER,
            "name": STABLE_OWNER_NAME,
            "release": manifest,
            "current": identity,
            "stable_permission_owner": bool(identity.get("stable_owner")),
        }


__all__ = [
    "HELPER_ENV", "NATIVE_OWNER_SCHEMA", "NATIVE_RELEASE_SCHEMA",
    "NativeHelperReleaseStore", "NativeOwnerError", "STABLE_OWNER_IDENTIFIER",
    "STABLE_OWNER_NAME", "inspect_code_identity", "owner_root",
    "resolve_native_helper", "stable_helper_candidates",
]
