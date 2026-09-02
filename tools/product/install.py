"""Artifact install/update with a digest stamp.

`hcli.cli.install_shims` already copies the hcli package to
`~/.local/share/hcli/build-*`, writes `install.json` `{source, digest,
installed}`, keeps three snapshots, and `warn_if_stale` compares the source
digest to the stamp. That path is not config-driven and is not this module.

This installer copies a configured artifact tree into a configured product
home, stamps the same kind of digest, and treats staleness as a gate rather
than a warning. It never writes under `/Volumes/corpdrive`. It never
restarts a worker.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from tools.odyssey.product_boundary import resolve_artifact
from tools.product.config import ConfigClosed

EVIDENCE_TIER = "STATIC"
STAMP_NAME = "install.json"
STAMP_SCHEMA = "hawking.product.install_stamp.v1"
LIVE_VOLUME = "/Volumes/corpdrive"


class InstallError(ValueError):
    """Install/update refused. Nothing was left half-written on the dest path."""


def assert_not_live_volume(path: Path, *, op: str) -> Path:
    resolved = Path(path).resolve()
    text = str(resolved)
    if text == LIVE_VOLUME or text.startswith(LIVE_VOLUME + os.sep):
        raise InstallError(f"refusing to {op} live volume path {text}")
    return resolved


def artifact_digest(root: str | Path) -> str:
    """Content hash of an artifact tree. Bytes, not mtimes. Skips the stamp."""
    root_path = Path(root)
    digest = hashlib.sha256()
    if root_path.is_file():
        digest.update(b"file:")
        digest.update(root_path.read_bytes())
        return digest.hexdigest()
    if not root_path.is_dir():
        raise InstallError(f"artifact is not a file or directory: {root_path}")
    files = sorted(
        p for p in root_path.rglob("*")
        if p.is_file() and p.name != STAMP_NAME
    )
    for path in files:
        rel = path.relative_to(root_path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _stamp_payload(source: Path, digest: str, slug: str) -> dict[str, Any]:
    return {
        "schema": STAMP_SCHEMA,
        "evidence_tier": EVIDENCE_TIER,
        "source": str(source),
        "digest": digest,
        "slug": slug,
        "installed": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
    }


def _write_stamp(dest: Path, payload: Mapping[str, Any]) -> None:
    (dest / STAMP_NAME).write_text(
        json.dumps(dict(payload), indent=2) + "\n",
        encoding="utf-8",
    )


def _read_stamp(dest: Path) -> Optional[dict[str, Any]]:
    stamp = dest / STAMP_NAME
    if not stamp.is_file():
        return None
    try:
        doc = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def resolve_install_paths(
    slug: str,
    config: Mapping[str, Any],
    *,
    source_root: str = "partial",
    dest_root: str = "install",
) -> tuple[Path, Path]:
    """Resolve source and dest from configuration. Calls `resolve_artifact`."""
    if not slug:
        raise InstallError("artifact slug is empty")
    roots = dict(config.get("artifact_roots") or {})
    if source_root not in roots:
        raise ConfigClosed(
            f"config.artifact_roots has no {source_root!r}; refusing to guess a source"
        )
    dest_key = dest_root if dest_root in roots else ("home" if "home" in roots else None)
    if dest_key is None:
        raise ConfigClosed(
            "config.artifact_roots has no install/home; refusing to guess a destination"
        )
    src = resolve_artifact(f"{source_root}:{slug}", config)
    dst = resolve_artifact(f"{dest_key}:{slug}", config)
    return Path(src["path"]), Path(dst["path"])


def staleness(source: str | Path, dest: str | Path) -> dict[str, Any]:
    """Compare the source digest to the dest stamp. Missing dest is stale."""
    source_path = Path(source)
    dest_path = Path(dest)
    if not dest_path.exists():
        return {
            "schema": "hawking.product.staleness.v1",
            "evidence_tier": EVIDENCE_TIER,
            "stale": True,
            "reason": "not installed",
            "source": str(source_path),
            "destination": str(dest_path),
            "source_digest": artifact_digest(source_path) if source_path.exists() else None,
            "installed_digest": None,
        }
    stamp = _read_stamp(dest_path)
    if stamp is None:
        return {
            "schema": "hawking.product.staleness.v1",
            "evidence_tier": EVIDENCE_TIER,
            "stale": True,
            "reason": "missing or corrupt stamp",
            "source": str(source_path),
            "destination": str(dest_path),
            "source_digest": artifact_digest(source_path) if source_path.exists() else None,
            "installed_digest": None,
        }
    current = artifact_digest(source_path)
    installed = stamp.get("digest")
    stale = current != installed
    return {
        "schema": "hawking.product.staleness.v1",
        "evidence_tier": EVIDENCE_TIER,
        "stale": stale,
        "reason": "source digest differs from stamp" if stale else "digests match",
        "source": str(source_path),
        "destination": str(dest_path),
        "source_digest": current,
        "installed_digest": installed,
    }


def install_artifact(
    source: str | Path,
    dest: str | Path,
    *,
    slug: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy `source` to `dest` with an install stamp. Atomic replace of dest."""
    source_path = Path(source)
    dest_path = Path(dest)
    assert_not_live_volume(dest_path, op="write")
    if not source_path.exists():
        raise InstallError(f"source is not present: {source_path}")
    if dest_path.exists() and not overwrite:
        raise InstallError(f"destination exists; refusing overwrite: {dest_path}")
    parent = dest_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_name(dest_path.name + ".installing")
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        if source_path.is_file():
            tmp.mkdir()
            shutil.copy2(source_path, tmp / source_path.name)
        else:
            shutil.copytree(source_path, tmp)
        digest = artifact_digest(tmp)
        _write_stamp(tmp, _stamp_payload(source_path, digest, slug))
        if dest_path.exists():
            shutil.rmtree(dest_path)
        os.replace(tmp, dest_path)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "schema": "hawking.product.install.v1",
        "evidence_tier": EVIDENCE_TIER,
        "action": "INSTALLED",
        "slug": slug,
        "source": str(source_path),
        "destination": str(dest_path),
        "digest": digest,
        "wrote": True,
        "overwrite": overwrite,
        "never_restart_healthy_worker": True,
    }


def update_artifact(
    source: str | Path,
    dest: str | Path,
    *,
    slug: str,
) -> dict[str, Any]:
    """Replace dest when the source digest no longer matches the stamp.

    Keeps one previous snapshot at `dest.prev`. Rolls back that rename if the
    new copy fails. Does not fetch. Does not restart a worker.
    """
    source_path = Path(source)
    dest_path = Path(dest)
    assert_not_live_volume(dest_path, op="write")
    if not dest_path.exists():
        result = install_artifact(source_path, dest_path, slug=slug)
        result["action"] = "INSTALLED"
        return result
    check = staleness(source_path, dest_path)
    if not check["stale"]:
        return {
            "schema": "hawking.product.install.v1",
            "evidence_tier": EVIDENCE_TIER,
            "action": "ALREADY_CURRENT",
            "slug": slug,
            "source": str(source_path),
            "destination": str(dest_path),
            "wrote": False,
            "stale": False,
            "never_restart_healthy_worker": True,
        }
    prev = dest_path.with_name(dest_path.name + ".prev")
    if prev.exists():
        shutil.rmtree(prev)
    os.replace(dest_path, prev)
    try:
        result = install_artifact(source_path, dest_path, slug=slug)
    except Exception:
        if dest_path.exists():
            shutil.rmtree(dest_path, ignore_errors=True)
        os.replace(prev, dest_path)
        raise
    result["action"] = "UPDATED"
    result["previous"] = str(prev)
    result["stale_before"] = True
    return result
