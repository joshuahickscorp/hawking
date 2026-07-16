#!/usr/bin/env python3.12
"""Self-heal entrypoint that understands the optional acceleration marker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ULTRA_ROOT = ROOT / "reports/condense/doctor_v5_ultra"
CONTROL = ULTRA_ROOT / "control.json"
STATE = ULTRA_ROOT / "queue_state.json"
MARKER = ULTRA_ROOT / "staged_acceleration/active_stack.json"
BASE_QUEUE = ROOT / "tools/condense/doctor_v5_ultra_queue.py"
ACCEL_QUEUE = ROOT / "tools/condense/doctor_v5_ultra_accelerated_queue.py"
SCHEMA = "hawking.doctor_v5_acceleration_active_marker.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object")
    return value


def _hash_regular(path: Path) -> str:
    resolved = path.resolve(strict=True)
    resolved.relative_to(ROOT.resolve())
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("accelerated queue binding is not a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                                        after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("accelerated queue binding changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _artifact_matches(row: object, expected: Path) -> bool:
    if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
        return False
    try:
        path = Path(row["path"]).resolve(strict=True)
        return (path == expected.resolve(strict=True)
                and path.stat().st_size == row["bytes"]
                and _hash_regular(path) == row["sha256"])
    except (OSError, KeyError, TypeError, ValueError):
        return False


def _marker() -> dict[str, Any] | None:
    if not MARKER.exists():
        return None
    value = _read(MARKER)
    expected = {"schema", "activated_at", "overlay_path", "overlay_sha256",
                "pending_runtime_generation_sha256", "accelerated_queue",
                "accelerated_autoresume", "marker_sha256"}
    if set(value) != expected or value.get("schema") != SCHEMA:
        raise ValueError("acceleration marker keys/schema invalid")
    payload = {key: row for key, row in value.items() if key != "marker_sha256"}
    if value.get("marker_sha256") != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError("acceleration marker hash mismatch")
    overlay = Path(value["overlay_path"]).resolve(strict=True)
    overlay.relative_to((ULTRA_ROOT / "staged_acceleration").resolve())
    observed = json.loads(overlay.read_text(encoding="utf-8"))
    if not isinstance(observed, dict) \
            or observed.get("overlay_sha256") != value["overlay_sha256"] \
            or observed.get("overlay_sha256") != hashlib.sha256(_canonical({
                key: row for key, row in observed.items() if key != "overlay_sha256"
            })).hexdigest():
        raise ValueError("active overlay identity changed")
    if not _artifact_matches(value.get("accelerated_queue"), ACCEL_QUEUE) \
            or not _artifact_matches(value.get("accelerated_autoresume"), Path(__file__)):
        raise ValueError("accelerated entrypoint source identity changed")
    return value


def main() -> int:
    try:
        control = _read(CONTROL)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return 0
    try:
        state = _read(STATE)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        state = {}
    if control.get("mode") != "run" or state.get("status") == "complete":
        return 0
    try:
        marker = _marker()
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError):
        # A partial/tampered activation must never silently fall back to a
        # different supervisor implementation.
        return 2
    queue = BASE_QUEUE
    env = os.environ.copy()
    if marker is not None:
        queue = ACCEL_QUEUE
        env["DOCTOR_V5_STACKED_ADMISSION_OVERLAY"] = marker["overlay_path"]
        env["DOCTOR_V5_STACKED_ADMISSION_SHA256"] = marker["overlay_sha256"]
    return subprocess.run([sys.executable, str(queue), "start"], cwd=ROOT, env=env,
                          check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
