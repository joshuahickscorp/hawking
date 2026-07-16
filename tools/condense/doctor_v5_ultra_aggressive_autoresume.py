#!/usr/bin/env python3.12
"""Fail-closed LaunchAgent entrypoint for the aggressive-v2 supervisor."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import doctor_v5_ultra_aggressive_queue as queue


ROOT = Path(__file__).resolve().parents[2]
ULTRA_ROOT = ROOT / "reports/condense/doctor_v5_ultra"
CONTROL = ULTRA_ROOT / "control.json"
STATE = ULTRA_ROOT / "queue_state.json"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object")
    return value


def _marker() -> dict:
    marker = queue._read_json(queue.MARKER, root=queue.STAGE_ROOT)
    errors = queue._marker_errors(marker, verify_files=True)
    if errors:
        raise ValueError("invalid aggressive marker: " + "; ".join(errors))
    return marker


def main() -> int:
    try:
        control = _read(CONTROL)
        state = _read(STATE)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return 0
    if state.get("status") == "complete":
        return 0
    # The aggressive launch agent is installed only by its atomic transaction.
    # Absence/tampering must never fall back to a different queue generation.
    try:
        marker = _marker()
    except (OSError, KeyError, TypeError, ValueError, queue.AggressiveQueueError):
        return 2
    if control.get("mode") != "run":
        return 0
    env = os.environ.copy()
    env[queue.ENV_OVERLAY] = marker["overlay"]["path"]
    env[queue.ENV_OVERLAY_SHA256] = marker["overlay_sha256"]
    env[queue.ENV_MARKER_SHA256] = marker["marker_sha256"]
    return subprocess.run(
        [sys.executable, str(Path(queue.__file__).resolve()), "start"],
        cwd=ROOT, env=env, check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
