#!/usr/bin/env python3.12
"""Unattended self-heal for the detached doctor_v5_ultra campaign.

Run on login/reboot and on a periodic heartbeat by a user LaunchAgent. It resumes the
supervisor ONLY when the campaign is supposed to be running (control mode "run") but no
live owner process exists, which is exactly the reboot/crash case. It deliberately does
nothing when the mode is "drain" or "pause", so an intentional stop is never overridden.

The queue's own "start" verb is a no-op when a live owner already exists and respawns it
when it is dead, and it does not change an already-"run" control mode. So this guard is
safe to fire on a short interval: at most one supervisor ever runs. No em dashes.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "reports" / "condense" / "doctor_v5_ultra" / "control.json"
QUEUE = ROOT / "tools" / "condense" / "doctor_v5_ultra_queue.py"


def main() -> int:
    try:
        mode = json.loads(CONTROL.read_text()).get("mode")
    except (OSError, ValueError):
        return 0
    if mode != "run":
        return 0
    return subprocess.run(
        [sys.executable, str(QUEUE), "start"], cwd=str(ROOT), check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
