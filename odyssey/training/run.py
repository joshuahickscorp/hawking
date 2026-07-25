#!/usr/bin/env python3.12
"""Odyssey stage runner. Refuses to start unless the fence says otherwise.

The fence is read, never written. Building the package and authorizing the run are
different acts by construction, so no amount of re-running the builder can start
training.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FENCE = HERE.parent / "launch" / "ODYSSEY_LAUNCH_AUTHORIZED"
STOP = HERE.parent / "launch" / "STOP"


def authorized() -> bool:
    return FENCE.is_file() and FENCE.read_text().strip().lower() == "true"


def main(argv: list[str]) -> int:
    if STOP.is_file():
        print("ODYSSEY_STOPPED: emergency stop file present", file=sys.stderr)
        return 2
    if not authorized():
        print("ODYSSEY_LAUNCH_AUTHORIZED=false -- refusing to start.", file=sys.stderr)
        print(f"Authorize deliberately by writing 'true' to {FENCE}", file=sys.stderr)
        return 1
    print("authorized; stage execution is implemented in the session that starts Odyssey")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
