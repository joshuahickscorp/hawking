#!/usr/bin/env python3
"""Repository-owned entry point for the fail-closed Ascension V3 launch gate.

The controller emits a runtime copy under its protected record root as required
by Bible §19.  This source-owned entry point is the implementation authority.
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.ascension_lifecycle import evaluate_launch_gate, main  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(main())
