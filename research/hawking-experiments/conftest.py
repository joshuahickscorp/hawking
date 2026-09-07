"""Pytest path shim when tests are collected under hawking-experiments/."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
for _p in (
    _REPO / "hawking-experiments" / "frankenstein" / "condense",
    _REPO / "hawking-experiments" / "frankenstein" / "operators",
):
    _s = str(_p)
    if _p.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)
