"""Repo-root pytest path shim for hawking-experiments modules.

``hawking-experiments`` contains a dash and is not a valid Python package.
Frankenstein operators/condense and prometheus tools are importable by bare
module name once their directories are on ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
# Insert order: operators last so it is searched first (condense wrappers share names).
for _p in (
    _REPO / "hawking-experiments" / "prometheus" / "tools",
    _REPO / "hawking-experiments" / "frankenstein" / "condense",
    _REPO / "hawking-experiments" / "frankenstein" / "operators",
):
    _s = str(_p)
    if _p.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)
