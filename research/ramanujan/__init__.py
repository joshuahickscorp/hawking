"""Ramanujan's non-authorizing local scaffold.

The public module names remain stable while the implementation is grouped below
``scaffold/``.  This is intentionally a compatibility path, not an authority
surface: see ``README.md`` and ``governance/boundary/HAWKING_COMPLETION_GATE.json``.
"""
from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parent
for _module_root in (
    _ROOT / "scaffold",
    _ROOT / "scaffold" / "core",
    _ROOT / "scaffold" / "research",
    _ROOT / "scaffold" / "guards",
    _ROOT / "scaffold" / "tooling",
    _ROOT / "scaffold" / "tests",
):
    if _module_root.is_dir():
        __path__.append(str(_module_root))
