"""Locally generated Ramanujan data sources D1–D4, D6, D7 from pinned Mathlib.

Does not flip RAMANUJAN_RESEARCH_AUTHORIZED. Does not touch Math-Preserve.
Does not modify the Mathlib checkout; only reads it.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["SCHEMA", "LOCAL_SOURCE_IDS"]

SCHEMA = "hawking.ramanujan.data.v1"
LOCAL_SOURCE_IDS = ("D1", "D2", "D3", "D4", "D6", "D7")

# Keep the established ``ramanujan.data.*`` imports stable while separating
# runnable pipeline code, immutable corpora, and its tests.
_ROOT = Path(__file__).resolve().parent
for _module_root in (_ROOT / "pipeline", _ROOT / "tests"):
    if _module_root.is_dir():
        __path__.append(str(_module_root))
