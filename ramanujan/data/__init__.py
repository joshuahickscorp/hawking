"""Locally generated Ramanujan data sources D1–D4, D6, D7 from pinned Mathlib.

Does not flip RAMANUJAN_RESEARCH_AUTHORIZED. Does not touch Math-Preserve.
Does not modify the Mathlib checkout; only reads it.
"""
from __future__ import annotations

__all__ = ["SCHEMA", "LOCAL_SOURCE_IDS"]

SCHEMA = "hawking.ramanujan.data.v1"
LOCAL_SOURCE_IDS = ("D1", "D2", "D3", "D4", "D6", "D7")
