"""Internal compatibility view of Hawking's canonical Genesis lineage.

Research evidence may retain its historic import paths.  Runtime code belongs
to :mod:`hawking.lineage`; submodules below are aliases to those exact modules
so class identity and durable state contracts cannot split.
"""
from __future__ import annotations

import importlib
import sys

from hawking import lineage as _canonical


for _name in (
    "bus",
    "canon",
    "continuity",
    "cycle",
    "four_slots",
    "identity",
    "lifecycle",
    "promotion",
    "state",
    "testing",
    "transfer",
):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(
        f"hawking.lineage.{_name}"
    )

# A legacy import resolves to the canonical package object, not a re-export
# copy.  That keeps ``isinstance`` and module-level singleton behavior intact.
sys.modules[__name__] = _canonical
