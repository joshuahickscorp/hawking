"""Re-export the four-slot scheduler that lives under tools/agentos/."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "tools" / "agentos" / "slots.py"
if not _PATH.is_file():
    raise ImportError(f"four-slot scheduler missing at {_PATH}")
_spec = importlib.util.spec_from_file_location("hawking_agentos_slots", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load four-slot scheduler from {_PATH}")
_mod = importlib.util.module_from_spec(_spec)
# dataclasses in 3.14 look up cls.__module__ in sys.modules during decoration.
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

FourSlotScheduler = _mod.FourSlotScheduler
Slot = _mod.Slot
SlotError = _mod.SlotError
SlotCapError = _mod.SlotCapError
SLOT_ROLES = _mod.SLOT_ROLES
HARD_CAP = _mod.HARD_CAP
PREEMPTIBLE_SLOT = _mod.PREEMPTIBLE_SLOT
QUALIFICATION_PURPOSE = _mod.QUALIFICATION_PURPOSE

__all__ = [
    "HARD_CAP",
    "PREEMPTIBLE_SLOT",
    "QUALIFICATION_PURPOSE",
    "SLOT_ROLES",
    "FourSlotScheduler",
    "Slot",
    "SlotCapError",
    "SlotError",
]
