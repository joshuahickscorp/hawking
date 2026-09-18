"""Compatibility re-export for the Hawking-owned four-slot scheduler."""
from hawking.genesis_slots import (
    HARD_CAP,
    PREEMPTIBLE_SLOT,
    QUALIFICATION_PURPOSE,
    SLOT_ROLES,
    FourSlotScheduler,
    Slot,
    SlotCapError,
    SlotError,
)

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
