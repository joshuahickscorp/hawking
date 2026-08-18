#!/usr/bin/env python3
"""Four-slot Genesis scheduler. Hard cap. Slot 3 is preemptible.

Slot 0 parent/integrator
Slot 1 child A representation
Slot 2 child B execution
Slot 3 protected test / temporary experiment — never permanently occupied;
the scheduler must be able to preempt it for qualification.

stdlib only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


HARD_CAP = 4
SLOT_ROLES: dict[int, str] = {
    0: "parent_integrator",
    1: "child_a_representation",
    2: "child_b_execution",
    3: "protected_test_or_temp_experiment",
}
PREEMPTIBLE_SLOT = 3
QUALIFICATION_PURPOSE = "qualification"


class SlotError(ValueError):
    """Unlawful slot operation."""


class SlotCapError(SlotError):
    """Hard cap of four slots."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class Slot:
    index: int
    role: str
    occupant_id: str | None = None
    purpose: str = ""
    permanent: bool = False
    occupied_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "role": self.role,
            "occupant_id": self.occupant_id,
            "purpose": self.purpose,
            "permanent": self.permanent,
            "occupied_at": self.occupied_at,
        }

    @property
    def empty(self) -> bool:
        return self.occupant_id is None


class FourSlotScheduler:
    """Exactly four slots. A fifth occupant is refused."""

    def __init__(self) -> None:
        self._slots: dict[int, Slot] = {
            index: Slot(index=index, role=role) for index, role in SLOT_ROLES.items()
        }
        self._events: list[dict[str, Any]] = []

    def slot(self, index: int) -> Slot:
        if index not in self._slots:
            raise SlotCapError(f"hard cap {HARD_CAP} slots; refused index {index}")
        return self._slots[index]

    def occupied(self) -> list[Slot]:
        return [s for s in self._slots.values() if not s.empty]

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        self._events.append({"ts": _utc_now(), "kind": kind, **payload})

    def occupy(
        self,
        index: int,
        occupant_id: str,
        *,
        role: str | None = None,
        purpose: str = "",
        permanent: bool = False,
    ) -> Slot:
        if not isinstance(index, int) or isinstance(index, bool) or index not in SLOT_ROLES:
            raise SlotCapError(f"hard cap {HARD_CAP} slots; refused index {index!r}")
        if not isinstance(occupant_id, str) or not occupant_id.strip():
            raise SlotError("occupant_id must be a non-empty string")
        expected = SLOT_ROLES[index]
        if role is not None and role != expected:
            raise SlotError(f"slot {index} is {expected}, not {role}")
        if index == PREEMPTIBLE_SLOT and permanent:
            raise SlotError("slot 3 must not be permanently occupied")
        current = self._slots[index]
        if current.occupant_id is not None:
            if index == PREEMPTIBLE_SLOT:
                raise SlotError(
                    "slot 3 is occupied; call request_qualification to preempt it"
                )
            raise SlotError(f"slot {index} occupied by {current.occupant_id}")
        if index != PREEMPTIBLE_SLOT:
            for home in (0, 1, 2):
                other = self._slots[home]
                if home != index and other.occupant_id == occupant_id:
                    raise SlotError(
                        f"{occupant_id} already holds home slot {home}; "
                        "the same occupant cannot hold two home slots"
                    )
        slot = Slot(
            index=index,
            role=expected,
            occupant_id=occupant_id.strip(),
            purpose=purpose,
            permanent=False if index == PREEMPTIBLE_SLOT else permanent,
            occupied_at=_utc_now(),
        )
        self._slots[index] = slot
        self._record("occupy", {"index": index, "occupant_id": slot.occupant_id, "purpose": purpose})
        return slot

    def release(self, index: int, *, reason: str = "") -> Slot | None:
        if index not in SLOT_ROLES:
            raise SlotCapError(f"hard cap {HARD_CAP} slots; refused index {index}")
        previous = self._slots[index]
        if previous.empty:
            return None
        self._slots[index] = Slot(index=index, role=SLOT_ROLES[index])
        self._record(
            "release",
            {
                "index": index,
                "occupant_id": previous.occupant_id,
                "reason": reason,
            },
        )
        return previous

    def preempt(self, index: int, *, reason: str) -> Slot | None:
        if index != PREEMPTIBLE_SLOT:
            raise SlotError("only slot 3 is preemptible")
        previous = self.release(index, reason=f"preempt: {reason}")
        if previous is not None:
            self._record(
                "preempt",
                {"index": index, "occupant_id": previous.occupant_id, "reason": reason},
            )
        return previous

    def request_qualification(self, occupant_id: str, *, purpose: str = QUALIFICATION_PURPOSE) -> dict[str, Any]:
        """Seat a qualification occupant in slot 3, preempting any temp experiment."""
        preempted = None
        if not self._slots[PREEMPTIBLE_SLOT].empty:
            previous = self.preempt(PREEMPTIBLE_SLOT, reason="qualification requires slot 3")
            preempted = None if previous is None else previous.to_dict()
        slot = self.occupy(
            PREEMPTIBLE_SLOT,
            occupant_id,
            purpose=purpose,
            permanent=False,
        )
        return {
            "slot": slot.to_dict(),
            "preempted": preempted,
            "qualification": True,
        }

    def mark_permanent(self, index: int, permanent: bool = True) -> Slot:
        if index not in SLOT_ROLES:
            raise SlotCapError(f"hard cap {HARD_CAP} slots; refused index {index}")
        if index == PREEMPTIBLE_SLOT and permanent:
            raise SlotError("slot 3 must not be permanently occupied")
        slot = self._slots[index]
        if slot.empty:
            raise SlotError(f"slot {index} is empty")
        slot.permanent = permanent
        return slot

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "hawking.agentos.four_slots.v1",
            "hard_cap": HARD_CAP,
            "slots": {str(i): self._slots[i].to_dict() for i in range(HARD_CAP)},
            "occupied_count": len(self.occupied()),
            "events": list(self._events),
        }


if __name__ == "__main__":
    sched = FourSlotScheduler()
    sched.occupy(0, "parent")
    sched.occupy(1, "child-a", purpose="representation")
    sched.occupy(2, "child-b", purpose="execution")
    sched.occupy(3, "temp-exp", purpose="experiment")
    result = sched.request_qualification("child-a")
    assert result["preempted"]["occupant_id"] == "temp-exp"
    assert sched.slot(3).occupant_id == "child-a"
    try:
        sched.occupy(4, "fifth")
        raise SystemExit("hard cap failed to refuse slot 4")
    except SlotCapError:
        pass
    try:
        sched.mark_permanent(3, True)
        raise SystemExit("slot 3 permanent mark was not refused")
    except SlotError:
        pass
    print(json.dumps({"self_check": "ok", "state": sched.to_dict()}, indent=2))
