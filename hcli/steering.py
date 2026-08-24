from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .persist import atomic_write_json

if TYPE_CHECKING:
    from .ledger import Ledger

STEER_KINDS = ("knowledge", "correction", "constraint")

_ADD_RE = re.compile(r"(?is)^\s*add(?:\s+obligation)?\s*[:=]\s*(.+)$")
_ALTER_RE = re.compile(
    r"(?is)^\s*(?:alter|amend|change)\s+(G\d+)\s*[:=]\s*(.+)$"
)
_REMOVE_RE = re.compile(r"(?is)^\s*remove\s+(G\d+)\s*\.?\s*$")
_MARK_VERIFIED_RE = re.compile(
    r"(?is)(?:mark|set)\s+(G\d+)\s+(?:as\s+)?VERIFIED"
)
_STRIP_VERIFIED_RE = re.compile(
    r"(?is)\s*(?:mark|set)\s+G\d+\s+(?:as\s+)?VERIFIED\s*"
)


def _with_steer_cite(text: str, steer_id: str) -> str:
    tag = f"(steer {steer_id})"
    if tag in text:
        return text
    text = text.rstrip()
    return f"{text} {tag}" if text else tag


def _strip_verified_instruction(text: str) -> str:
    return _STRIP_VERIFIED_RE.sub(" ", text).strip()


class SteerKindError(ValueError):
    """Raised when apply_constraint is called on a non-constraint steer."""


@dataclass
class SteerEvent:
    id: str
    text: str
    session_id: str
    timestamp: float
    applied: bool = False
    applied_at: Optional[float] = None
    kind: str = "knowledge"

    def __post_init__(self) -> None:
        if self.kind not in STEER_KINDS:
            raise ValueError(
                f"invalid steer kind {self.kind!r}; "
                f"expected one of {STEER_KINDS}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "applied": self.applied,
            "applied_at": self.applied_at,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SteerEvent":
        return cls(
            id=data["id"],
            text=data["text"],
            session_id=data["session_id"],
            timestamp=data["timestamp"],
            applied=data.get("applied", False),
            applied_at=data.get("applied_at"),
            kind=data.get("kind", "knowledge"),
        )


class SteeringQueue:
    def __init__(self, workspace: str, session_id: str):
        self.workspace = workspace
        self.session_id = session_id
        self._dir = os.path.join(workspace, ".hcli", "steering")
        os.makedirs(self._dir, exist_ok=True)
        self._path = os.path.join(self._dir, f"{session_id}.json")
        self._events: List[SteerEvent] = []
        self._load()

    def _load(self):
        if os.path.isfile(self._path):
            try:
                with open(self._path) as f:
                    data = json.load(f)
                self._events = [SteerEvent.from_dict(e) for e in data]
            except Exception:
                self._events = []

    def _save(self):
        atomic_write_json(self._path, [e.to_dict() for e in self._events])

    def enqueue(self, text: str, kind: str = "knowledge") -> SteerEvent:
        event = SteerEvent(
            id=str(uuid.uuid4()),
            text=text,
            session_id=self.session_id,
            timestamp=time.time(),
            kind=kind,
        )
        self._events.append(event)
        self._save()
        return event

    def pending(self) -> List[SteerEvent]:
        return [e for e in self._events if not e.applied]

    def apply_pending(self) -> List[SteerEvent]:
        applied = []
        for e in self._events:
            if not e.applied:
                e.applied = True
                e.applied_at = time.time()
                applied.append(e)
        if applied:
            self._save()
        return applied

    def clear_applied(self):
        self._events = [e for e in self._events if not e.applied]
        self._save()

    def all(self) -> List[SteerEvent]:
        return list(self._events)

    def apply_constraint(self, event: SteerEvent, ledger: "Ledger") -> None:
        """Amend ``ledger`` from a constraint steer. Never marks VERIFIED.

        mission.py (once it exists) should call this only for constraint
        events, immediately after enqueue, at the point a steer is absorbed
        into the mission's obligation set:

            event = queue.enqueue(text, kind=kind)
            if event.kind == "constraint":
                queue.apply_constraint(event, ledger)
            # knowledge / correction: do not call apply_constraint.

        Calling this on knowledge or correction raises ``SteerKindError``.
        Persistence reuses ``_save``; the event is marked applied.
        """
        if event.kind != "constraint":
            raise SteerKindError(
                f"apply_constraint requires kind='constraint', got {event.kind!r}"
            )
        self._amend_ledger(event, ledger)
        if all(event is not existing for existing in self._events):
            self._events.append(event)
        event.applied = True
        event.applied_at = time.time()
        self._save()

    def _amend_ledger(self, event: SteerEvent, ledger: "Ledger") -> None:
        stripped = (event.text or "").strip()
        add_match = _ADD_RE.match(stripped)
        alter_match = _ALTER_RE.match(stripped)
        remove_match = _REMOVE_RE.match(stripped)
        mark_match = _MARK_VERIFIED_RE.search(stripped)

        if remove_match:
            ledger.remove(remove_match.group(1))
            return
        if alter_match:
            body = _strip_verified_instruction(alter_match.group(2).strip())
            ledger.replace_text(
                alter_match.group(1),
                _with_steer_cite(body, event.id),
            )
            return
        if add_match:
            body = _strip_verified_instruction(add_match.group(1).strip())
            ledger.add(_with_steer_cite(body, event.id))
            return
        if mark_match:
            # A steer never marks anything VERIFIED. Status is unchanged.
            return
        body = _strip_verified_instruction(stripped)
        ledger.add(_with_steer_cite(body, event.id))
