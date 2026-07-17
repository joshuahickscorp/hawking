#!/usr/bin/env python3.12
"""Canonical successor state machine with journaled checkpoints and exact resume.

Implements the master-goal state machine (section 6.1) as ENFORCED code driven by the
hash-chained event log (succ_events). Every transition:
  - is validated against the allowed-transition table (illegal transitions fail closed);
  - appends an event to the durable log;
  - writes a journaled checkpoint atomically (temp + fsync + fsync-parent + validate);
  - binds the checkpoint to the current event-log head, so resume rejects ambiguous state.

Resume reconstructs the current state from the last checkpoint and refuses to continue
unless the checkpoint's event-head hash matches the live log head (no split-brain).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import EcoError, seal_field, sealed, now_iso, atomic_write_json, read_json_safe  # noqa: E402
from succ_events import EventLog  # noqa: E402

CHECKPOINT_SCHEMA = "hawking.successor.checkpoint.v1"

STATES: tuple[str, ...] = (
    "BOOT", "AUDIT", "WAIT_OLD_RELEASE", "IMPORT_LEGACY", "RECONCILE", "FIT_PRIORS",
    "CHOOSE_PARENT", "BRACKET_HORIZON", "DIAGNOSE", "PRESCRIBE", "MATERIALIZE_PROGRAM",
    "VALIDATE_PROGRAM", "RESOURCE_ADMISSION", "LAUNCH", "MONITOR", "CHECKPOINT", "ATTEST",
    "EVALUATE", "INGEST_RESULT", "UPDATE_FRONTIER", "RETIRE_DOMINATED",
    "RECLAIM_ELIGIBLE_BYTES", "CHOOSE_NEXT", "SEALED_PARENT", "DRAINED", "BLOCKED",
)

# Allowed successor states. The machine fails closed on any transition not listed here.
TRANSITIONS: dict[str, tuple[str, ...]] = {
    "BOOT": ("AUDIT",),
    "AUDIT": ("WAIT_OLD_RELEASE", "IMPORT_LEGACY", "BLOCKED"),
    # heartbeat self-loop while waiting; -> IMPORT_LEGACY on signed release; -> BLOCKED/DRAINED
    "WAIT_OLD_RELEASE": ("WAIT_OLD_RELEASE", "IMPORT_LEGACY", "BLOCKED", "DRAINED"),
    "IMPORT_LEGACY": ("RECONCILE", "BLOCKED"),
    "RECONCILE": ("FIT_PRIORS", "BLOCKED"),
    "FIT_PRIORS": ("CHOOSE_PARENT", "BLOCKED", "DRAINED"),
    "CHOOSE_PARENT": ("BRACKET_HORIZON", "DRAINED", "BLOCKED"),
    "BRACKET_HORIZON": ("DIAGNOSE", "CHOOSE_NEXT", "BLOCKED"),
    "DIAGNOSE": ("PRESCRIBE", "BLOCKED"),
    "PRESCRIBE": ("MATERIALIZE_PROGRAM", "BLOCKED"),
    "MATERIALIZE_PROGRAM": ("VALIDATE_PROGRAM", "BLOCKED"),
    "VALIDATE_PROGRAM": ("RESOURCE_ADMISSION", "BLOCKED"),
    "RESOURCE_ADMISSION": ("LAUNCH", "WAIT_OLD_RELEASE", "BLOCKED"),
    "LAUNCH": ("MONITOR", "BLOCKED"),
    "MONITOR": ("CHECKPOINT", "ATTEST", "BLOCKED"),
    "CHECKPOINT": ("MONITOR", "ATTEST"),
    "ATTEST": ("EVALUATE", "BLOCKED"),
    "EVALUATE": ("INGEST_RESULT", "BLOCKED"),
    "INGEST_RESULT": ("UPDATE_FRONTIER",),
    "UPDATE_FRONTIER": ("RETIRE_DOMINATED", "CHOOSE_NEXT"),
    "RETIRE_DOMINATED": ("RECLAIM_ELIGIBLE_BYTES", "CHOOSE_NEXT"),
    "RECLAIM_ELIGIBLE_BYTES": ("CHOOSE_NEXT",),
    "CHOOSE_NEXT": ("CHOOSE_PARENT", "BRACKET_HORIZON", "SEALED_PARENT", "DRAINED", "BLOCKED"),
    "SEALED_PARENT": ("CHOOSE_PARENT", "DRAINED"),
    "BLOCKED": ("AUDIT", "WAIT_OLD_RELEASE", "CHOOSE_NEXT", "DRAINED"),
    "DRAINED": ("BOOT",),
}


class StateError(EcoError):
    """Fail-closed state-machine error."""


class Controller:
    """Event-sourced controller: one event log + one journaled checkpoint."""

    def __init__(self, root: str | os.PathLike[str], *, generation: str = "gen-1",
                 controller_tree_sha256: str | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = EventLog(self.root / "events.jsonl")
        self.checkpoint_path = self.root / "checkpoint.json"
        self.generation = generation
        self.controller_tree_sha256 = controller_tree_sha256

    # -- lifecycle -----------------------------------------------------------------------
    def boot(self) -> dict[str, Any]:
        if self.log.head() is not None:
            raise StateError("boot refused: event log already has history (use resume)")
        self.log.append("state", {"state": "BOOT", "generation": self.generation})
        return self._write_checkpoint("BOOT", {})

    def current_state(self) -> str | None:
        for event in reversed(self.log.events()):
            if event.get("kind") == "state":
                return event["payload"].get("state")
        return None

    def transition(self, to_state: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if to_state not in STATES:
            raise StateError(f"unknown state {to_state}")
        cur = self.current_state()
        if cur is None:
            raise StateError("no current state; boot first")
        allowed = TRANSITIONS.get(cur, ())
        if to_state not in allowed:
            raise StateError(f"illegal transition {cur} -> {to_state} (allowed: {allowed})")
        self.log.append("state", {"state": to_state, "from": cur, **(payload or {})})
        return self._write_checkpoint(to_state, payload or {})

    def _write_checkpoint(self, state: str, payload: dict[str, Any]) -> dict[str, Any]:
        head = self.log.head()
        cp = {
            "schema": CHECKPOINT_SCHEMA,
            "generation": self.generation,
            "controller_tree_sha256": self.controller_tree_sha256,
            "state": state,
            "event_cursor_seq": head["seq"] if head else -1,
            "event_head_hash": head["chain_hash"] if head else None,
            "selected_parent": payload.get("selected_parent"),
            "candidate_identity": payload.get("candidate_identity"),
            "program_cursor": payload.get("program_cursor"),
            "resource_reservation": payload.get("resource_reservation"),
            "child_pgid": payload.get("child_pgid"),
            "notification_cursor": payload.get("notification_cursor"),
            "next_command": payload.get("next_command"),
            "written_at": now_iso(),
        }
        cp = seal_field(cp, "checkpoint_sha256")
        atomic_write_json(self.checkpoint_path, cp)
        # validate after write (reject ambiguous resume state)
        back = read_json_safe(self.checkpoint_path)
        if not sealed(back, "checkpoint_sha256") or back.get("event_head_hash") != cp["event_head_hash"]:
            raise StateError("checkpoint post-write validation failed")
        return cp

    def resume(self) -> dict[str, Any]:
        """Reload from checkpoint; refuse ambiguous state (split-brain)."""
        if not self.checkpoint_path.exists():
            raise StateError("no checkpoint to resume from")
        cp = read_json_safe(self.checkpoint_path)
        if not sealed(cp, "checkpoint_sha256"):
            raise StateError("checkpoint self-seal invalid")
        ok, why = self.log.verify_chain()
        if not ok:
            raise StateError(f"event log corrupt: {why}")
        head = self.log.head()
        head_hash = head["chain_hash"] if head else None
        if cp.get("event_head_hash") != head_hash:
            raise StateError(
                f"ambiguous resume: checkpoint head {cp.get('event_head_hash')} != log head {head_hash}")
        live_state = self.current_state()
        if cp.get("state") != live_state:
            raise StateError(f"checkpoint state {cp.get('state')} != log state {live_state}")
        return {"resumed_state": cp["state"], "event_cursor_seq": cp["event_cursor_seq"],
                "next_command": cp.get("next_command")}

    def status(self) -> dict[str, Any]:
        head = self.log.head()
        ok, why = self.log.verify_chain()
        return {
            "generation": self.generation,
            "state": self.current_state(),
            "event_count": (head["seq"] + 1) if head else 0,
            "event_head_hash": head["chain_hash"] if head else None,
            "chain_ok": ok,
            "chain_reasons": why,
            "checkpoint_present": self.checkpoint_path.exists(),
        }


def selftest() -> dict[str, Any]:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        c = Controller(Path(d) / "gen-1", generation="gen-1", controller_tree_sha256="a" * 64)
        c.boot()
        c.transition("AUDIT")
        c.transition("WAIT_OLD_RELEASE", {"reason": "campaign running"})
        c.transition("WAIT_OLD_RELEASE", {"heartbeat": 1})  # self-loop heartbeat
        if c.current_state() != "WAIT_OLD_RELEASE":
            raise StateError("state wrong")
        # illegal transition refused
        refused = False
        try:
            c.transition("LAUNCH")
        except StateError:
            refused = True
        if not refused:
            raise StateError("illegal transition not refused")
        # exact resume from a fresh handle
        c2 = Controller(Path(d) / "gen-1", generation="gen-1")
        r = c2.resume()
        if r["resumed_state"] != "WAIT_OLD_RELEASE":
            raise StateError(f"resume wrong: {r}")
        # tamper the checkpoint head -> resume refuses
        cp = json.loads((Path(d) / "gen-1" / "checkpoint.json").read_text())
        cp["event_head_hash"] = "b" * 64
        (Path(d) / "gen-1" / "checkpoint.json").write_text(json.dumps(cp))
        split = False
        try:
            Controller(Path(d) / "gen-1").resume()
        except StateError:
            split = True
        if not split:
            raise StateError("split-brain resume not refused")
    return {"ok": True, "enforced_transitions": True, "illegal_refused": True,
            "exact_resume": True, "split_brain_refused": True}


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=2, sort_keys=True))
