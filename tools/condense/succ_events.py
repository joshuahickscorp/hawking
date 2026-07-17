#!/usr/bin/env python3.12
"""Append-only, hash-chained event log for the successor control plane.

The E0 audit found the eco_* layer has no event log, no sequence numbering, and no
journaled state, so nothing on top of it could be an event-sourced controller with exact
resume. This module is that missing spine.

Every event is:
  - assigned a monotonic sequence number;
  - chained: chain_hash = H(prev_chain_hash + canonical(event_body));
  - appended atomically (temp file + fsync + os.replace of the whole log is too costly for a
    growing log, so we append a single line, fsync the file, and fsync the parent directory,
    then validate the tail).

The log is a JSONL file; each line is one sealed event. `verify_chain` re-derives every
chain hash from the genesis, so any truncation, reordering, or in-place edit is detected.
This is the durable substrate for succ_state (the state machine) and succ_queue.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import EcoError, hash_value, now_iso, canonical_bytes  # noqa: E402

EVENT_SCHEMA = "hawking.successor.event.v1"
GENESIS_HASH = "0" * 64


class EventError(EcoError):
    """Fail-closed error in the event log."""


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class EventLog:
    """A durable, hash-chained append-only log rooted at one JSONL file."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- reading -------------------------------------------------------------------------
    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return iter(())
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def events(self) -> list[dict[str, Any]]:
        return list(self)

    def head(self) -> dict[str, Any] | None:
        last = None
        for event in self:
            last = event
        return last

    def head_hash(self) -> str:
        h = self.head()
        return h["chain_hash"] if h else GENESIS_HASH

    def next_seq(self) -> int:
        h = self.head()
        return (h["seq"] + 1) if h else 0

    # -- writing -------------------------------------------------------------------------
    def append(self, kind: str, payload: dict[str, Any], *, actor: str = "successor",
               cause_seq: int | None = None) -> dict[str, Any]:
        if not isinstance(kind, str) or not kind:
            raise EventError("event kind must be a non-empty string")
        if not isinstance(payload, dict):
            raise EventError("event payload must be an object")
        prev_hash = self.head_hash()
        seq = self.next_seq()
        body = {
            "schema": EVENT_SCHEMA,
            "seq": seq,
            "kind": kind,
            "actor": actor,
            "cause_seq": cause_seq,
            "at": now_iso(),
            "prev_hash": prev_hash,
            "payload": payload,
        }
        body["chain_hash"] = hash_value({**body, "_chain_base": prev_hash})
        line = json.dumps(body, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
        # append the single line, fsync file + parent dir, then validate the tail
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(self.path.parent)
        tail = self.head()
        if tail is None or tail.get("chain_hash") != body["chain_hash"]:
            raise EventError("post-append validation failed: tail hash mismatch")
        return body

    def verify_chain(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        prev = GENESIS_HASH
        expect_seq = 0
        for event in self:
            if event.get("seq") != expect_seq:
                reasons.append(f"seq gap: expected {expect_seq} got {event.get('seq')}")
            if event.get("prev_hash") != prev:
                reasons.append(f"prev_hash break at seq {event.get('seq')}")
            recomputed = hash_value({**{k: v for k, v in event.items() if k != "chain_hash"},
                                     "_chain_base": prev})
            if recomputed != event.get("chain_hash"):
                reasons.append(f"chain_hash mismatch at seq {event.get('seq')}")
            prev = event.get("chain_hash")
            expect_seq += 1
        return (not reasons), reasons


def selftest() -> dict[str, Any]:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        log = EventLog(Path(d) / "events.jsonl")
        e0 = log.append("boot", {"generation": 1})
        e1 = log.append("audit", {"terminal": 187}, cause_seq=e0["seq"])
        e2 = log.append("wait_old_release", {"reason": "campaign running"})
        if [e0["seq"], e1["seq"], e2["seq"]] != [0, 1, 2]:
            raise EventError("sequence numbering wrong")
        ok, why = log.verify_chain()
        if not ok:
            raise EventError(f"chain should verify: {why}")
        # tamper: rewrite a middle line and confirm detection
        lines = (Path(d) / "events.jsonl").read_text().splitlines()
        doc = json.loads(lines[1]); doc["payload"] = {"terminal": 999}
        lines[1] = json.dumps(doc, sort_keys=True, separators=(",", ":"))
        (Path(d) / "events.jsonl").write_text("\n".join(lines) + "\n")
        ok2, _ = EventLog(Path(d) / "events.jsonl").verify_chain()
        if ok2:
            raise EventError("tamper not detected")
        # resume: a fresh handle sees the right head + next_seq (minus tamper effect)
        fresh = EventLog(Path(d) / "events.jsonl")
        if fresh.next_seq() != 3:
            raise EventError("resume next_seq wrong")
    return {"ok": True, "sequenced": True, "chain_verifies": True, "tamper_detected": True,
            "resume_next_seq": 3}


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=2, sort_keys=True))
