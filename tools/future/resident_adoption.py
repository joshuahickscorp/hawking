"""Resident adoption (S010 §1-3). The reaper STAYS; this decides what it spares.

MEASURED MOTIVE: the resident's 9.9 GB weight load is lazy inside the first
completion call and costs 211.3 s. Every HCLI round spawns a fresh pool, whose
startup reaps the previous round's now-orphaned resident, so every round pays it
again. Adoption removes that cost WITHOUT weakening the reaper: a resident that
cannot prove it is the right process, healthy, and unowned is still killed.

The decision is a PURE function of a record and an observation so every failure
mode S010 enumerates can be tested exhaustively, with no 9.9 GB process in the
loop. `decide()` returns ADOPT only when every check passes; everything else,
including anything unrecognised, returns REAP. Fail closed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ADOPT, REAP = "ADOPT", "REAP"
SCHEMA = "hawking.resident.ownership.v1"
HEARTBEAT_MAX_S = 90.0          # panic showed a 94 s stall; stale below that
RSS_CEILING_GB = 24.0           # a 9.9 GB body; 24 catches runaway growth


@dataclass(frozen=True)
class Record:
    """The durable ownership record. Every field exists to refuse an adoption."""
    schema: str
    pid: int
    proc_start: str             # defeats PID REUSE -- a recycled pid has a new start
    model_hash: str
    config_hash: str
    owner_generation: int
    owner_pid: int
    heartbeat_epoch: float
    rss_bytes: int
    endpoint: str
    ready: bool                 # false while half-initialised

    def as_dict(self) -> dict:
        return asdict(self)


def decide(rec: Any, obs: dict, now: float) -> tuple[str, str]:
    """(ADOPT|REAP, reason). Pure. Never adopts on a bare PID."""
    if not isinstance(rec, Record):
        return REAP, "ownership record missing or corrupt"
    if rec.schema != SCHEMA:
        return REAP, f"unknown schema {rec.schema!r}"
    if not obs.get("pid_alive"):
        return REAP, f"stale record: pid {rec.pid} is not alive"
    if obs.get("proc_start") != rec.proc_start:
        return REAP, (f"PID REUSE: pid {rec.pid} start {obs.get('proc_start')!r} "
                      f"!= recorded {rec.proc_start!r}")
    if obs.get("model_hash") != rec.model_hash:
        return REAP, "wrong model: hash mismatch"
    if obs.get("config_hash") != rec.config_hash:
        return REAP, "wrong config: hash mismatch"
    if not rec.ready:
        return REAP, "half-initialised: resident never reported ready"
    if not obs.get("responsive"):
        return REAP, "dead resident: process alive but not responding"
    age = now - rec.heartbeat_epoch
    if age > HEARTBEAT_MAX_S:
        return REAP, f"stale heartbeat: {age:.0f}s > {HEARTBEAT_MAX_S:.0f}s"
    if rec.rss_bytes > RSS_CEILING_GB * (1 << 30):
        return REAP, (f"over resource ceiling: {rec.rss_bytes / (1<<30):.1f} GB > "
                      f"{RSS_CEILING_GB} GB")
    if obs.get("guard_state") == "STOP":
        return REAP, "campaign guard STOP: refuse to adopt under resource pressure"
    if obs.get("owned_by_live_other"):
        return REAP, f"already owned by live pid {rec.owner_pid}"
    return ADOPT, f"verified pid {rec.pid} gen {rec.owner_generation}"


def claim(path: Path, gen: int) -> bool:
    """Atomically claim ownership. O_CREAT|O_EXCL: exactly one racer wins."""
    lock = path.with_suffix(".claim")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    try:
        os.write(fd, json.dumps({"owner_pid": os.getpid(), "generation": gen}).encode())
        return True
    finally:
        os.close(fd)


def release(path: Path) -> None:
    path.with_suffix(".claim").unlink(missing_ok=True)


def load(path: Path) -> Record | None:
    """Corrupt, truncated, or foreign records return None -> REAP."""
    try:
        d = json.loads(path.read_text())
        return Record(**{f: d[f] for f in Record.__dataclass_fields__})
    except Exception:
        return None
