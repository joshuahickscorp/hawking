#!/usr/bin/env python3.12
"""Live truth for the Second Light PQ Gravity campaign - the `hawking status` equivalent.

This reads state, it never mutates it and never acquires-and-holds the controller lease.
Liveness is decided by the FLOCK, not by a PID field or a JSON file: a controller is
RUNNING only if a live process provably holds the singleton lease. A stale PID, a dead
process, or a committed/historical checkpoint.json can therefore never report RUNNING.

Reported truth:
  * lease liveness (flock held by a live pid) + the holder pid and its aliveness;
  * heartbeat freshness (age < 3 * interval);
  * real progress, RECOUNTED from the durable per-row checkpoints (not trusted from the
    cursor), so a stale cursor cannot inflate or deflate progress;
  * derived state: RUNNING / DRAINING / PAUSED / COMPLETE / FAILED / NOT_STARTED;
  * PID, lease, program hash, parent, current row, completed/total, last checkpoint time,
    last heartbeat, ETA (from measured per-row time), resource snapshot, best candidate.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import os
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import EcoError, read_json_safe  # noqa: E402
from second_light_controller import (  # noqa: E402
    DEFAULT_CAMPAIGN_ROOT, DEFAULT_PROGRAM_NAME, DEFAULT_MANIFEST, HEARTBEAT_INTERVAL_SECONDS,
    ControllerConfig, SecondLightController,
)

STATUS_SCHEMA = "hawking.second_light.status.v1"


def _pid_alive(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except OSError:
        return False
    return True


def _lease_flock_held(lease_path: Path) -> tuple[bool, int | None, str | None]:
    """Return (held, holder_pid, acquired_at). `held` is True only when a LIVE process
    holds the exclusive flock (we probe non-blocking and immediately release if we win)."""
    holder_pid: int | None = None
    acquired_at: str | None = None
    if lease_path.exists():
        try:
            first = lease_path.read_text(encoding="utf-8").splitlines()
            if first:
                stamp = json.loads(first[0])
                holder_pid = stamp.get("pid")
                acquired_at = stamp.get("acquired_at")
        except (OSError, json.JSONDecodeError):
            pass
    if not lease_path.exists():
        return False, holder_pid, acquired_at
    handle = None
    try:
        handle = lease_path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # We won the lock => no live controller holds it. Release at once.
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False, holder_pid, acquired_at
    except BlockingIOError:
        return True, holder_pid, acquired_at
    except OSError:
        return False, holder_pid, acquired_at
    finally:
        if handle is not None:
            handle.close()


def _heartbeat_age_seconds(hb: dict[str, Any] | None) -> float | None:
    if not hb:
        return None
    beat_at = hb.get("beat_at")
    if not isinstance(beat_at, str):
        return None
    try:
        when = _dt.datetime.fromisoformat(beat_at)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    return (_dt.datetime.now(_dt.timezone.utc) - when).total_seconds()


def snapshot(config: ControllerConfig) -> dict[str, Any]:
    controller = SecondLightController(config)
    program_ok = False
    program_error: str | None = None
    try:
        controller.load_program()
        program_ok = True
    except EcoError as exc:
        program_error = str(exc)

    checkpoint_present = controller.checkpoint_path.exists()
    cursor: dict[str, Any] | None = None
    if checkpoint_present:
        try:
            cursor = read_json_safe(controller.checkpoint_path)
        except EcoError:
            cursor = None

    lease_held, holder_pid, acquired_at = _lease_flock_held(controller.lease_path)
    holder_alive = _pid_alive(holder_pid)
    lease_live = bool(lease_held)  # flock is authoritative

    hb: dict[str, Any] | None = None
    if controller.heartbeat_path.exists():
        try:
            hb = read_json_safe(controller.heartbeat_path)
        except EcoError:
            hb = None
    hb_age = _heartbeat_age_seconds(hb)
    hb_fresh = (hb_age is not None) and (hb_age < 3 * config.heartbeat_interval)

    # Real progress, recounted from the durable per-row checkpoints.
    state_counts = controller._collect_state() if program_ok else {
        "total_working_rows": 0, "completed_rows": 0, "failed_rows": 0, "pending_rows": 0,
        "completed_row_ids": [], "failed_row_ids": [], "avg_row_seconds": None,
        "eta_seconds": None, "best_candidate": None}

    any_progress = (state_counts["completed_rows"] + state_counts["failed_rows"]) > 0
    started = checkpoint_present or any_progress

    # A LIVE lease is authoritative for RUNNING (a live process provably holds it), even in
    # the brief window before the first cursor write. Only when the lease is NOT live do we
    # distinguish NOT_STARTED / PAUSED / COMPLETE / FAILED, so a stale PID is never RUNNING.
    if lease_live:
        draining = bool(cursor and cursor.get("state_hint") == "draining")
        state = "DRAINING" if draining else "RUNNING"
    elif not started:
        state = "NOT_STARTED"
    elif state_counts["pending_rows"] == 0 and state_counts["failed_rows"] > 0:
        state = "FAILED"
    elif state_counts["pending_rows"] == 0:
        state = "COMPLETE"
    else:
        state = "PAUSED"

    checkpoint_written_at = cursor.get("written_at") if cursor else None
    current_row = cursor.get("current_row") if cursor else None
    process_start = cursor.get("process_start_time") if cursor else None

    return {
        "schema": STATUS_SCHEMA,
        "state": state,
        "campaign_root": str(controller.campaign_root),
        "program_path": str(controller.program_path),
        "program_sha256": controller.program_sha256,
        "program_loaded": program_ok,
        "program_error": program_error,
        "lease": {
            "path": str(controller.lease_path),
            "live": lease_live,
            "holder_pid": holder_pid,
            "holder_alive": holder_alive,
            "acquired_at": acquired_at,
            "heavy_controller_count": 1 if lease_live else 0,
        },
        "controller_pid": cursor.get("controller_pid") if cursor else None,
        "process_start_time": process_start,
        "parent_pid": os.getppid(),
        "checkpoint_present": checkpoint_present,
        "checkpoint_written_at": checkpoint_written_at,
        "current_row": current_row,
        "heartbeat": {
            "path": str(controller.heartbeat_path),
            "beat_at": (hb.get("beat_at") if hb else None),
            "age_seconds": (round(hb_age, 1) if hb_age is not None else None),
            "fresh": hb_fresh,
            "interval_seconds": config.heartbeat_interval,
        },
        "progress": {
            "completed_rows": state_counts["completed_rows"],
            "failed_rows": state_counts["failed_rows"],
            "pending_rows": state_counts["pending_rows"],
            "total_rows": state_counts["total_working_rows"],
            "completed_row_ids": state_counts["completed_row_ids"],
            "failed_row_ids": state_counts["failed_row_ids"],
        },
        "eta_seconds": state_counts["eta_seconds"],
        "avg_row_seconds": state_counts["avg_row_seconds"],
        "best_candidate": state_counts["best_candidate"],
        "resource_snapshot": (cursor.get("resource_snapshot") if cursor else None),
        "sampled_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def render_human(snap: dict[str, Any]) -> str:
    prog = snap["progress"]
    lease = snap["lease"]
    hb = snap["heartbeat"]
    lines = [
        "SECOND LIGHT :: GPT-OSS-120B PQ Gravity",
        f"  state            {snap['state']}",
        f"  program_sha256   {(snap['program_sha256'] or 'n/a')[:16]}",
        f"  lease            live={lease['live']} holder_pid={lease['holder_pid']} "
        f"alive={lease['holder_alive']} count={lease['heavy_controller_count']}",
        f"  controller_pid   {snap['controller_pid']}  start={snap['process_start_time']}",
        f"  current_row      {snap['current_row']}",
        f"  progress         {prog['completed_rows']} done / {prog['failed_rows']} failed / "
        f"{prog['pending_rows']} pending  ({prog['total_rows']} total)",
        f"  last_checkpoint  {snap['checkpoint_written_at']}",
        f"  last_heartbeat   {hb['beat_at']}  age={hb['age_seconds']}s fresh={hb['fresh']}",
        f"  avg_row          {snap['avg_row_seconds']}s   ETA {_fmt_eta(snap['eta_seconds'])}",
    ]
    best = snap.get("best_candidate")
    if best:
        lines.append(f"  best_candidate   {best['row_id']} rel_error={best['rel_error']} "
                     f"bpw={best.get('whole_artifact_bpw')}")
    res = snap.get("resource_snapshot")
    if res:
        swap = res.get("swap", {})
        lines.append(f"  resources        free_disk={res.get('free_disk_gb')}GB "
                     f"swap_used={swap.get('used_bytes')} rss={res.get('self_rss_bytes')}")
    return "\n".join(lines)


def _config_from_args(args: argparse.Namespace) -> ControllerConfig:
    root = Path(args.root) if args.root else DEFAULT_CAMPAIGN_ROOT
    program = Path(args.program) if args.program else (root / DEFAULT_PROGRAM_NAME)
    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) if args.only else None
    return ControllerConfig(
        campaign_root=root, program_path=program,
        manifest_path=args.manifest or DEFAULT_MANIFEST,
        only_rows=only, heartbeat_interval=HEARTBEAT_INTERVAL_SECONDS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Second Light live status (read-only truth).")
    ap.add_argument("--json", action="store_true", help="machine-readable JSON output")
    ap.add_argument("--only", default=None, help="restrict progress to these row_ids")
    ap.add_argument("--root", default=None)
    ap.add_argument("--program", default=None)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)
    snap = snapshot(_config_from_args(args))
    if args.json:
        print(json.dumps(snap, indent=2, sort_keys=True, default=str))
    else:
        print(render_human(snap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
