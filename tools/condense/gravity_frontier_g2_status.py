#!/usr/bin/env python3.12
"""Live truth for the Hawking Full Frontier Gate G2 (complete-layer) campaign - read-only.

This reads state, it never mutates it and never acquires-and-holds the controller lease. Liveness is
decided by the FLOCK, not by a PID field or a JSON file: a controller is RUNNING only if a live
process provably holds the singleton lease (com.hawking.frontier_g2). A stale PID, a dead process, or
a committed/historical checkpoint.json can therefore never report RUNNING.

Reported truth:
  * lease liveness (flock held by a live pid) + holder pid + aliveness;
  * heartbeat freshness (age < 3 * interval);
  * real progress, RECOUNTED from the durable per-row checkpoints (not trusted from the cursor);
  * derived state: RUNNING / DRAINING / PAUSED / COMPLETE / FAILED / NOT_STARTED;
  * the HIDDEN-STATE COSINE frontier: the highest complete-layer hidden-state cosine seen so far and
    its winning family, the per-tensor-class frontier, and the sealed G2_SELECTION.json winners;
  * PID, lease, program hash, generation binding, current row, completed/total, heartbeat, ETA.
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
from gravity_frontier_g2_controller import (  # noqa: E402
    DEFAULT_CAMPAIGN_ROOT, DEFAULT_GENERATION_PATH, DEFAULT_MANIFEST, DEFAULT_PROGRAM_PATH,
    HEARTBEAT_INTERVAL_SECONDS, LEASE_LABEL, G2Config, G2Controller,
)

STATUS_SCHEMA = "hawking.frontier_g2.status.v1"


def _pid_alive(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lease_flock_held(lease_path: Path) -> tuple[bool, int | None, str | None]:
    """(held, holder_pid, acquired_at). `held` is True only when a LIVE process holds the exclusive
    flock (we probe non-blocking and immediately release if we win)."""
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


def snapshot(config: G2Config) -> dict[str, Any]:
    controller = G2Controller(config)
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
    lease_live = bool(lease_held)

    hb: dict[str, Any] | None = None
    if controller.heartbeat_path.exists():
        try:
            hb = read_json_safe(controller.heartbeat_path)
        except EcoError:
            hb = None
    hb_age = _heartbeat_age_seconds(hb)
    hb_fresh = (hb_age is not None) and (hb_age < 3 * config.heartbeat_interval)

    state_counts = controller._collect_state() if program_ok else {
        "total_working_rows": 0, "completed_rows": 0, "failed_rows": 0, "pending_rows": 0,
        "completed_row_ids": [], "failed_row_ids": [], "avg_row_seconds": None,
        "eta_seconds": None, "best_by_hidden_cosine": None, "frontier_by_tensor_class": {}}

    any_progress = (state_counts["completed_rows"] + state_counts["failed_rows"]) > 0
    started = checkpoint_present or any_progress

    if lease_live:
        draining = bool(cursor and cursor.get("state_hint") == "draining")
        state = "DRAINING" if draining else "RUNNING"
    elif not started:
        state = "NOT_STARTED"
    elif state_counts["pending_rows"] == 0 and state_counts["failed_rows"] > 0 \
            and state_counts["completed_rows"] == 0:
        state = "FAILED"
    elif state_counts["pending_rows"] == 0:
        state = "COMPLETE"
    else:
        state = "PAUSED"

    selection: dict[str, Any] | None = None
    if controller.selection_path.exists():
        try:
            selection = read_json_safe(controller.selection_path)
        except EcoError:
            selection = None

    return {
        "schema": STATUS_SCHEMA,
        "state": state,
        "gate": "G2_complete_layer",
        "campaign": "frontier_g2",
        "campaign_root": str(controller.campaign_root),
        "program_path": str(controller.program_path),
        "program_sha256": controller.program_sha256,
        "program_loaded": program_ok,
        "program_error": program_error,
        "generation_binding": (cursor.get("generation_binding") if cursor
                               else controller._generation_binding()),
        "lease": {
            "path": str(controller.lease_path),
            "label": LEASE_LABEL,
            "live": lease_live,
            "holder_pid": holder_pid,
            "holder_alive": holder_alive,
            "acquired_at": acquired_at,
            "heavy_controller_count": 1 if lease_live else 0,
        },
        "controller_pid": cursor.get("controller_pid") if cursor else None,
        "process_start_time": cursor.get("process_start_time") if cursor else None,
        "parent_pid": os.getppid(),
        "checkpoint_present": checkpoint_present,
        "checkpoint_written_at": cursor.get("written_at") if cursor else None,
        "current_row": cursor.get("current_row") if cursor else None,
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
        "hidden_cosine_frontier": {
            "best_by_hidden_cosine": state_counts["best_by_hidden_cosine"],
            "frontier_by_tensor_class": state_counts["frontier_by_tensor_class"],
            "selection_present": selection is not None,
            "winners_by_tensor_class": (selection.get("winners_by_tensor_class")
                                        if selection else None),
        },
        # top-level aliases for downstream consumers (e.g. the ignition receipt): the current best
        # complete-layer candidate by hidden-state cosine, controls excluded.
        "frontier": state_counts["best_by_hidden_cosine"],
        "best_candidate": state_counts["best_by_hidden_cosine"],
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
    hc = snap["hidden_cosine_frontier"]
    gen = snap.get("generation_binding") or {}
    lines = [
        "FULL FRONTIER G2 :: GPT-OSS-120B complete-layer-0 quality",
        f"  state            {snap['state']}",
        f"  program_sha256   {(snap['program_sha256'] or 'n/a')[:16]}",
        f"  generation       {gen.get('generation')} sha16={gen.get('file_sha16')}",
        f"  lease            live={lease['live']} holder_pid={lease['holder_pid']} "
        f"alive={lease['holder_alive']} count={lease['heavy_controller_count']} "
        f"label={lease['label']}",
        f"  controller_pid   {snap['controller_pid']}  start={snap['process_start_time']}",
        f"  current_row      {snap['current_row']}",
        f"  progress         {prog['completed_rows']} sealed / {prog['failed_rows']} over-budget / "
        f"{prog['pending_rows']} pending  ({prog['total_rows']} total)",
        f"  last_checkpoint  {snap['checkpoint_written_at']}",
        f"  last_heartbeat   {hb['beat_at']}  age={hb['age_seconds']}s fresh={hb['fresh']}",
        f"  avg_row          {snap['avg_row_seconds']}s   ETA {_fmt_eta(snap['eta_seconds'])}",
    ]
    best = hc.get("best_by_hidden_cosine")
    if best:
        lines.append(f"  best_geometry    {best['family']} ({best['row_id']}) "
                     f"hidden_cos={best['layer_hidden_state_cosine']} "
                     f"combine_div={best.get('weighted_combine_divergence')} "
                     f"layer_bpw={best.get('complete_layer_bpw')}")
    for tc, entry in (hc.get("frontier_by_tensor_class") or {}).items():
        lines.append(f"  frontier[{tc}]  {entry['family']} "
                     f"hidden_cos={entry['layer_hidden_state_cosine']}")
    return "\n".join(lines)


def _config_from_args(args: argparse.Namespace) -> G2Config:
    root = Path(args.root) if args.root else DEFAULT_CAMPAIGN_ROOT
    program = Path(args.program) if args.program else DEFAULT_PROGRAM_PATH
    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) if args.only else None
    return G2Config(
        campaign_root=root, program_path=program,
        manifest_path=args.manifest or DEFAULT_MANIFEST,
        generation_path=DEFAULT_GENERATION_PATH,
        only_rows=only, heartbeat_interval=HEARTBEAT_INTERVAL_SECONDS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G2 complete-layer live status (read-only truth).")
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
