#!/usr/bin/env python3.12
"""Live truth for the Hawking Full Frontier Gate G3 (cross-layer-transfer) campaign - read-only.

This reads state, it never mutates it and never acquires-and-holds the controller lease. Liveness is
decided by the FLOCK, not by a PID field or a JSON file: a controller is RUNNING only if a live
process provably holds the singleton lease (com.hawking.frontier_g3). A stale PID, a dead process, or
a committed/historical checkpoint.json can therefore never report RUNNING.

Reported truth:
  * lease liveness (flock held by a live pid) + holder pid + aliveness;
  * heartbeat freshness (age < 3 * interval);
  * real progress, RECOUNTED from the durable per-row checkpoints (not trusted from the cursor);
  * derived state: RUNNING / DRAINING / PAUSED / COMPLETE / FAILED / NOT_STARTED;
  * active_generation = M + the bound base_execution_provider map;
  * the current probe layer + row, the per-(class, layer) hidden-cosine frontier, and the sealed
    G3_TRANSFER.json cross-layer verdict (does the layer-0 winner transfer to mid/late?);
  * PID, lease, program hash, generation binding, completed/total, heartbeat, ETA.
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
from gravity_frontier_g3_controller import (  # noqa: E402
    BASE_EXECUTION_PROVIDER, DEFAULT_CAMPAIGN_ROOT, DEFAULT_GENERATION_PATH, DEFAULT_MANIFEST,
    DEFAULT_PROGRAM_PATH, DEFAULT_PROVIDER_MAP_PATH, EXECUTION_GENERATION, HEARTBEAT_INTERVAL_SECONDS,
    LEASE_LABEL, G3Config, G3Controller,
)

STATUS_SCHEMA = "hawking.frontier_g3.status.v1"


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


def snapshot(config: G3Config) -> dict[str, Any]:
    controller = G3Controller(config)
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
        "completed_row_ids": [], "failed_row_ids": [], "layers_touched": [], "avg_row_seconds": None,
        "eta_seconds": None, "best_by_hidden_cosine": None, "frontier_by_class_layer": {}}

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

    transfer: dict[str, Any] | None = None
    if controller.transfer_path.exists():
        try:
            transfer = read_json_safe(controller.transfer_path)
        except EcoError:
            transfer = None

    return {
        "schema": STATUS_SCHEMA,
        "state": state,
        "gate": "G3_cross_layer_transfer",
        "campaign": "frontier_g3",
        "campaign_root": str(controller.campaign_root),
        "program_path": str(controller.program_path),
        "program_sha256": controller.program_sha256,
        "program_loaded": program_ok,
        "program_error": program_error,
        "active_generation": EXECUTION_GENERATION,
        "base_execution_provider": BASE_EXECUTION_PROVIDER,
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
        "current_layer": cursor.get("current_layer") if cursor else None,
        "layers_touched": state_counts["layers_touched"],
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
        "transfer_frontier": {
            "best_by_hidden_cosine": state_counts["best_by_hidden_cosine"],
            "frontier_by_class_layer": state_counts["frontier_by_class_layer"],
            "transfer_present": transfer is not None,
            "transfer_by_tensor_class": (transfer.get("transfer_by_tensor_class")
                                         if transfer else None),
        },
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
    tf = snap["transfer_frontier"]
    gen = snap.get("generation_binding") or {}
    lines = [
        "FULL FRONTIER G3 :: GPT-OSS-120B cross-layer transfer (early 0 / mid 18 / late 35)",
        f"  state            {snap['state']}",
        f"  active_gen       {snap['active_generation']}  provider={(snap['base_execution_provider'] or {}).get('name')}",
        f"  program_sha256   {(snap['program_sha256'] or 'n/a')[:16]}",
        f"  generation       {gen.get('generation')} closure_sha16={(gen.get('closure_sha256') or 'n/a')[:16]}",
        f"  lease            live={lease['live']} holder_pid={lease['holder_pid']} "
        f"alive={lease['holder_alive']} count={lease['heavy_controller_count']} "
        f"label={lease['label']}",
        f"  controller_pid   {snap['controller_pid']}  start={snap['process_start_time']}",
        f"  current          row={snap['current_row']} layer={snap['current_layer']}",
        f"  layers_touched   {snap['layers_touched']}",
        f"  progress         {prog['completed_rows']} sealed / {prog['failed_rows']} over-budget / "
        f"{prog['pending_rows']} pending  ({prog['total_rows']} total)",
        f"  last_checkpoint  {snap['checkpoint_written_at']}",
        f"  last_heartbeat   {hb['beat_at']}  age={hb['age_seconds']}s fresh={hb['fresh']}",
        f"  avg_row          {snap['avg_row_seconds']}s   ETA {_fmt_eta(snap['eta_seconds'])}",
    ]
    best = tf.get("best_by_hidden_cosine")
    if best:
        lines.append(f"  best_geometry    {best['family']} ({best['row_id']}) "
                     f"L{best.get('layer')}({best.get('layer_role')}) "
                     f"hidden_cos={best['layer_hidden_state_cosine']} "
                     f"layer_bpw={best.get('complete_layer_bpw')}")
    for key, entry in (tf.get("frontier_by_class_layer") or {}).items():
        lines.append(f"  frontier[{key}]  {entry['family']} "
                     f"hidden_cos={entry['layer_hidden_state_cosine']}")
    for tc, entry in (tf.get("transfer_by_tensor_class") or {}).items():
        lines.append(f"  transfer[{tc}]  layer0={entry.get('layer0_winner_family')} "
                     f"per_layer={entry.get('winner_family_per_layer')} "
                     f"->mid={entry.get('transfers_to_mid')} late={entry.get('transfers_to_late')} "
                     f"full={entry.get('fully_transfers')}")
    return "\n".join(lines)


def _config_from_args(args: argparse.Namespace) -> G3Config:
    root = Path(args.root) if args.root else DEFAULT_CAMPAIGN_ROOT
    program = Path(args.program) if args.program else DEFAULT_PROGRAM_PATH
    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) if args.only else None
    return G3Config(
        campaign_root=root, program_path=program,
        manifest_path=args.manifest or DEFAULT_MANIFEST,
        generation_path=DEFAULT_GENERATION_PATH, provider_map_path=DEFAULT_PROVIDER_MAP_PATH,
        only_rows=only, heartbeat_interval=HEARTBEAT_INTERVAL_SECONDS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="G3 cross-layer-transfer live status (read-only truth).")
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
