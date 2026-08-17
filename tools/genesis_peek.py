#!/usr/bin/env python3
"""One screen of truth about the running organism. Read-only, safe any time.

This exists so "how is it going" has a single answer that comes from process state
and receipts rather than from a status file. Status files on this box have reported
dead lanes as running and logged healthy ticks while five separate faults starved
the loop, so liveness comes from the process table and the resident health socket.

    genesis_peek.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parent.parent
LINEAGE = REPO / "receipts" / "ascent-2026-08-16" / "GENESIS_LINEAGE_CURRENT.json"
DAEMON_LOG = REPO / "workspace" / "ops" / "ascent-daemon.log"
RESIDENT_CLIENT = REPO / "tools" / "agentos" / "genesis_resident.py"
GPU_LOCK = Path("/tmp/hawking-gpu-lane.lock")
TASKS = Path.home() / ".claude-grok" / "tasks"


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except subprocess.SubprocessError:
        return ""


def process_snapshot() -> list[dict[str, Any]] | None:
    """Return one kernel process-table snapshot, excluding zombies from liveness."""
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,state=,command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        try:
            pid, ppid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        state = fields[2]
        rows.append(
            {
                "pid": pid,
                "ppid": ppid,
                "state": state,
                "command": fields[3],
                "alive": bool(state) and not state.startswith("Z"),
            }
        )
    return rows


def _command_tokens(row: dict[str, Any]) -> list[str]:
    # These commands contain no significant quoted payload; whitespace tokenizing
    # is deliberately conservative and cannot execute anything from the process list.
    return str(row.get("command") or "").split()


def _has_program(row: dict[str, Any], name: str, required_arg: str | None = None) -> bool:
    if not row.get("alive"):
        return False
    tokens = _command_tokens(row)
    if not tokens:
        return False
    invoked = 0 if Path(tokens[0]).name == name else None
    executable = Path(tokens[0]).name.lower()
    if invoked is None and "python" in executable:
        # The script is the first non-option argument to the interpreter. This
        # excludes rg/ps/editor command lines that merely mention its filename.
        for i, token in enumerate(tokens[1:], start=1):
            if token.startswith("-"):
                continue
            invoked = i if Path(token).name == name else None
            break
    if invoked is None:
        return False
    return required_arg is None or required_arg in tokens[invoked + 1 :]


def daemon_pids(rows: list[dict[str, Any]]) -> list[int]:
    return sorted(
        int(row["pid"])
        for row in rows
        if _has_program(row, "ascent_daemon.py", "loop")
    )


def resident_pids(rows: list[dict[str, Any]]) -> list[int]:
    return sorted(
        int(row["pid"])
        for row in rows
        if _has_program(row, "genesis-resident")
        or _has_program(row, "genesis_resident.py", "serve")
    )


def resident_health() -> dict[str, Any] | None:
    """Ask the existing body for health; this never starts or stops it."""
    try:
        spec = importlib.util.spec_from_file_location("_genesis_resident_peek", RESIDENT_CLIENT)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        info = module.health(module.default_socket(REPO), timeout=0.75)
    except Exception:  # status must survive a broken or mid-update client
        return None
    return info if isinstance(info, dict) else None


def pid_alive(pid: int) -> bool:
    """Kernel liveness for a lock owner, with zombies treated as dead."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except OSError:
        return False
    try:
        state = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(state) and not state.startswith("Z")


def _first_line(path: Path) -> str:
    try:
        return path.read_text().splitlines()[0].strip()
    except (OSError, IndexError, UnicodeError):
        return ""


def gpu_lock_state(
    lock: Path = GPU_LOCK,
    alive: Callable[[int], bool] = pid_alive,
) -> dict[str, Any]:
    """Describe the mkdir lock without repairing or otherwise mutating it."""
    if not lock.is_dir():
        return {"state": "FREE", "owner": None, "pid": None}
    owner = _first_line(lock / "owner") or "(unknown)"
    raw_pid = _first_line(lock / "pid")
    try:
        pid = int(raw_pid)
    except ValueError:
        return {"state": "INCOMPLETE", "owner": owner, "pid": None}
    return {
        "state": "HELD" if alive(pid) else "STALE",
        "owner": owner,
        "pid": pid,
    }


def lineage_reality(current: dict[str, Any] | None, body_live: bool) -> tuple[str, list[str]]:
    """Compare lineage's runtime flags with each other and with the live body."""
    if not current:
        return "UNKNOWN", ["CURRENT is empty"]
    live = current.get("live") if isinstance(current.get("live"), bool) else None
    launched = current.get("launched") if isinstance(current.get("launched"), bool) else None
    issues: list[str] = []
    if live is None or launched is None:
        issues.append("CURRENT live/launched flag missing")
    elif live != launched:
        issues.append("CURRENT live/launch flags disagree")
    if live is not None and live != body_live:
        issues.append("CURRENT.live does not match resident body")
    if launched is not None and launched != body_live:
        issues.append("CURRENT.launched does not match resident body")
    return ("MISMATCH" if issues else "MATCH"), issues


def _flag(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else "unknown"


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _gib(value: Any) -> str | None:
    try:
        return f"{int(value) / 2**30:.2f} GiB"
    except (TypeError, ValueError):
        return None


def main() -> int:
    print("=" * 66)
    print("HAWKING GENESIS")
    print("=" * 66)

    slots: dict[str, Any] = {}
    if LINEAGE.is_file():
        try:
            slots = json.loads(LINEAGE.read_text()).get("slots", {})
        except (OSError, json.JSONDecodeError, AttributeError):
            print("  LINEAGE UNREADABLE")
        for name in ("CURRENT", "CANDIDATE", "LAST_KNOWN_GOOD"):
            v = slots.get(name)
            if not v:
                print(f"  {name:16} (empty)")
            else:
                print(f"  {name:16} gen {v['generation']}  {v['representation_bpw']} BPW  "
                      f"{v['complete_token_ns']:,} ns  {v.get('tps', 0):.2f} TPS")
    else:
        print("  NOT SEATED - run tools/genesis_seat.py seat")

    # Runtime truth: one process snapshot plus the resident's health RPC.
    rows = process_snapshot()
    daemons = daemon_pids(rows or [])
    if rows is None:
        print("\n  loop              UNKNOWN (process table unavailable)")
    elif not daemons:
        print("\n  loop              STOPPED (no live daemon process)")
    elif len(daemons) == 1:
        print(f"\n  loop              RUNNING pid {daemons[0]}")
    else:
        print(f"\n  loop              RUNNING pids {','.join(map(str, daemons))} (DUPLICATE)")

    health = resident_health()
    body_live = bool(health and health.get("ok") and health.get("body_resident"))
    body_processes = resident_pids(rows or [])
    health_pid = _safe_int(health.get("pid")) if health else None
    if body_live:
        print(f"  resident body     HEALTHY pid {health_pid if health_pid is not None else 'unknown'}")
        load_bits = []
        if health.get("load_count") is not None:
            load_bits.append(f"count {health['load_count']}")
        if health.get("load_ns") is not None:
            try:
                load_bits.append(f"last {int(health['load_ns']) / 1e9:.3f}s")
            except (TypeError, ValueError):
                pass
        weight_gib = _gib(health.get("resident_weight_bytes"))
        if weight_gib:
            load_bits.append(f"weights {weight_gib}")
        if health.get("generation") is not None:
            load_bits.append(f"generation {health['generation']}")
        print(f"  resident load     {', '.join(load_bits) if load_bits else 'loaded (details unavailable)'}")
        roles = health.get("session_roles")
        if isinstance(roles, list):
            print(f"  resident sessions {len(roles)}: {', '.join(map(str, roles))}")
        else:
            print("  resident sessions attached; roles not reported")
        if health.get("reload_error"):
            print(f"  resident reload   ERROR {health['reload_error']}")
        if rows is not None and health_pid not in body_processes:
            print("  WARNING           healthy PID not recognized as a resident command")
    elif body_processes:
        print(f"  resident body     UNHEALTHY pid(s) {','.join(map(str, body_processes))}")
        print("  resident load     health socket unavailable (loading or wedged)")
        print("  resident sessions unavailable")
    else:
        print("  resident body     DOWN (no healthy body or live resident process)")
        print("  resident load     not loaded")
        print("  resident sessions unavailable")

    lock = gpu_lock_state()
    if lock["state"] == "FREE":
        print("  GPU lock          FREE")
    elif lock["pid"] is None:
        print(f"  GPU lock          {lock['state']} owner {lock['owner']} (no valid pid)")
    else:
        print(f"  GPU lock          {lock['state']} owner {lock['owner']} pid {lock['pid']}")

    current = slots.get("CURRENT") if isinstance(slots, dict) else None
    reality, issues = lineage_reality(current, body_live)
    if current:
        print(
            f"  lineage runtime   {reality} live={_flag(current.get('live'))} "
            f"launched={_flag(current.get('launched'))} resident={_flag(body_live)}"
        )
        if issues:
            print(f"  WARNING           {'; '.join(issues)}")

    proposing = bool(rows and any("genesis-propose" in str(r.get("command")) for r in rows))
    print(f"  genesis proposing {'yes' if proposing else 'no'}")

    if DAEMON_LOG.is_file():
        age = time.time() - DAEMON_LOG.stat().st_mtime
        print(f"  last tick         {age/60:.1f} min ago")
        lines = [l for l in DAEMON_LOG.read_text().splitlines() if l.startswith("{")]
        if lines:
            t = json.loads(lines[-1])
            print(f"  lanes live        {t.get('our_live_lanes')} / cap {t.get('memory_lane_cap')}")
            print(f"  queue             {t.get('queued')} queued, {t.get('merge_ready')} merge-ready")
            print(f"  disk free         {t.get('disk_free_gib')} GiB")
            if t.get("hold"):
                print(f"  HELD              {t['hold']}")
            if t.get("launched"):
                print(f"  launched          {t['launched']}")

    live = sh("~/.claude-grok/bin/grok-run status 2>/dev/null | grep -c running")
    print(f"  grok lanes        {live or 0} running")

    mem = sh("python3 " + str(REPO / "tools" / "genesis_capacity.py") + " plan 2>/dev/null")
    if mem:
        try:
            d = json.loads(mem)
            print(f"  memory            {d['used_gib']} / {d['target_used_gib']} GiB target "
                  f"({d['used_gib']/d['total_gib']*100:.0f}% of {d['total_gib']})")
        except ValueError:
            pass
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
