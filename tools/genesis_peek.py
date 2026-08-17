#!/usr/bin/env python3
"""One screen of truth about the running organism. Read-only, safe any time.

This exists so "how is it going" has a single answer that comes from process state
and receipts rather than from a status file. Status files on this box have reported
dead lanes as running and logged healthy ticks while five separate faults starved
the loop, so every liveness fact here is derived from pgrep or a file mtime.

    genesis_peek.py
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LINEAGE = REPO / "receipts" / "ascent-2026-08-16" / "GENESIS_LINEAGE_CURRENT.json"
DAEMON_LOG = REPO / "workspace" / "ops" / "ascent-daemon.log"
TASKS = Path.home() / ".claude-grok" / "tasks"


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except subprocess.SubprocessError:
        return ""


def main() -> int:
    print("=" * 66)
    print("HAWKING GENESIS")
    print("=" * 66)

    if LINEAGE.is_file():
        slots = json.loads(LINEAGE.read_text()).get("slots", {})
        for name in ("CURRENT", "CANDIDATE", "LAST_KNOWN_GOOD"):
            v = slots.get(name)
            if not v:
                print(f"  {name:16} (empty)")
            else:
                print(f"  {name:16} gen {v['generation']}  {v['representation_bpw']} BPW  "
                      f"{v['complete_token_ns']:,} ns  {v.get('tps', 0):.2f} TPS")
    else:
        print("  NOT SEATED - run tools/genesis_seat.py seat")

    # Liveness from process state, never from a status file.
    daemon = sh("pgrep -f 'ascent_daemon.py loop'")
    print(f"\n  loop              {'RUNNING pid ' + daemon.split()[0] if daemon else 'STOPPED'}")
    proposing = sh("pgrep -f genesis-propose")
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
