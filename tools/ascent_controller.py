#!/usr/bin/env python3
"""Detached ascent controller — keeps optimizing after the interactive chat ends.

The loop the ULTRAGOAL asks for:

    rank target -> form Grok contract -> launch isolated lane -> wait
      -> Tier-1 verify -> retain/reject -> update ledger -> reprofile -> repeat

Targets live in a JSON state file so the loop is resumable and auditable. Value
ranking is the doc's value function; no fake precision, it only has to order the
queue sensibly.

    python3 tools/ascent_controller.py status
    python3 tools/ascent_controller.py run [--max-lanes N]

Governors reused rather than rebuilt: machine_state.clean_box_ok (resource) and
tools/reclaim_safe.sh (disk).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "receipts" / "ascent-2026-08-16" / "ASCENT_STATE.json"
LANES = REPO / "workspace" / "ops" / "ascent-lanes"
GROK = Path.home() / ".claude-grok" / "bin" / "grok-run"
DISK_FLOOR_GIB = 15.0
DISK_WARN_GIB = 40.0

sys.path.insert(0, str(REPO / "tools"))


def machine_snapshot() -> dict:
    """Resource governor. Falls back to a permissive snapshot if agentos moved."""
    try:
        from agentos.machine_state import clean_box_ok, snapshot  # type: ignore

        snap = snapshot()
        ok, why = clean_box_ok(snap, min_free_gib=DISK_FLOOR_GIB)
        snap["clean_box_ok"], snap["clean_box_reason"] = ok, why
        return snap
    except Exception as exc:  # pragma: no cover - environment drift
        free = os.statvfs(REPO)
        gib = free.f_bavail * free.f_frsize / 1024**3
        return {
            "disk_free_gib": round(gib, 1),
            "clean_box_ok": gib >= DISK_FLOOR_GIB,
            "clean_box_reason": f"machine_state unavailable ({exc})",
        }


def value(t: dict) -> float:
    """VALUE = p_success * (recoverable_ns + density_gain + info + transfer) / cost.

    Ordering heuristic only. Deliberately prefers a large uncertain win over a
    small certain one, per the doctrine.
    """
    gain = (
        t.get("recoverable_ns_per_token", 0.0)
        + t.get("density_frontier_gain_ns_equiv", 0.0)
        + t.get("information_gain", 0.0)
        + t.get("transfer_value", 0.0)
    )
    return t.get("probability_of_success", 0.5) * gain / max(t.get("experiment_cost", 1.0), 0.1)


def load() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text())
    return {"schema": "hawking.ascent_controller.v1", "targets": [], "history": []}


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def reclaim_if_tight(snap: dict) -> None:
    if (snap.get("disk_free_gib") or 999) < DISK_WARN_GIB:
        script = REPO / "tools" / "reclaim_safe.sh"
        if script.is_file():
            subprocess.run(["bash", str(script)], cwd=REPO, check=False)


def tier1_verify(target: dict) -> tuple[bool, str]:
    """Seconds, dirty allowed, REJECT-ONLY. Never promotes anything.

    A pass here means "not obviously broken", never "correct". Tier 2 stays
    protected and manual.
    """
    cmd = target.get("tier1_command")
    if not cmd:
        return True, "no tier1 gate declared (not a promotion)"
    proc = subprocess.run(
        cmd, shell=True, cwd=REPO, capture_output=True, text=True, timeout=1800
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return False, f"tier1 exit {proc.returncode}"
    expect = target.get("tier1_expect")
    if expect and expect not in out:
        return False, f"tier1 missing expected marker {expect!r}"
    forbid = target.get("tier1_forbid")
    if forbid and forbid in out:
        return False, f"tier1 hit forbidden marker {forbid!r}"
    return True, "tier1 pass (reject-only; NOT a promotion)"


def _extract_task_id(text: str, slug: str) -> str | None:
    """Pull `<slug>-<YYYYMMDD>-<HHMMSS>` out of grok-run's launch output.

    grok-run only ever emits the id embedded in a path or a branch name
    ("...worktrees/<slug>-<stamp>", "grok/<slug>-<stamp>"), never as a bare
    leading token -- so match anywhere, not at a token boundary.
    """
    hit = re.search(rf"{re.escape(slug)}-\d{{8}}-\d{{6}}", text)
    return hit.group(0) if hit else None


def launch(target: dict) -> str | None:
    """Launch one isolated Grok lane. Returns task id, or None if it failed."""
    contract = Path(target["contract"])
    if not contract.is_file():
        print(f"  ! contract missing: {contract}")
        return None
    slug = target["id"]
    proc = subprocess.run(
        [
            str(GROK), "delegate", "--task", slug,
            "--contract", str(contract), "--repo", str(REPO),
            "--profile", target.get("profile", "gate"), "--background",
        ],
        capture_output=True, text=True,
    )
    return _extract_task_id(proc.stdout + proc.stderr, slug)


def wait_for(task_id: str, timeout: int = 14400) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        proc = subprocess.run(
            [str(GROK), "status"], capture_output=True, text=True
        )
        for line in proc.stdout.splitlines():
            if task_id in line:
                if line.startswith("done"):
                    return "done"
                if line.startswith(("failed", "error")):
                    return "failed"
        time.sleep(60)
    return "timeout"


def run(max_lanes: int) -> int:
    state = load()
    launched = 0
    while launched < max_lanes:
        snap = machine_snapshot()
        reclaim_if_tight(snap)
        if (snap.get("disk_free_gib") or 0) < DISK_FLOOR_GIB:
            print(f"HALT: disk {snap.get('disk_free_gib')} GiB below floor {DISK_FLOOR_GIB}")
            return 2

        pending = [t for t in state["targets"] if t.get("status") == "pending"]
        if not pending:
            print("no pending targets; queue is dry")
            return 0
        target = max(pending, key=value)

        print(f"-> {target['id']}  value={value(target):.1f}  {target.get('hypothesis','')}")
        task_id = launch(target)
        if not task_id:
            target["status"] = "launch_failed"
            save(state)
            continue

        target.update(status="running", task_id=task_id)
        save(state)

        outcome = wait_for(task_id)
        ok, why = (False, f"lane {outcome}") if outcome != "done" else tier1_verify(target)
        target.update(status="retained" if ok else "rejected", tier1=why)
        state["history"].append(
            {"id": target["id"], "task_id": task_id, "outcome": outcome, "tier1": why}
        )
        save(state)
        print(f"   {target['status']}: {why}")

        # Cleanup in the same pass that judged it. Nothing self-reaps.
        subprocess.run([str(GROK), "cleanup", "--id", task_id], check=False,
                       capture_output=True)
        launched += 1
    return 0


def status() -> int:
    state = load()
    snap = machine_snapshot()
    print(f"disk_free={snap.get('disk_free_gib')} GiB  clean_box={snap.get('clean_box_ok')}"
          f"  ({snap.get('clean_box_reason')})")
    if not state["targets"]:
        print("no targets registered")
    for t in sorted(state["targets"], key=value, reverse=True):
        print(f"  [{t.get('status','pending'):>13}] {t['id']:<28} value={value(t):8.1f}"
              f"  {t.get('hypothesis','')[:60]}")
    return 0


def _selfcheck() -> None:
    """Smallest thing that fails if the ranking or gate logic breaks."""
    big_risky = {"probability_of_success": 0.5, "recoverable_ns_per_token": 200e6,
                 "experiment_cost": 1.0}
    small_safe = {"probability_of_success": 0.95, "recoverable_ns_per_token": 2e6,
                  "experiment_cost": 1.0}
    assert value(big_risky) > value(small_safe), "must prefer large uncertain wins"
    assert value({"experiment_cost": 0}) == 0.0, "zero-cost target must not divide by zero"
    ok, why = tier1_verify({})
    assert ok and "not a promotion" in why, "empty gate must pass without promoting"
    ok, _ = tier1_verify({"tier1_command": "echo hello", "tier1_expect": "nope"})
    assert not ok, "missing expected marker must reject"
    ok, _ = tier1_verify({"tier1_command": "echo boom", "tier1_forbid": "boom"})
    assert not ok, "forbidden marker must reject"

    # Real grok-run launch output: the id appears only inside a path and a
    # branch name, never as a bare leading token. A token-prefix match misses it.
    real = ("grok-run: DRY RUN - would create worktree "
            "/Users/x/.claude-grok/worktrees/q80-pack-20260816-002534 "
            "on grok/q80-pack-20260816-002534 from ac69ffd3")
    assert _extract_task_id(real, "q80-pack") == "q80-pack-20260816-002534"
    assert _extract_task_id("no id here", "q80-pack") is None
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["status", "run", "selfcheck"])
    ap.add_argument("--max-lanes", type=int, default=4)
    args = ap.parse_args()
    if args.command == "status":
        sys.exit(status())
    if args.command == "selfcheck":
        _selfcheck()
        sys.exit(0)
    sys.exit(run(args.max_lanes))
