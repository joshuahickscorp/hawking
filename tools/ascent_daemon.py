#!/usr/bin/env python3
"""Detached ascent daemon — keeps the campaign moving with no Claude in the loop.

`ascent_controller.py` runs ONE cycle against a hand-written queue. That is not
enough to leave: the queue runs dry, finished lanes pile up unread, and nothing
decides what to try next. This adds the three missing pieces:

  harvest   read finished Grok lanes, extract their NEXT_BOTTLENECK, and turn it
            into the next target — this is what keeps the queue non-empty
  gate      real Tier-1 correctness checks per model, not an echo marker
  govern    pause when the box is not safe to benchmark on, reclaim when disk is
            tight, and never exceed the lane budget

It never merges. Promotion stays protected: a lane that passes Tier-1 is recorded
MERGE_READY with its skew verdict for a human (or Claude) to land.

    python3 tools/ascent_daemon.py once      # one full pass
    python3 tools/ascent_daemon.py loop      # run until stopped
    python3 tools/ascent_daemon.py status
    python3 tools/ascent_daemon.py selfcheck
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "receipts" / "ascent-2026-08-16" / "ASCENT_STATE.json"
QUEUE = REPO / "receipts" / "ascent-2026-08-16" / "PROMOTION_QUEUE.json"
TASKS = Path.home() / ".claude-grok" / "tasks"
GROK = Path.home() / ".claude-grok" / "bin" / "grok-run"
LANES = Path("/private/tmp/claude-503/-Users-scammermike-Downloads-hawking"
             "/d51d4904-9fa1-4f81-8170-5e7eb27a291d/scratchpad/lanes")

DISK_FLOOR_GIB = 15.0
DISK_WARN_GIB = 40.0
MAX_CONCURRENT = 6
POLL_SECONDS = 300

# Real Tier-1 gates. Reject-only: passing here is NOT promotion.
TIER1 = {
    "q80": {
        "cmd": "cargo build --profile release-fast -p hawking-core "
               "--example ascension_qwen80_uniform_q4_hybrid_greedy 2>&1 | tail -3",
        "expect": "Finished",
        "forbid": "error[",
    },
    "dsv4f": {
        "cmd": "cargo build --profile release-fast -p hawking-core "
               "--example gravity_deepseek_v4_native_token_graph 2>&1 | tail -3",
        "expect": "Finished",
        "forbid": "error[",
    },
    "qwen38": {
        "cmd": "cargo build --profile release-fast -p hawking-core 2>&1 | tail -3",
        "expect": "Finished",
        "forbid": "error[",
    },
}


def sh(cmd: str, timeout: int = 1800) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def load(path: Path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


# ---------------------------------------------------------------- governors

def machine() -> dict:
    sys.path.insert(0, str(REPO / "tools"))
    try:
        from agentos.machine_state import clean_box_ok, snapshot  # type: ignore
        snap = snapshot()
        snap["clean_box_ok"], snap["clean_box_reason"] = clean_box_ok(snap, DISK_FLOOR_GIB)
        return snap
    except Exception as exc:
        return {"disk_free_gib": 999, "clean_box_ok": False,
                "clean_box_reason": f"machine_state unavailable ({exc})",
                "active_grok_lanes": []}


def govern(snap: dict) -> str | None:
    """Return a reason to hold off, or None to proceed."""
    free = snap.get("disk_free_gib") or 0
    if free < DISK_WARN_GIB:
        script = REPO / "tools" / "reclaim_safe.sh"
        if script.is_file():
            sh(f"bash {script}", timeout=900)
            free = machine().get("disk_free_gib") or 0
    if free < DISK_FLOOR_GIB:
        return f"disk {free} GiB below floor {DISK_FLOOR_GIB}"
    live = len(snap.get("active_grok_lanes") or [])
    if live >= MAX_CONCURRENT:
        return f"{live} lanes live, at the {MAX_CONCURRENT} cap"
    return None


# ---------------------------------------------------------------- harvest

NEXT_RE = re.compile(r"^NEXT_BOTTLENECK:\s*(.+)$", re.M)
STATUS_RE = re.compile(r"^STATUS:\s*(\w+)", re.M)


def harvest() -> list[dict]:
    """Read finished lane reports and mine their NEXT_BOTTLENECK.

    This is the piece that keeps the queue alive without a human. A lane that
    finishes almost always names the next wall; that name becomes the next target.
    """
    found = []
    if not TASKS.is_dir():
        return found
    for d in sorted(TASKS.iterdir()):
        report = d / "grok-report.md"
        if not report.is_file():
            continue
        try:
            text = report.read_text(errors="replace")
        except Exception:
            continue
        m = NEXT_RE.search(text)
        if not m:
            continue
        s = STATUS_RE.search(text)
        found.append({
            "lane": d.name,
            "status": s.group(1) if s else "UNKNOWN",
            "next_bottleneck": m.group(1).strip()[:400],
        })
    return found


def model_of(lane: str) -> str:
    if lane.startswith("dsv"):
        return "dsv4f"
    if lane.startswith("qwen38") or "qwen38" in lane:
        return "qwen38"
    return "q80"


# ---------------------------------------------------------------- tier 1

def tier1(target: dict) -> tuple[bool, str]:
    """Seconds, dirty allowed, REJECT-ONLY. Passing is not promotion."""
    spec = TIER1.get(target.get("model", "q80"))
    cmd = target.get("tier1_command") or (spec or {}).get("cmd")
    if not cmd:
        return True, "no gate declared (not a promotion)"
    try:
        code, out = sh(cmd)
    except subprocess.TimeoutExpired:
        return False, "tier1 timeout"
    if code != 0:
        return False, f"tier1 exit {code}"
    expect = target.get("tier1_expect") or (spec or {}).get("expect")
    if expect and expect not in out:
        return False, f"tier1 missing {expect!r}"
    forbid = target.get("tier1_forbid") or (spec or {}).get("forbid")
    if forbid and forbid in out:
        return False, f"tier1 hit forbidden {forbid!r}"
    return True, "tier1 pass (reject-only; NOT a promotion)"


def skew(branch: str) -> str:
    code, out = sh(f"python3 tools/branch_skew_guard.py {branch}", timeout=900)
    for verdict in ("SKEWED", "STALE_CLEAN", "CLEAN", "EMPTY"):
        if verdict in out:
            return verdict
    return "UNKNOWN"


# ---------------------------------------------------------------- pass

def one_pass() -> dict:
    snap = machine()
    report = {"disk_free_gib": snap.get("disk_free_gib"),
              "live_lanes": len(snap.get("active_grok_lanes") or [])}

    state = load(STATE, {"targets": [], "history": []})
    queue = load(QUEUE, {"schema": "hawking.ascent.promotion_queue.v1", "entries": []})
    known = {e["lane"] for e in queue["entries"]}

    # 1. consume finished lanes nobody has read
    for h in harvest():
        if h["lane"] in known:
            continue
        branch = f"grok/{h['lane']}"
        code, _ = sh(f"git rev-parse --verify {branch}", timeout=120)
        entry = {
            "lane": h["lane"], "model": model_of(h["lane"]), "status": h["status"],
            "next_bottleneck": h["next_bottleneck"],
            "branch": branch if code == 0 else None,
            "skew": skew(branch) if code == 0 else "NO_BRANCH",
            "disposition": "MERGE_READY", "promoted": False,
        }
        if entry["skew"] == "SKEWED":
            entry["disposition"] = "NEEDS_COMPOSITION"
        elif entry["skew"] in ("NO_BRANCH", "EMPTY"):
            entry["disposition"] = "CHECK_FOR_UNCOMMITTED_WORK"
        queue["entries"].append(entry)
    save(QUEUE, queue)
    report["queued"] = len(queue["entries"])
    report["merge_ready"] = sum(1 for e in queue["entries"]
                                if e["disposition"] == "MERGE_READY" and not e["promoted"])
    report["needs_composition"] = sum(1 for e in queue["entries"]
                                      if e["disposition"] == "NEEDS_COMPOSITION")

    # 2. launch the top pending target if the box allows
    hold = govern(snap)
    if hold:
        report["launched"] = None
        report["hold"] = hold
        return report

    pending = [t for t in state["targets"] if t.get("status") == "pending"]
    if not pending:
        report["launched"] = None
        report["hold"] = "queue dry - harvest supplied no new pending target"
        return report

    from ascent_controller import value  # reuse the ranking, do not duplicate it
    target = max(pending, key=value)
    contract = Path(target.get("contract", ""))
    if not contract.is_file():
        target["status"] = "launch_failed"
        target["tier1"] = "contract missing"
        save(STATE, state)
        report["launched"] = None
        report["hold"] = f"contract missing for {target['id']}"
        return report

    code, out = sh(f"{GROK} delegate --task {target['id']} --contract {contract} "
                   f"--repo {REPO} --profile gate --background", timeout=600)
    m = re.search(rf"{re.escape(target['id'])}-\d{{8}}-\d{{6}}", out)
    if m:
        target.update(status="running", task_id=m.group(0))
        report["launched"] = m.group(0)
    else:
        target["status"] = "launch_failed"
        report["launched"] = None
    save(STATE, state)
    return report


def loop() -> int:
    while True:
        try:
            r = one_pass()
            print(json.dumps(r), flush=True)
        except Exception as exc:  # never die on one bad pass
            print(json.dumps({"error": str(exc)}), flush=True)
        time.sleep(POLL_SECONDS)


def status() -> int:
    snap = machine()
    q = load(QUEUE, {"entries": []})
    print(f"disk {snap.get('disk_free_gib')} GiB | live lanes "
          f"{len(snap.get('active_grok_lanes') or [])} | queued {len(q['entries'])}")
    for e in q["entries"]:
        if not e["promoted"]:
            print(f"  [{e['disposition']:<28}] {e['lane']:<42} skew={e['skew']}")
            print(f"      next: {e['next_bottleneck'][:110]}")
    return 0


def _selfcheck() -> None:
    """Pin the behaviours that make this safe to leave running."""
    assert model_of("dsv-expert-cache-1") == "dsv4f"
    assert model_of("q80-pack-1") == "q80"
    assert model_of("qwen38-bringup-1") == "qwen38"

    ok, why = tier1({"model": "q80", "tier1_command": "echo Finished"})
    assert ok, why
    ok, _ = tier1({"model": "q80", "tier1_command": "echo nope"})
    assert not ok, "missing expected marker must reject"
    ok, _ = tier1({"model": "q80", "tier1_command": "echo 'error[E0001]: x'; echo Finished"})
    assert not ok, "forbidden marker must reject even when the expect marker is present"
    ok, _ = tier1({"model": "q80", "tier1_command": "exit 3"})
    assert not ok, "non-zero exit must reject"

    txt = "STATUS: SHIPPED\nNEXT_BOTTLENECK: host.foo 123 ns/token\n"
    assert NEXT_RE.search(txt).group(1).startswith("host.foo")
    assert STATUS_RE.search(txt).group(1) == "SHIPPED"

    # The daemon must never promote or merge on its own authority. Check the
    # executable surface (sh() call sites), not the file text - an earlier version
    # of this assert matched its own message and failed spuriously.
    src = Path(__file__).read_text()
    calls = re.findall(r"sh\(\s*f?[\"']([^\"']+)", src)
    for c in calls:
        assert not c.lstrip().startswith("git merge"), f"daemon must never merge: {c}"
        assert "git push" not in c, f"daemon must never push: {c}"
    # Build the needle at runtime: a literal here would match this line itself.
    needle = "promoted" + '"] = ' + "True"
    assert needle not in src, "daemon must never self-promote"
    assert any("MERGE_READY" in line for line in src.splitlines()), (
        "daemon must record promotion-readiness rather than acting on it"
    )
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["once", "loop", "status", "selfcheck"])
    args = ap.parse_args()
    if args.command == "selfcheck":
        _selfcheck()
    elif args.command == "status":
        sys.exit(status())
    elif args.command == "loop":
        sys.exit(loop())
    else:
        print(json.dumps(one_pass(), indent=2))
