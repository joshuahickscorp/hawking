#!/usr/bin/env python3.12
"""Two counters that must never be averaged (G014, steer S004 sections 9 and 18).

Claude can discover fifteen things in an hour while HCLI selects nothing. A single
"campaign progress" number scores that as a good hour. It is not a good hour for the
obligation that actually matters, so this reports the two separately and refuses to
divide one by the other.

  CLAUDE-PROGRESS   artifacts Claude produced -- commits in the goal window that no
                    lane and no HCLI unit produced.
  HCLI-AUTONOMY     WorkUnits HCLI selected AND closed, where "closed" is
                    hcli.engine.is_accepted_work -- the repo's own predicate, imported
                    rather than reimplemented so it cannot drift from the verifier.

The contention log is the other half of G014: every time a lane is paused or reaped
for the authoritative experiment, the guard reading that FORCED it is recorded with
the decision. An entry without a live guard reading is refused, not counted -- a
decision whose measurement was never taken cannot be audited later, and the whole
point of the steer is that contention must be shown to have paid.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / ".hcli" / "receipts"
CONTENTION = ROOT / "receipts" / "future" / "CONTENTION_DECISIONS.jsonl"

# The goal window. Everything before this belongs to the superseded ledger.
GOAL_START = "2026-09-07 02:00:00"

sys.path.insert(0, str(ROOT))
from hcli.engine import is_accepted_work  # noqa: E402  the authority, not a copy


def _epoch(stamp: str) -> float:
    return time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))


def hcli_rounds(since: float) -> list[dict]:
    """One entry per HCLI receipt in the window, with selection and acceptance."""
    out = []
    for path in sorted(RECEIPTS.glob("*.json")):
        if path.stat().st_mtime < since:
            continue
        try:
            r = json.loads(path.read_text())
        except Exception:
            continue  # a truncated receipt is not a round; it is a broken file
        goal = r.get("goal") or ""
        ops = r.get("operations") or []
        # HCLI selected the target itself only when the prompt refused to name one.
        open_target = "CHOOSE YOUR OWN TARGET" in goal.upper()
        out.append({
            "receipt": path.name,
            "mtime": path.stat().st_mtime,
            "open_target": open_target,
            "n_operations": len(ops),
            "status": r.get("status"),
            "accepted": is_accepted_work(r.get("validation"), r.get("result_envelope")),
        })
    return out


def _lane_commits() -> set[str]:
    """Commits that landed a Grok lane -- searched for by task-id shape, not by trust."""
    ids = set()
    tasks = Path.home() / ".claude-grok" / "tasks"
    if tasks.is_dir():
        ids = {p.name for p in tasks.iterdir() if p.is_dir()}
    if not ids:
        return set()
    found = set()
    for tid in ids:
        cp = subprocess.run(["git", "log", "--since", GOAL_START, "--format=%H",
                             "--grep", tid], cwd=ROOT, capture_output=True, text=True)
        found.update(h for h in cp.stdout.split() if h)
    return found


def claude_artifacts(since_stamp: str) -> list[dict]:
    """Commits in the window that are not a landed lane."""
    cp = subprocess.run(["git", "log", "--since", since_stamp, "--format=%H%x1f%s"],
                        cwd=ROOT, capture_output=True, text=True)
    lanes = _lane_commits()
    rows = []
    for line in cp.stdout.splitlines():
        if "\x1f" not in line:
            continue
        h, subject = line.split("\x1f", 1)
        if h in lanes:
            continue
        rows.append({"sha": h[:9], "subject": subject})
    return rows


def contention_decisions() -> tuple[list[dict], list[dict]]:
    """(usable, refused). An entry with no live guard reading is refused."""
    usable, refused = [], []
    if not CONTENTION.exists():
        return usable, refused
    for line in CONTENTION.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            refused.append({"reason": "unparseable", "raw": line[:120]})
            continue
        guard = e.get("guard")
        if not isinstance(guard, dict) or "free_gb" not in guard:
            refused.append({"reason": "no measured guard reading", "entry": e})
            continue
        usable.append(e)
    return usable, refused


def record_contention(action: str, subject: str, why: str, expected_gb: float = 0.0) -> dict:
    """Append one decision WITH a guard reading taken now. The reading is the evidence."""
    sys.path.insert(0, str(ROOT / "tools" / "future"))
    import campaign_memory_guard as g
    snap = g.sample(expected_gb=expected_gb)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": action,          # paused | reaped | allowed | refused_launch
        "subject": subject,        # which lane or job
        "why": why,
        "guard": snap.as_dict(),
    }
    CONTENTION.parent.mkdir(parents=True, exist_ok=True)
    with CONTENTION.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def report() -> dict:
    since = _epoch(GOAL_START)
    rounds = hcli_rounds(since)
    claude = claude_artifacts(GOAL_START)
    usable, refused = contention_decisions()
    selected = [r for r in rounds if r["open_target"]]
    accepted = [r for r in rounds if r["accepted"]]
    sel_and_acc = [r for r in rounds if r["open_target"] and r["accepted"]]
    return {
        "window_start": GOAL_START,
        "claude_progress": {"artifacts": len(claude), "commits": claude[:40]},
        "hcli_autonomy": {
            "rounds": len(rounds),
            "self_selected_target": len(selected),
            "accepted_work": len(accepted),
            "selected_AND_accepted": len(sel_and_acc),
        },
        "contention": {"usable": usable, "refused": refused},
        # Deliberately absent: any ratio of the two. See the module docstring.
    }


def _selftest() -> None:
    # The acceptance predicate must be the repo's, and it must reject the two
    # shapes that fooled earlier counters: a bare ok=True, and a read_only unit.
    assert is_accepted_work({"ok": True}) is False, "bare ok=True must not count"
    assert is_accepted_work(
        {"ok": True, "kind": "read_only",
         "checks": [{"kind": "test", "exit_code": 0}]}) is False, "read_only must not count"
    assert is_accepted_work(
        {"ok": True, "checks": [{"kind": "test", "exit_code": 0}]}) is True, "real pass must count"

    # A contention entry without a guard reading is refused, never silently counted.
    import tempfile
    global CONTENTION
    keep = CONTENTION
    try:
        with tempfile.TemporaryDirectory() as d:
            CONTENTION = Path(d) / "c.jsonl"
            CONTENTION.write_text(
                json.dumps({"action": "paused", "subject": "x", "why": "y"}) + "\n"
                + json.dumps({"action": "paused", "subject": "x", "why": "y",
                              "guard": {"free_gb": 3.0}}) + "\n")
            usable, refused = contention_decisions()
            assert len(usable) == 1 and len(refused) == 1, (usable, refused)
            assert refused[0]["reason"] == "no measured guard reading"
    finally:
        CONTENTION = keep

    # The report must never carry a combined score.
    r = report()
    flat = json.dumps(r)
    for banned in ("ratio", "combined", "overall_progress"):
        assert banned not in flat, f"report leaked a combined metric: {banned}"
    print("selftest: PASS (acceptance predicate is hcli's own; guard-less contention "
          "entries refused; no combined metric)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)
    if "--record" in sys.argv:
        i = sys.argv.index("--record")
        action, subject, why = sys.argv[i + 1], sys.argv[i + 2], sys.argv[i + 3]
        exp = float(sys.argv[i + 4]) if len(sys.argv) > i + 4 else 0.0
        print(json.dumps(record_contention(action, subject, why, exp), indent=2))
        raise SystemExit(0)
    rep = report()
    if "--json" in sys.argv:
        print(json.dumps(rep, indent=2))
        raise SystemExit(0)
    c, h, k = rep["claude_progress"], rep["hcli_autonomy"], rep["contention"]
    print(f"  window since {rep['window_start']}")
    print(f"  CLAUDE-PROGRESS   artifacts={c['artifacts']}")
    print(f"  HCLI-AUTONOMY     rounds={h['rounds']}  self-selected={h['self_selected_target']}"
          f"  accepted={h['accepted_work']}  BOTH={h['selected_AND_accepted']}")
    print(f"  CONTENTION        usable={len(k['usable'])}  refused={len(k['refused'])}")
    for e in k["usable"]:
        g = e["guard"]
        print(f"    {e['ts']} {e['action']:<14} {e['subject']}  "
              f"free={g.get('free_gb')}GB state={g.get('state')}  why={e['why']}")
    for e in k["refused"]:
        print(f"    REFUSED: {e['reason']}")
    print("  no ratio is printed: the two counters are not commensurable")
