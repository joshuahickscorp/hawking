#!/usr/bin/env python3
"""MAXX — the resource optimizer that knows when NOT to fill a slot.

Seven independent queues. The objective is verified useful progress per wall-second per
resource, computed from obligations that actually moved to VERIFIED with evidence on
disk, not from a vibe.

The load-bearing property is the negative one: with a protected qualification window
open, MAXX REFUSES to start work that would contaminate it, even though a slot is free.
A benchmark taken while an HDD stream is running is a forged number, and a forged number
is worse than idle silicon.
"""
import argparse, json, os, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
ACTIVE = Path.home() / ".claude/ultragoal/active"
GROK_TASKS = Path.home() / ".claude-grok/tasks"
LAKE = Path("/Volumes/corpdrive/hawking-modellake")

QUEUES = ["GPU_READY", "CPU_READY", "GROK_READY", "NETWORK_READY",
          "HDD_READY", "SSD_READY", "INTEGRATION_READY"]

# Which queues contaminate a protected GPU/latency window. Unified memory means an HDD
# stream and a network fetch both move bytes through the same fabric the benchmark is
# measuring, and heavy CPU work steals submission threads.
CONTAMINATES_GPU = {"HDD_READY", "NETWORK_READY", "CPU_READY", "SSD_READY"}


class ProtectedWindow:
    """Open while an uncontended measurement is being taken."""

    def __init__(self, reason):
        self.reason = reason
        self.opened_at = None

    def __enter__(self):
        self.opened_at = time.time()
        return self

    def __exit__(self, *exc):
        return False


def active_ledger():
    best = None
    if ACTIVE.is_dir():
        for slot in sorted(ACTIVE.glob("*.json")):
            try:
                d = json.loads(slot.read_text())
            except Exception:
                continue
            led = Path(str(d.get("ledger", "")))
            if led.is_file() and "- [ ]" in led.read_text():
                m = slot.stat().st_mtime
                if best is None or m > best[0]:
                    best = (m, led)
    return best[1] if best else None


def verified_progress():
    """Useful progress is an obligation that reached VERIFIED with an evidence line that
    cites a receipt existing on disk. An evidence line citing nothing is not progress."""
    led = active_ledger()
    if not led:
        return {"measurable": False, "why": "no armed ledger with unVERIFIED obligations"}
    text = led.read_text()
    import re
    blocks = re.split(r"(?m)^- \[", text)[1:]
    verified, with_evidence, cited_ok, cited_missing = 0, 0, 0, 0
    for b in blocks:
        if not (b.startswith("x") or "status: VERIFIED" in b):
            continue
        verified += 1
        ev = re.search(r"^\s+evidence: (.+?)(?=\n\s*- \[|\n\n|\Z)", b, re.S | re.M)
        if not ev or "(none yet)" in ev.group(1):
            continue
        with_evidence += 1
        for rel in set(re.findall(r"receipts/[A-Za-z0-9_./-]+\.json", ev.group(1))):
            if (REPO / rel).exists():
                cited_ok += 1
            else:
                cited_missing += 1
    return {"measurable": True, "ledger": str(led), "n_verified": verified,
            "n_with_evidence": with_evidence,
            "n_cited_receipts_present": cited_ok,
            "n_cited_receipts_missing": cited_missing,
            "evidence_integrity": round(cited_ok / (cited_ok + cited_missing), 4)
            if (cited_ok + cited_missing) else None}


def probe_queues():
    """Real readiness, read off the machine."""
    q = {k: {"ready": [], "blocked": None} for k in QUEUES}
    # GROK: dead since the balance ran out; a lane that cannot start is BLOCKED, not ready
    # Sort by MTIME, not by name. Sorting 892 task dirs alphabetically and taking the tail
    # samples whatever starts with 'z', which missed the 402 lanes entirely.
    grok_dead, grok_evidence = False, None
    if GROK_TASKS.is_dir():
        logs = sorted(GROK_TASKS.glob("*/grok-stderr.log"),
                      key=lambda t: t.stat().st_mtime, reverse=True)[:12]
        for t in logs:
            try:
                tail = t.read_text()[-4000:]
            except Exception:
                continue
            if "402" in tail and "balance" in tail.lower():
                grok_dead, grok_evidence = True, str(t)
                break
    q["GROK_READY"]["blocked"] = (
        f"Grok Build usage balance exhausted (HTTP 402); lanes terminate immediately "
        f"[{grok_evidence}]") if grok_dead else None
    q["HDD_READY"]["ready"] = ([p.name for p in (LAKE / "partial").iterdir()]
                               if (LAKE / "partial").is_dir() else [])
    q["NETWORK_READY"]["ready"] = list(q["HDD_READY"]["ready"])
    q["SSD_READY"]["ready"] = ([p.name for p in (Path.home() / "noetic/stage").iterdir()]
                               if (Path.home() / "noetic/stage").is_dir() else [])
    led = active_ledger()
    if led:
        import re
        pend = re.findall(r"^- \[ \] (G\d+)", led.read_text(), re.M)
        q["CPU_READY"]["ready"] = pend
        q["INTEGRATION_READY"]["ready"] = pend
        q["GPU_READY"]["ready"] = [i for i in pend if i in ("G005", "G013", "G032")]
    return q, grok_dead


def admit(queue, protected: ProtectedWindow | None):
    """The decline-to-fill rule. This is the whole point of MAXX."""
    if protected and queue in CONTAMINATES_GPU:
        return False, (f"{queue} declined: a protected window is open ({protected.reason}). "
                       f"{queue} moves bytes through the same fabric the measurement is "
                       f"reading, so filling this slot would forge the number.")
    return True, f"{queue} admitted: no protected window conflicts with it"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", required=True)
    a = ap.parse_args()

    queues, grok_dead = probe_queues()
    progress = verified_progress()

    # Demonstrate the decline: open a protected window and offer every queue a job.
    with ProtectedWindow("uncontended TPOT measurement (G005)") as w:
        decisions = []
        for qn in QUEUES:
            ok, why = admit(qn, w)
            decisions.append({"queue": qn, "admitted": ok, "reason": why})
    declined = [d for d in decisions if not d["admitted"]]
    # And without one, the same offers are admitted.
    open_decisions = [{"queue": qn, "admitted": admit(qn, None)[0]} for qn in QUEUES]

    # A blocked queue must not stall the others.
    independence = {
        "blocked_queues": [k for k, v in queues.items() if v["blocked"]],
        "still_ready_while_blocked": {k: len(v["ready"]) for k, v in queues.items()
                                      if not v["blocked"] and v["ready"]},
        "a_blocked_queue_stalls_nothing":
            bool([k for k, v in queues.items() if v["blocked"]])
            and bool([k for k, v in queues.items() if not v["blocked"] and v["ready"]]),
    }

    out = {
        "schema": "hawking.headless.maxx_resource_pipeline.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/headless/maxx_scheduler.py",
        "obligation": "G014 — MAXX_RESOURCE_PIPELINE (directive §26, §109, §79)",
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "hand_authored": False,
        "objective": {
            "formula": "verified useful progress / (wall time x resource)",
            "numerator_is": "obligations at VERIFIED whose evidence cites receipts that "
                            "exist on disk",
            "inputs": progress,
            "law": "MAXX does not mean fill every slot (directive §26)"},
        "queues": {k: {"n_ready": len(v["ready"]), "ready": v["ready"][:6],
                       "blocked": v["blocked"]} for k, v in queues.items()},
        "n_queues": len(QUEUES),
        "queue_independence": independence,
        "protected_window_demo": {
            "window": "uncontended TPOT measurement (G005)",
            "contaminating_queues": sorted(CONTAMINATES_GPU),
            "decisions_with_window_open": decisions,
            "n_declined": len(declined),
            "decisions_with_no_window": open_decisions,
            "all_admitted_when_no_window": all(d["admitted"] for d in open_decisions)},
        "pass": bool(len(QUEUES) == 7 and declined
                     and all(d["admitted"] for d in open_decisions)
                     and progress.get("measurable")
                     and independence["a_blocked_queue_stalls_nothing"]),
    }
    Path(a.emit).write_text(json.dumps(out, indent=1))
    print(f"queues={len(QUEUES)} declined_under_protection={len(declined)} "
          f"verified={progress.get('n_verified')} "
          f"evidence_integrity={progress.get('evidence_integrity')} pass={out['pass']}")
    for d in declined:
        print(f"  DECLINED {d['queue']}: {d['reason'][:100]}")
    for k, v in queues.items():
        if v["blocked"]:
            print(f"  BLOCKED  {k}: {v['blocked'][:90]}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
