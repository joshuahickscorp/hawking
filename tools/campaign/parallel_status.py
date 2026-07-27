#!/usr/bin/env python3.12
"""Parallel-continuation controller: derive lane state from live evidence.

`HAWKING_PARALLEL_LANE_OWNERSHIP.json` and `HAWKING_PARALLEL_DEPENDENCY_DAG.json` are
hand-authored architecture -- the controller's decisions about who owns what and what
genuinely depends on what.  This script does *not* edit them.  It reads them, then reads
what is actually true right now (Grok task statuses, git branches, which deliverables
exist on disk) and republishes the derived files.

    python3.12 tools/campaign/parallel_status.py          # republish
    python3.12 tools/campaign/parallel_status.py --json   # print only

A lane's declared status in the ownership file is an intention.  The status published
here is evidence: a lane is RUNNING only if its Grok process is alive, and COMPLETE only
if its declared outputs exist.  Where the two disagree, the evidence wins and the
disagreement is reported.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = Path.home() / ".claude-grok/tasks"

OWNERSHIP = ROOT / "HAWKING_PARALLEL_LANE_OWNERSHIP.json"
DAG = ROOT / "HAWKING_PARALLEL_DEPENDENCY_DAG.json"
STATUS_MD = ROOT / "HAWKING_PARALLEL_STATUS.md"
STATUS_JSON = ROOT / "HAWKING_PARALLEL_STATUS.json"
LEDGER = ROOT / "HAWKING_PARALLEL_LEDGER.jsonl"
NEXT_COMMAND = ROOT / "HAWKING_PARALLEL_NEXT_COMMAND.sh"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def grok_task_state(task_id: str | None) -> dict:
    """What the Grok runner and the OS say about one delegated task."""
    if not task_id:
        return {"present": False}
    d = TASKS / task_id
    if not d.is_dir():
        return {"present": False}
    status = (d / "status").read_text().strip() if (d / "status").exists() else "unknown"
    exit_code = None
    if (d / "exit_code").exists():
        exit_code = (d / "exit_code").read_text().strip()
    # `status` says running even when the process died with the session that spawned it,
    # which is how four slots were held by corpses.  Trust the process table.
    alive = False
    if status == "running":
        # The pattern must not begin with a dash or pgrep parses it as an option.
        alive = bool(subprocess.run(
            ["pgrep", "-f", f"{d}/task.md"],
            capture_output=True,
        ).stdout.strip())
    return {
        "present": True,
        "status": status,
        "exit_code": exit_code,
        "process_alive": alive,
        "stale_running": status == "running" and not alive,
        "report": str(d / "grok-report.md") if (d / "grok-report.md").exists() else None,
        "diff": str(d / "diff.patch") if (d / "diff.patch").exists() else None,
    }


def outputs_present(lane: dict) -> dict:
    """Which of a lane's declared deliverables actually exist in the repo root."""
    found, missing = [], []
    for out in lane.get("outputs", []) or []:
        # Only repo-root artifacts are checkable this way; code-change entries are prose.
        if out.endswith((".json", ".md", ".sh")) and "/" not in out:
            (found if (ROOT / out).exists() else missing).append(out)
    return {"found": found, "missing": missing}


def derive(lane: dict) -> dict:
    task = grok_task_state(lane.get("task_id"))
    outs = outputs_present(lane)
    declared = lane.get("status", "UNKNOWN")

    if declared in {"BLOCKED", "QUEUED"}:
        evidence_status = declared
    elif not task["present"]:
        evidence_status = declared
    elif task["stale_running"]:
        evidence_status = "ABANDONED_PROCESS_DIED"
    elif task["status"] == "running":
        evidence_status = "RUNNING"
    elif task["status"].startswith("abandoned"):
        evidence_status = "ABANDONED_PROCESS_DIED"
    elif task["exit_code"] not in (None, "0"):
        evidence_status = "FAILED"
    elif outs["missing"]:
        evidence_status = "FINISHED_OUTPUTS_MISSING"
    else:
        evidence_status = "FINISHED_AWAITING_REVIEW"

    return {
        "id": lane["id"],
        "name": lane["name"],
        "owner": lane.get("owner"),
        "declared_status": declared,
        "evidence_status": evidence_status,
        "disagrees": declared != evidence_status,
        "task": task,
        "outputs": outs,
        "dependencies": lane.get("dependencies", []),
        "promotion_gate": lane.get("promotion_gate"),
        "reviewed_by_controller": False,
    }


def main() -> int:
    own = json.loads(OWNERSHIP.read_text())
    dag = json.loads(DAG.read_text())
    lanes = [derive(l) for l in own["lanes"]]

    running = [l for l in lanes if l["evidence_status"] == "RUNNING"]
    ready = [l for l in lanes if l["evidence_status"].startswith("FINISHED")]
    blocked = [l for l in lanes if l["evidence_status"] == "BLOCKED"]
    queued = [l for l in lanes if l["evidence_status"] == "QUEUED"]

    # The controller's next action is always the cheapest thing that unblocks the most.
    if ready:
        nxt = f"review {ready[0]['id']} ({ready[0]['name']}) -- a report is a claim; re-derive before merging"
        cmd = f"cat {ready[0]['task']['report']}" if ready[0]["task"].get("report") else "# no report file"
    elif queued and len(running) < 6:
        nxt = f"launch {queued[0]['id']} ({queued[0]['name']}) -- a slot is free"
        cmd = "# see HAWKING_PARALLEL_LANE_OWNERSHIP.json for the lane's contract"
    elif running:
        nxt = f"{len(running)} lanes running at width {len(running)}/6; do not poll, wait on a milestone"
        cmd = "~/.claude-grok/bin/grok-run wait"
    else:
        nxt = "no lane running and none ready: the campaign is gated, not idle. Read hard_walls in the DAG."
        cmd = "python3.12 tools/campaign/parallel_status.py --json"

    doc = {
        "schema": "hawking.parallel.status.v1",
        "at": now(),
        "endpoint": "RAMANUJAN_SANDBOX_READY",
        "endpoint_reached": False,
        "fences": own["fences"],
        "sealed_inputs": own["sealed_inputs"],
        "width": {"running": len(running), "limit": 6},
        "lanes": lanes,
        "critical_path": dag["critical_path"],
        "external": own["external_lanes"],
        "next_action": nxt,
    }

    if "--json" in sys.argv:
        print(json.dumps(doc, indent=2))
        return 0

    STATUS_JSON.write_text(json.dumps(doc, indent=2) + "\n")

    def rows(ls: list[dict]) -> str:
        if not ls:
            return "- none\n"
        return "".join(
            f"- `{l['id']}` {l['name']} ({l['owner']})"
            + (f" -- outputs missing: {', '.join(l['outputs']['missing'])}" if l["outputs"]["missing"] else "")
            + (f" -- DECLARED {l['declared_status']} BUT EVIDENCE {l['evidence_status']}" if l["disagrees"] else "")
            + "\n"
            for l in ls
        )

    STATUS_MD.write_text(
        "# HAWKING PARALLEL CONTINUATION STATUS\n\n"
        "Generated from live evidence by `tools/campaign/parallel_status.py`. Do not hand-edit.\n"
        "Ownership and the DAG are hand-authored architecture and are read, never written, by this tool.\n\n"
        f"    at:       {doc['at']}\n"
        f"    endpoint: RAMANUJAN_SANDBOX_READY (not reached)\n"
        f"    width:    {len(running)}/6 lanes running\n"
        f"    fences:   ODYSSEY_LAUNCH_AUTHORIZED={own['fences']['ODYSSEY_LAUNCH_AUTHORIZED']} "
        f"RAMANUJAN_RESEARCH_AUTHORIZED={own['fences']['RAMANUJAN_RESEARCH_AUTHORIZED']} "
        f"HIDE_KERNEL_TURN={own['fences']['HIDE_KERNEL_TURN']}\n\n"
        "## Running\n\n" + rows(running) +
        "\n## Finished, awaiting controller review\n\n" + rows(ready) +
        "\n## Queued\n\n" + rows(queued) +
        "\n## Blocked (real data dependencies, see the DAG)\n\n" + rows(blocked) +
        "\n## Hard walls on the critical path\n\n"
        + "".join(f"- **{w['at']}**: {w['wall']}\n" for w in dag["critical_path"]["hard_walls"])
        + f"\n## Next action\n\n    {nxt}\n\n```bash\n{cmd}\n```\n"
    )

    NEXT_COMMAND.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + cmd + "\n")
    NEXT_COMMAND.chmod(0o755)

    # Append to the ledger only when the derived shape actually changes.
    shape = {l["id"]: l["evidence_status"] for l in lanes}
    last = None
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            if line.strip():
                last = json.loads(line).get("shape")
    if shape != last:
        with LEDGER.open("a") as fh:
            fh.write(json.dumps({"at": doc["at"], "shape": shape, "next": nxt}) + "\n")

    print(f"{len(running)} running, {len(ready)} awaiting review, {len(queued)} queued, {len(blocked)} blocked")
    print(f"next: {nxt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
