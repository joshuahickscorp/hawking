"""Harness success metrics (S008 section 18).

The critical one is VERIFIED SCIENTIFIC PROGRESS PER SUPERVISOR INTERVENTION.
It must be computed from artifacts on disk, never from a narrative, because the
whole point is that it should be uncomfortable when autonomy is low. A metric
the supervisor can talk its way past measures nothing.

Definitions, chosen so they cannot be inflated:

  verified_progress   a receipt whose numbers came from EXECUTION and which
                      changed a frontier, a bound, or killed a hypothesis.
                      A receipt that only restates prior state does not count.
  supervisor_intervention
                      any event where the supervisor supplied the scientific
                      target: authored an experiment, fixed the harness so an
                      experiment could run, or corrected a conclusion.
                      Harness repairs COUNT -- autonomy that needs a human to
                      keep repairing its instrument is not autonomy.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / "receipts" / "future" / "O003_HCLI_LADDER.jsonl"


def ladder_stats(path: Path = LADDER) -> dict[str, Any]:
    if not path.exists():
        return {"rounds": 0}
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out: dict[str, Any] = {"rounds": len(rows)}
    for r in rows:
        k = str(r.get("outcome", "?")).split(":")[0]
        out[k] = out.get(k, 0) + 1
    ex = [r for r in rows if r.get("outcome") == "EXECUTED"]
    out["executed"] = len(ex)
    out["resident_originated_executions"] = len(ex)
    out["distinct_specs_proposed"] = len({r.get("spec_proposed") for r in rows})
    out["wall_s"] = round(sum(float(r.get("wall_s") or 0) for r in rows), 1)
    out["wall_s_per_execution"] = (round(out["wall_s"] / len(ex), 1) if ex else None)
    return out


def supervisor_interventions(since: str = "HEAD~20") -> dict[str, Any]:
    """Count from git: every commit is an intervention this session made."""
    try:
        log = subprocess.run(["git", "log", "--format=%s", f"{since}..HEAD"],
                             cwd=ROOT, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {"commits": None}
    subjects = [s for s in log.splitlines() if s.strip()]
    kinds = {"harness_repair": 0, "authored_experiment": 0, "correction": 0, "other": 0}
    for s in subjects:
        low = s.lower()
        if low.startswith("fix("):
            kinds["harness_repair"] += 1
        elif low.startswith(("feat(", "measure(")):
            kinds["authored_experiment"] += 1
        elif "correct" in low or "supersede" in low:
            kinds["correction"] += 1
        else:
            kinds["other"] += 1
    kinds["commits"] = len(subjects)
    return kinds


def autonomy(verified_progress: int, interventions: int) -> dict[str, Any]:
    ratio = (verified_progress / interventions) if interventions else None
    return {
        "verified_progress": verified_progress,
        "supervisor_interventions": interventions,
        "progress_per_intervention": (round(ratio, 3) if ratio is not None else None),
        "reading": (
            "below 1.0 means the supervisor is producing the science, not the harness"
            if ratio is not None and ratio < 1.0 else
            "at or above 1.0 the harness is carrying its own weight"
        ),
    }
