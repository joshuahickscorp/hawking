"""How much wall does the resident lose to tool calls that fail?

Every failed deterministic tool call costs model time, and model time is the
scarcest thing in an Odyssey. This reads the HCLI mission event logs -- READ
ONLY; that campaign owns them -- and answers three questions the wall model
cannot answer without it:

    what fraction of tool calls fail
    what the failures cost, relative to the model wall they interrupt
    how many are RECOVERABLE (the file exists under a name the model nearly
    guessed) versus genuinely absent

The distinction matters more than the rate. A call that fails because the file
does not exist is information. A call that fails because the model asked for
`hcli/mission/state.json` when the file is at another path, and the error says
only "FileNotFoundError" plus a type signature, is pure loss: the model has the
right basename and no way to find the right directory except by guessing again.

Found by running this: ONE path was requested 59 times across 55 distinct goals
and the file was really there, its name mangled from
HCLI_SELF_IMPROVEMENT_DIRECTIVE.md to HCLI_SELF_IMPROVEMENT_DIRECTifact.md.
Half of all FileNotFoundError friction was one broken rename.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
EVENT_GLOB = ".hcli/**/events.jsonl"

_PATH_IN_ERROR = re.compile(r"((?:/[\w./\-]+))")
_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".worktrees", ".hcli"}


def _events() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(REPO.glob(EVENT_GLOB)):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _basenames_on_disk() -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs
                   if d not in _SKIP_DIRS and not d.startswith(".venv")]
        for name in files:
            index.setdefault(name, []).append(os.path.join(root, name))
    return index


def census() -> dict[str, Any]:
    """The friction census. Every number is counted, none is assumed."""
    events = _events()
    finished = [e for e in events if e.get("type") == "tool_call_finished"]
    model = [e for e in events if e.get("type") == "model_call_finished"]

    failed = [e for e in finished if (e.get("data") or {}).get("ok") is False]
    tool_wall = sum(float((e.get("data") or {}).get("elapsed_s") or 0) for e in finished)
    model_wall = sum(float((e.get("data") or {}).get("elapsed_s") or 0) for e in model)

    by_tool = Counter(str((e.get("data") or {}).get("tool")) for e in failed)
    by_class = Counter(
        str((e.get("data") or {}).get("error") or "").split(":")[0].strip() or "(none)"
        for e in failed
    )

    missing: Counter[str] = Counter()
    goals: dict[str, set[str]] = {}
    for e in failed:
        d = e.get("data") or {}
        err = str(d.get("error") or "")
        if "FileNotFoundError" not in err:
            continue
        m = _PATH_IN_ERROR.search(err)
        if not m:
            continue
        missing[m.group(1)] += 1
        goals.setdefault(m.group(1), set()).add(str(d.get("goal_id")))

    index = _basenames_on_disk()
    names = list(index)
    rows = []
    recoverable = absent = present_now = 0
    for path, count in missing.most_common():
        base = os.path.basename(path)
        if os.path.exists(path):
            kind, hint = "PRESENT_NOW", path
            present_now += count
        elif base in index:
            kind, hint = "WRONG_DIRECTORY", index[base][0]
            recoverable += count
        else:
            near = difflib.get_close_matches(base, names, n=1, cutoff=0.75)
            if near:
                kind, hint = "NEAR_NAME", index[near[0]][0]
                recoverable += count
            else:
                kind, hint = "ABSENT", ""
                absent += count
        rows.append({
            "path": path, "requested": count, "distinct_goals": len(goals[path]),
            "kind": kind, "hint": hint,
        })

    total = len(finished)
    n_failed = len(failed)
    return {
        "schema": "hawking.future.tool_friction.v1",
        "evidence_tier": "STATIC",
        "source": "HCLI mission event logs (read only; that campaign owns them)",
        "tool_calls": total,
        "failed": n_failed,
        "failure_rate": round(n_failed / total, 4) if total else None,
        "tool_execution_wall_s": round(tool_wall, 1),
        "model_wall_s": round(model_wall, 1),
        "model_calls": len(model),
        "mean_model_call_s": round(model_wall / len(model), 1) if model else None,
        "tool_execution_share_of_wall": (
            round(tool_wall / (tool_wall + model_wall), 6) if (tool_wall + model_wall) else None
        ),
        "failures_by_tool": dict(by_tool.most_common()),
        "failures_by_error_class": dict(by_class.most_common()),
        "missing_paths": rows,
        "file_not_found": {
            "total": present_now + recoverable + absent,
            "present_now": present_now,
            "recoverable_wrong_directory_or_near_name": recoverable,
            "genuinely_absent": absent,
        },
        "claim_boundary": (
            "STATIC census of committed event logs. Wall figures are the logs' own "
            "elapsed_s sums, not a new measurement. No Odyssey was launched."
        ),
    }


def wasted_model_wall_s(doc: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the failures plausibly cost, as a RANGE. A point estimate would lie.

    The two honest bounds are far apart and both are defensible:

      pro-rata   failed calls as a share of all tool calls, applied to the model
                 wall. Assumes a failed call costs what an average call costs.
      per-turn   one whole model call per failure. An upper bound: there are
                 more tool calls than model calls, so failures cannot each own a
                 separate turn.
    """
    doc = doc or census()
    total, failed = doc["tool_calls"], doc["failed"]
    model_wall, model_calls = doc["model_wall_s"], doc["model_calls"]
    if not (total and model_calls):
        return {"unavailable": "no tool or model calls in the logs"}
    mean = model_wall / model_calls
    return {
        "pro_rata_s": round(model_wall * failed / total, 1),
        "per_turn_upper_bound_s": round(failed * mean, 1),
        "calls_per_model_turn": round(total / model_calls, 2),
        "note": "a point estimate between these would be invented; both bounds are stated",
    }


def build() -> Path:
    from tools.future._common import write_receipt
    doc = census()
    doc["wasted_model_wall"] = wasted_model_wall_s(doc)
    return write_receipt("TOOL_FRICTION.json", doc,
                         recorded_by="tools/future/tool_friction.py")


def main() -> int:
    doc = census()
    print(f"tool calls {doc['tool_calls']}, failed {doc['failed']} "
          f"({doc['failure_rate']:.1%})")
    print(f"tool execution {doc['tool_execution_wall_s']}s vs model "
          f"{doc['model_wall_s']}s -- execution is "
          f"{doc['tool_execution_share_of_wall']:.2%} of the wall")
    fnf = doc["file_not_found"]
    print(f"FileNotFoundError {fnf['total']}: present_now {fnf['present_now']}, "
          f"recoverable {fnf['recoverable_wrong_directory_or_near_name']}, "
          f"absent {fnf['genuinely_absent']}")
    waste = wasted_model_wall_s(doc)
    print(f"model wall lost to failures: {waste['pro_rata_s']}s pro-rata, "
          f"{waste['per_turn_upper_bound_s']}s upper bound")
    print("worst paths:")
    for row in doc["missing_paths"][:5]:
        print(f"  {row['requested']:3d}x over {row['distinct_goals']:3d} goals  "
              f"[{row['kind']}] {os.path.basename(row['path'])}")
    print(f"wrote {build()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
