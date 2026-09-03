"""Rank the remaining work so a fresh operator does not have to guess.

Classification says what blocks a gate. It does not say which to do first, and
presenting nineteen software connections as peers leaves the hardest decision
unmade. This computes the dependency value of each remaining gate and ranks
across blocker classes.

The scalar SCHEDULES WORK. It never certifies truth. A high score means "do this
early", never "this is more real".
"""
from __future__ import annotations

from typing import Any

from tools.roadmap.blockers import classify

# Which lane the work consumes. Wrong-laning is how two heavy jobs land on one
# machine at once, so this is emitted per gate rather than assumed.
_LANES = {
    "PHYSICAL_HARDWARE_REQUIRED": "hardware",
    "EXTERNAL_ENVIRONMENT_REQUIRED": "operator",
    "DEFERRED_PROGRAM": "campaign",
    "EXPERIMENTATION_REQUIRED": "cpu",
    "SOFTWARE_CONNECTION_REMAINING": "cpu",
    "SOFTWARE_BUILD_REQUIRED": "cpu",
    "LONG_RUN_EVIDENCE_REQUIRED": "wall",
    "UNKNOWN_RESEARCH": "research",
}

# Deliberately coarse and LABELLED AS A GUESS. A precise-looking estimate nobody
# measured is worse than an admitted bucket, because it gets planned against.
_WALL_GUESS = {
    "SOFTWARE_CONNECTION_REMAINING": "under an hour (guess)",
    "SOFTWARE_BUILD_REQUIRED": "a day or more (guess)",
    "EXPERIMENTATION_REQUIRED": "one run, minutes to hours (guess)",
    "EXTERNAL_ENVIRONMENT_REQUIRED": "operator action, then minutes (guess)",
    "LONG_RUN_EVIDENCE_REQUIRED": "wall-bound (guess)",
    "PHYSICAL_HARDWARE_REQUIRED": "blocked until the device exists",
    "DEFERRED_PROGRAM": "a campaign, not a task",
    "UNKNOWN_RESEARCH": "unknown by definition",
}


def owner(gate: dict[str, Any]) -> str:
    """Who may do this. Checking ownership before implementing is a hard rule."""
    for ref in gate.get("code_refs") or []:
        rel = str((ref or {}).get("file") or "")
        if rel.startswith("hcli/"):
            return "hcli-campaign (do not implement here)"
        if rel.startswith(("tools/", "crates/", "research/lab/")):
            return "this-lane"
    return "unassigned"


def _reverse_deps(gates: dict[str, Any]) -> dict[str, list[str]]:
    rev: dict[str, list[str]] = {g: [] for g in gates}
    for gid, gate in gates.items():
        for dep in gate.get("dependencies") or []:
            rev.setdefault(dep, []).append(gid)
    return rev


def _transitive(gid: str, rev: dict[str, list[str]]) -> list[str]:
    seen: set[str] = set()
    stack = list(rev.get(gid) or [])
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(rev.get(cur) or [])
    return sorted(seen)


def rank(gates: dict[str, Any]) -> list[dict[str, Any]]:
    rev = _reverse_deps(gates)
    rows: list[dict[str, Any]] = []
    for gid, gate in gates.items():
        cls, missing = classify(gate)
        if not cls:
            continue
        direct = sorted(rev.get(gid) or [])
        trans = _transitive(gid, rev)
        who = owner(gate)
        actionable = cls in {
            "SOFTWARE_CONNECTION_REMAINING",
            "SOFTWARE_BUILD_REQUIRED",
            "EXPERIMENTATION_REQUIRED",
        } and who == "this-lane"
        # Leverage: descendants unlocked, with a nudge for work that is actually
        # doable now. Blocked work keeps its dependency value but does not sort
        # to the top of a list of things to start.
        score = len(trans) * 3 + len(direct) * 2 + (5 if actionable else 0)
        rows.append({
            "gate": gid,
            "blocker_class": cls,
            "missing": missing,
            "owner": who,
            "resource_lane": _LANES.get(cls, "cpu"),
            "estimated_wall": _WALL_GUESS.get(cls, "unestimated"),
            "depends_on": sorted(gate.get("dependencies") or []),
            "unlocks_direct": direct,
            "unlocks_transitive": trans,
            "critical_path": bool(trans),
            "multiplier": len(trans) >= 2,
            "actionable_now": actionable,
            "parallel_safe_with": "any gate in a different resource_lane",
            "stop_condition": (
                "a non-test call site reaches the named symbol AND a test fails "
                "when that connection is cut"
                if cls == "SOFTWARE_CONNECTION_REMAINING"
                else f"the blocker named above clears: {missing[:80]}"
            ),
            "reopen_if": (
                "the caller is removed, the verifier is deleted, or the gate's "
                "acceptance criterion changes"
            ),
            "verifier": (gate.get("tests") or [{}])[0].get("file") if gate.get("tests") else "must be written",
            "leverage_score": score,
        })
    rows.sort(key=lambda r: (-r["leverage_score"], r["gate"]))
    return rows


def hot_frontier(gates: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    """The top actionable work, ranked. Blocked work is excluded from the HOT list.

    A hot frontier that lists hardware-blocked gates wastes the reader's first
    ten lines on work nobody can start.
    """
    return [r for r in rank(gates) if r["actionable_now"]][:limit]
