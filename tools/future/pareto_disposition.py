"""G033: the frontier is multi-axis, and disposition is a rule not a mood.

S012 §55-57: choosing a specimen on dense capability alone is wrong, because the
best parent BEFORE Gravity may not be the best after it. A slightly weaker parent
admitting a far smaller, faster executable may dominate for Hawking.

And §53-54: a dominated specimen is sealed and RELEASED without expensive
executable synthesis -- but it still owes its findings. Losing is not the same as
contributing nothing.
"""
from __future__ import annotations

from typing import Any, Iterable

# Axes the frontier must carry. Dense capability alone is explicitly not enough.
AXES = ("capability_passes", "complete_ebpw", "decode_tok_s", "active_bytes_per_token",
        "representation_complexity", "transfer_value")


def dominates(a: dict, b: dict) -> bool:
    """a dominates b: no worse on every axis, strictly better on at least one.

    Lower is better for ebpw, bytes/token and complexity; higher for throughput
    and transfer value; a capability failure cannot dominate a pass.
    """
    if a.get("capability_passes") and not b.get("capability_passes"):
        better_any = True
    elif b.get("capability_passes") and not a.get("capability_passes"):
        return False
    else:
        better_any = False
    lower = ("complete_ebpw", "active_bytes_per_token", "representation_complexity")
    higher = ("decode_tok_s", "transfer_value")
    for k in lower:
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            continue
        if x > y:
            return False
        if x < y:
            better_any = True
    for k in higher:
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            continue
        if x < y:
            return False
        if x > y:
            better_any = True
    return better_any


def frontier(points: Iterable[dict]) -> list[dict]:
    pts = list(points)
    return [p for p in pts if not any(dominates(q, p) for q in pts if q is not p)]


def disposition(specimen: dict) -> dict:
    """Seal-and-release, or earn executable work. §53: only finalists earn synthesis."""
    on_frontier = bool(specimen.get("on_frontier"))
    contributions = list(specimen.get("contributions") or [])
    # A specimen may only be released once it has paid what it owes: at least one
    # finding of lasting value. Releasing a specimen that taught nothing means the
    # wall spent on it bought nothing either.
    owes = not contributions
    if on_frontier:
        return {"action": "RETAIN", "earns_executable_synthesis": True,
                "reason": "on the Pareto frontier", "owes_findings": owes}
    return {"action": "SEAL_AND_RELEASE", "earns_executable_synthesis": False,
            "reason": "dominated; expensive executable synthesis is not earned",
            "owes_findings": owes,
            "retained_knowledge": contributions}
