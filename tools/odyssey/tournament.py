#!/usr/bin/env python3.12
"""Checkpoint tournament: newest does not automatically win.

Selection compares a challenger against the incumbent across the math profile
score and the support-halo dimensions. Ties go to the incumbent.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

SCHEMA = "hawking.odyssey.checkpoint_tournament.v1"

# Support-halo dimensions from SUPPORT_HALO_SCORING_RULES + uncertainty calibration.
HALO_DIMENSIONS = [
    "technical_language",
    "general_reasoning",
    "coding",
    "retrieval",
    "tools",
    "long_context",
    "self_correction",
    "uncertainty_calibration",
]


@dataclass(frozen=True)
class Scorecard:
    checkpoint_id: str
    math_profile: float
    halo: dict[str, float]

    def halo_aggregate(self) -> float:
        vals = [float(self.halo[d]) for d in HALO_DIMENSIONS if d in self.halo]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)


def compare(
    incumbent: Scorecard,
    challenger: Scorecard,
    *,
    math_eps: float = 1e-12,
    halo_eps: float = 1e-12,
) -> dict[str, Any]:
    """Return a tournament decision.

    Challenger wins only if it is strictly better on math_profile AND
    strictly better on halo aggregate (no dimension may regress beyond a
    hard floor relative to incumbent). Tie or any non-strict improvement
    → incumbent retains.
    """
    math_delta = challenger.math_profile - incumbent.math_profile
    halo_i = incumbent.halo_aggregate()
    halo_c = challenger.halo_aggregate()
    halo_delta = halo_c - halo_i

    regressions: list[dict[str, Any]] = []
    for dim in HALO_DIMENSIONS:
        if dim not in incumbent.halo or dim not in challenger.halo:
            continue
        delta = challenger.halo[dim] - incumbent.halo[dim]
        if delta < -halo_eps:
            regressions.append({"dimension": dim, "delta": delta})

    math_strictly_better = math_delta > math_eps
    halo_strictly_better = halo_delta > halo_eps and not regressions

    if math_strictly_better and halo_strictly_better:
        winner = challenger.checkpoint_id
        reason = "challenger strictly better on math profile and support-halo aggregate with no halo dimension regression"
    else:
        winner = incumbent.checkpoint_id
        if abs(math_delta) <= math_eps and abs(halo_delta) <= halo_eps and not regressions:
            reason = "tie → incumbent retains"
        elif regressions:
            reason = "halo dimension regression → incumbent retains"
        elif not math_strictly_better:
            reason = "math profile not strictly improved → incumbent retains"
        else:
            reason = "support-halo not strictly improved → incumbent retains"

    return {
        "schema": SCHEMA,
        "incumbent": incumbent.checkpoint_id,
        "challenger": challenger.checkpoint_id,
        "winner": winner,
        "reason": reason,
        "math": {
            "incumbent": incumbent.math_profile,
            "challenger": challenger.math_profile,
            "delta": math_delta,
            "strictly_better": math_strictly_better,
        },
        "support_halo": {
            "dimensions": HALO_DIMENSIONS,
            "incumbent_aggregate": halo_i,
            "challenger_aggregate": halo_c,
            "delta": halo_delta,
            "strictly_better": halo_strictly_better,
            "regressions": regressions,
            "per_dimension": {
                d: {
                    "incumbent": incumbent.halo.get(d),
                    "challenger": challenger.halo.get(d),
                }
                for d in HALO_DIMENSIONS
            },
        },
        "rule": "newest does not automatically win; tie goes to incumbent",
    }


def scorecard_from_dict(d: dict[str, Any]) -> Scorecard:
    return Scorecard(
        checkpoint_id=str(d["checkpoint_id"]),
        math_profile=float(d["math_profile"]),
        halo={k: float(v) for k, v in (d.get("halo") or {}).items()},
    )


def main(argv: list[str] | None = None) -> int:
    # Demo self-check: equal scores → incumbent.
    halo = {d: 0.5 for d in HALO_DIMENSIONS}
    inc = Scorecard("ckpt_0", 0.80, halo)
    ch = Scorecard("ckpt_1", 0.80, dict(halo))
    r = compare(inc, ch)
    assert r["winner"] == "ckpt_0", r
    # Strict win both.
    ch2 = Scorecard("ckpt_2", 0.85, {d: 0.6 for d in HALO_DIMENSIONS})
    r2 = compare(inc, ch2)
    assert r2["winner"] == "ckpt_2", r2
    # Math better but one halo regression → incumbent.
    bad_halo = {d: 0.6 for d in HALO_DIMENSIONS}
    bad_halo["coding"] = 0.1
    ch3 = Scorecard("ckpt_3", 0.99, bad_halo)
    r3 = compare(inc, ch3)
    assert r3["winner"] == "ckpt_0", r3
    print(json.dumps({"self_check": "PASS", "tie": r, "win": r2, "regression": r3}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
