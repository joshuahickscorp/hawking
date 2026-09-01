#!/usr/bin/env python3
"""G095: the bitcast levers under a real protected lease, and what promotion needs.

Every earlier measurement of these levers ran with ModelLake downloads live.
Twice the workers were SIGSTOPped and the supervisor respawned them, so twice
the window was declared contaminated and only the ratio was claimed.

This lease stops the SUPERVISOR FIRST - tools/odyssey/modellake_watch.py - so it
cannot respawn anything, then the workers, then measures, then resumes both in
reverse order. SIGSTOP only: nothing is killed and no download restarts.

    lease open    supervisor T, both workers T, loadavg 5.72
    lease close   loadavg 3.99, all three resumed R/S

    widen_f4      control 26.5410 ms      bitcast 22.0100 ms
                  saved    4.5311 ms      1.2059x
                  WALL    26.7447   ->    22.3347 ms
                  TPS     37.391    ->    44.773
                  32 tokens, 9 reps, TOKEN IDENTICAL, 580 dispatches, 0 fallbacks

The contaminated pairs put the combined saving at 4.5377 ms. The protected lease
says 4.5311. They agree to 0.15%, which is the strongest evidence yet for the
campaign's standing rule that ratios hold under load.

    python3 tools/future/protected_bitcast_absolute.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402

RECORDED_BY = "tools/future/protected_bitcast_absolute.py"
RECEIPT_NAME = "PROTECTED_BITCAST_ABSOLUTE.json"
CTRL_REL = "receipts/future/_G095_LEASE_CTRL_raw.json"
BITCAST_REL = "receipts/future/_G095_LEASE_BITCAST_raw.json"
BUDGET_REL = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"
LIVE_ARM = "widen_f4"

# What the unprotected pairs measured, for the load-invariance check.
UNPROTECTED_COMBINED_MS = 4.5377


class LeaseRefused(RuntimeError):
    """The pair is not matched, or the lease evidence is missing."""


def _arm(rel: str, arm: str) -> dict[str, Any]:
    p = REPO / rel
    if not p.is_file():
        raise LeaseRefused(f"{rel} is not on disk; run the lease first")
    d = json.loads(p.read_text())["decode"]
    if arm not in d:
        raise LeaseRefused(f"{rel} has no {arm} arm")
    return d[arm]


def _wall(arm: dict[str, Any]) -> float:
    r = sorted(arm["decode_wall_ns_reps"])
    return r[len(r) // 2] / 1e6 / len(arm["new_token_ids"])


def measured() -> dict[str, Any]:
    c, b = _arm(CTRL_REL, LIVE_ARM), _arm(BITCAST_REL, LIVE_ARM)
    if c["new_token_ids"] != b["new_token_ids"]:
        raise LeaseRefused("the arms produced different tokens; a regression, not a win")
    if c["theoretical_dispatches"] != b["theoretical_dispatches"]:
        raise LeaseRefused("dispatch count differs; not the same graph")
    if any(c["fallbacks_reps"]) or any(b["fallbacks_reps"]):
        raise LeaseRefused("a fallback fired; the graph is not the one claimed")
    for name, arm in (("control", c), ("bitcast", b)):
        if arm["dense_w_materialized"]:
            raise LeaseRefused(f"{name} materialised a dense W")
    cg, bg = c["gpu_ns_median"] / 1e6, b["gpu_ns_median"] / 1e6
    cw, bw = _wall(c), _wall(b)
    return {
        "arm": LIVE_ARM,
        "control_gpu_ms": round(cg, 4),
        "bitcast_gpu_ms": round(bg, 4),
        "gpu_ms_saved": round(cg - bg, 4),
        "speedup": round(cg / bg, 4),
        "control_wall_ms": round(cw, 4),
        "bitcast_wall_ms": round(bw, 4),
        "control_wall_tps": round(1000.0 / cw, 3),
        "bitcast_wall_tps": round(1000.0 / bw, 3),
        "token_identical": True,
        "n_tokens": len(c["new_token_ids"]),
        "reps": len(c["decode_wall_ns_reps"]),
        "dispatches": c["theoretical_dispatches"],
        "fallbacks": 0,
    }


def lease() -> dict[str, Any]:
    return {
        "authority": "S027 §28, and the user's own choice of SIGSTOP-measure-resume",
        "method": (
            "SIGSTOP the ModelLake SUPERVISOR first "
            "(tools/odyssey/modellake_watch.py), then the hf download workers, "
            "measure, then SIGCONT in reverse order"
        ),
        "why_the_supervisor_first": (
            "two earlier attempts stopped only the workers. The supervisor "
            "respawned them mid-window and both windows had to be declared "
            "contaminated. Stopping the parent is the difference between a "
            "lease and a pause."
        ),
        "nothing_was_killed": (
            "SIGSTOP and SIGCONT only. No download restarted from zero and no "
            "partial file was discarded."
        ),
        "loadavg_at_open": "5.72",
        "loadavg_at_close": "3.99",
        "states_at_open": "supervisor T, both workers T",
        "states_after_resume": "supervisor S, workers R and S",
        "evidence_class": "PROTECTED_ABSOLUTE",
    }


def load_invariance() -> dict[str, Any]:
    """The campaign's standing rule, now checked rather than asserted."""
    prot = measured()["gpu_ms_saved"]
    delta = abs(prot - UNPROTECTED_COMBINED_MS) / UNPROTECTED_COMBINED_MS
    return {
        "unprotected_combined_ms": UNPROTECTED_COMBINED_MS,
        "protected_ms": prot,
        "relative_difference": round(delta, 5),
        "agrees": delta < 0.02,
        "reading": (
            f"the contaminated pairs put the combined saving at "
            f"{UNPROTECTED_COMBINED_MS} ms and the protected lease says {prot}. "
            f"They agree to {delta:.2%}. This is the strongest evidence this "
            "campaign has for its own standing rule that RATIOS HOLD UNDER LOAD "
            "AND ABSOLUTES DO NOT - the rule has been used to license many "
            "claims and had not been checked against a protected window at this "
            "size before."
        ),
    }


def against_the_canonical_baseline() -> dict[str, Any]:
    """This profiler's control and the canonical budget receipt disagree."""
    m = measured()
    b = json.loads((REPO / BUDGET_REL).read_text())
    canon_wall = float(b["decode_wall_ms_per_token"])
    carried = canon_wall - m["gpu_ms_saved"]
    return {
        "canonical_wall_ms": canon_wall,
        "canonical_tps": round(1000.0 / canon_wall, 3),
        "this_lease_control_wall_ms": m["control_wall_ms"],
        "this_lease_control_tps": m["control_wall_tps"],
        "the_two_controls_differ_by_ms": round(canon_wall - m["control_wall_ms"], 4),
        "carried_onto_the_canonical_baseline_ms": round(carried, 4),
        "carried_onto_the_canonical_baseline_tps": round(1000.0 / carried, 3),
        "tps_range": sorted([round(1000.0 / carried, 3), m["bitcast_wall_tps"]]),
        "why_two_numbers": (
            "the canonical budget was produced by resident_reprofile.py and this "
            "lease by the organ profiler. They measure the same resident through "
            "different harnesses and their CONTROLS differ by "
            f"{round(canon_wall - m['control_wall_ms'], 3)} ms. Reporting one "
            "number would hide that, so the answer is a range and the harness "
            "disagreement is named as an open item."
        ),
        "open_item": (
            "the two harnesses should be reconciled before either absolute is "
            "promoted into PATH_TO_71 as the new baseline. Until then the "
            "levers are MEASURED and still not PROMOTED."
        ),
    }


def build() -> dict[str, Any]:
    m = measured()
    return {
        "obligation": "G095",
        "levers": ["HAWKING_AFFINE2_GEO=bitcast", "HAWKING_Q4_UNPACK=bitcast"],
        "default_is_unchanged": True,
        "lease": lease(),
        "measured": m,
        "load_invariance": load_invariance(),
        "against_the_canonical_baseline": against_the_canonical_baseline(),
        "still_short_of_60_by_ms": round(m["bitcast_wall_ms"] - 1000.0 / 60.0, 4),
        "checkpoints_crossed": ["40 TPS"],
        "claim_boundary": (
            "the window IS protected this time and the paired delta of "
            f"{m['gpu_ms_saved']} ms is a protected absolute. What remains "
            "unpromoted is the BASELINE: two harnesses report controls "
            "0.545 ms apart, and picking one silently would be choosing the "
            "flattering number."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(
                lock_held=True, lane="g094-lease", loadavg="{ 3.99 5.80 6.60 }"),
        ))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("measured", "load_invariance",
                       "against_the_canonical_baseline")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
