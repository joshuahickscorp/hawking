#!/usr/bin/env python3.12
"""Odyssey stage runner. Refuses to start training unless the fence says otherwise.

The fence is read, never written. Building the package and authorizing the run are
different acts by construction, so no amount of re-running the builder can start
training.

T0 baseline reproduction is implemented in tools/odyssey/t0_run.py and is
invoked from here only after authorization. Until the fence is true, this
process exits 1. It never flips the fence and never defaults it to true.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# This runner lives under the compact workspace hierarchy, but it is also a
# direct executable.  Walk to the repository marker rather than relying on the
# old shallow `odyssey/training/` location.
ROOT = next(
    (candidate for candidate in HERE.parents if (candidate / "Cargo.toml").is_file()),
    HERE.parents[5],
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.layout import odyssey_path

FENCE = odyssey_path("launch", "ODYSSEY_LAUNCH_AUTHORIZED")
STOP = odyssey_path("launch", "STOP")
PLAN = HERE / "ODYSSEY_TRAINING_PLAN.json"


def authorized() -> bool:
    return FENCE.is_file() and FENCE.read_text().strip().lower() == "true"


def _load_plan() -> dict:
    return json.loads(PLAN.read_text())


def _dispatch_stage(stage_id: str) -> int:
    """Dispatch a stage after authorization. Training stages are stubs that
    refuse heavy work without an explicit second confirmation file — but the
    fence is the first gate and is never written here.

    The fence is not the only gate. Every stage that would train must also pass
    the substrate capability register, which asks a different question: can this
    artifact generate at all? Math-Preserve is byte-perfect and cannot, so it is
    refused there regardless of what the fence says. T0 is exempt because
    reproducing a substrate is exactly how you learn it is broken.
    """
    sys.path.insert(0, str(ROOT))
    if stage_id == "T0":
        from tools.odyssey.t0_run import run as t0_run

        receipt = t0_run(max_shards=8, max_bytes=512 * 1024 * 1024)
        print(json.dumps({"stage": "T0", "status": receipt["status"], "summary": receipt["summary"]}))
        return 0 if receipt["status"] == "PASS" else 3

    # Second gate, independent of the fence: is the substrate fit to train on?
    from tools.odyssey._paths import EXPECTED_INDEX_SHA256
    from tools.odyssey.substrate_capability import SubstrateRefused, assert_trainable

    try:
        assert_trainable(EXPECTED_INDEX_SHA256)
    except SubstrateRefused as e:
        print(f"SUBSTRATE_REFUSED: {e}", file=sys.stderr)
        return 5

    # T1–T5: machinery exists as contracts; full training is not started here.
    from tools.odyssey.feasibility import estimate

    feas = estimate()
    stage = (feas.get("stages") or {}).get(stage_id, {})
    print(
        json.dumps(
            {
                "stage": stage_id,
                "status": "NOT_STARTED",
                "feasible_here": stage.get("feasible_here"),
                "reason": stage.get("reason"),
                "note": "Fence is true but this runner will not silently start 92 GB training.",
            },
            indent=2,
        )
    )
    return 4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default=None,
        help="Stage id (T0..T5). Required when authorized.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print training plan stages and exit (no fence required).",
    )
    args = parser.parse_args(argv)

    if args.list_stages:
        plan = _load_plan()
        for s in plan.get("stages") or []:
            print(f"{s['id']}\t{s['name']}\t{s['status']}")
        print(f"launch_authorized={plan.get('launch_authorized')}")
        return 0

    if STOP.is_file():
        print("ODYSSEY_STOPPED: emergency stop file present", file=sys.stderr)
        return 2
    if not authorized():
        print("ODYSSEY_LAUNCH_AUTHORIZED=false -- refusing to start.", file=sys.stderr)
        print(f"Authorize deliberately by writing 'true' to {FENCE}", file=sys.stderr)
        print(
            "Baseline reproduction without training: "
            "python3.12 tools/odyssey/t0_run.py",
            file=sys.stderr,
        )
        return 1
    if not args.stage:
        print("authorized; pass a stage id (T0..T5)", file=sys.stderr)
        return 1
    return _dispatch_stage(args.stage.upper())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
