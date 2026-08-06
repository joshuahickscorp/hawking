#!/usr/bin/env python3.12
"""Run the non-production framed-window lifecycle dry-run operator.

It consumes the fixture v2 protocol from stdin and writes a sealed
``FIXTURE_ONLY`` receipt to stdout.  This is intentionally not accepted by
the parent restream launcher and cannot be enabled alongside owner parent
authorization.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.lease import FixtureHeavyLease, FixtureLeaseError  # noqa: E402
from lab.operators.glm52_framed_window_operator import (  # noqa: E402
    FixtureFramedWindowOperator,
    FramedWindowOperatorError,
    LocalFixtureColdStorage,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-only", action="store_true", help="required acknowledgement of non-production semantics")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--cold-store", required=True)
    parser.add_argument("--contention-label", required=True, choices=("CLEAN", "CONTENDED", "INVALID"))
    parser.add_argument("--failure-stage", choices=("after_range", "after_pack", "after_handoff"))
    args = parser.parse_args(argv)
    if not args.fixture_only:
        parser.error("--fixture-only is required; this tool is not a parent-restream operator")
    if os.environ.get("HAWKING_PARENT_RESTREAM_AUTHORIZED") == "YES":
        print("REFUSED: fixture operator cannot run with parent-restream authorization", file=sys.stderr)
        return 78
    workspace = Path(args.workspace)
    lease = FixtureHeavyLease(workspace / "fixture-heavy.lease", campaign_id="glm52-framed-window-fixture")
    try:
        with lease.acquire(contention_label=args.contention_label):
            receipt = FixtureFramedWindowOperator(
                workspace_root=workspace,
                cold_storage=LocalFixtureColdStorage(Path(args.cold_store)),
                lease=lease,
                failure_stage=args.failure_stage,
            ).run(sys.stdin.buffer)
    except (FixtureLeaseError, FramedWindowOperatorError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 78
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
