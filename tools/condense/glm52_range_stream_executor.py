#!/usr/bin/env python3.12
"""Installed-path entry point for the bounded GLM-5.2 range executor."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.glm52_common import atomic_json, read_sealed_json  # noqa: E402
from lab.operators.glm52_range_stream_executor import RangeExecutorError, execute  # noqa: E402
from lab.layout import LOCAL_ROOT, evidence_dir  # noqa: E402
from ramanujan.layout import BOUNDARY_ROOT, resolve_ramanujan_path  # noqa: E402
from ramanujan.restream_guard import (  # noqa: E402
    green_light_status_main,
    owner_authorization_claim_path,
    owner_authorization_ledger_dir,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--operator", default=os.environ.get("HAWKING_GLM52_WINDOW_OPERATOR", ""))
    parser.add_argument("--receipt-dir", default=str(LOCAL_ROOT / "state/glm52/range-window-receipts"))
    parser.add_argument("--terminal-receipt", default=str(evidence_dir("glm52") / "GLM52_RANGE_RESTREAM_TERMINAL.json"))
    parser.add_argument("--preflight", default=str(BOUNDARY_ROOT / "RAMANUJAN_GREEN_LIGHT_TRANSITION.json"))
    args = parser.parse_args(argv)
    preflight = Path(args.preflight)
    if not preflight.is_absolute():
        preflight = resolve_ramanujan_path(preflight, repo_root=ROOT)
    # The three-family scoreboard selected the official high-performance Xet
    # path.  Bind logging before hf_xet is first imported so runtime evidence
    # is both exact and machine-readable.
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    os.environ["HF_XET_LOG_DEST"] = "stderr"
    os.environ["HF_XET_LOG_FORMAT"] = "json"
    preflight_args = [
        "status", "--schedule", args.schedule, "--policy", args.policy,
        "--repo-root", str(ROOT), "--transition-receipt", str(preflight), "--claim-launch",
    ]
    if green_light_status_main(preflight_args) != 0:
        print("REFUSED: fresh signed green-light FINAL_PREFLIGHT is absent", file=sys.stderr)
        return 78
    authorization_path = Path(os.environ.get("HAWKING_OWNER_GREEN_LIGHT_AUTHORIZATION", ""))
    try:
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        launch_claim_path = owner_authorization_claim_path(
            owner_authorization_ledger_dir(ROOT), authorization
        )
    except Exception as exc:  # noqa: BLE001 - no body read on malformed capability routing.
        print(f"REFUSED: cannot resolve signed single-use launch capability: {exc}", file=sys.stderr)
        return 78
    try:
        result = execute(
            read_sealed_json(Path(args.schedule)), read_sealed_json(Path(args.policy)),
            operator_path=Path(args.operator), receipt_dir=Path(args.receipt_dir), workspace_root=ROOT,
            final_preflight=read_sealed_json(preflight),
            launch_claim_path=launch_claim_path,
            range_executor_path=Path(__file__).resolve(),
        )
    except RangeExecutorError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 78
    atomic_json(Path(args.terminal_receipt), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
