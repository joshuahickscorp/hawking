#!/usr/bin/env python3.12
"""CLI entry point for the DeepSeek-V4 build-full resource supervisor.

DSV4F is retained as science but its weight body is deliberately deleted.  A
stale launchd job used to reconstruct those weights on every crash.  This
entrypoint enforces the seal before importing or starting the builder so stale
launch queues cannot bypass the active-model policy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEAL_RECEIPT = ROOT / "receipts" / "ascent-2026-08-16" / "DSV4F_SEALED_SCIENCE_WEIGHTS_DELETED.json"
SEALED_STATUS = "SEALED_SCIENCE_RETAINED_WEIGHTS_DELETED"


def sealed_by_policy(path: Path = SEAL_RECEIPT) -> bool:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return receipt.get("status") == SEALED_STATUS


def main(argv: list[str] | None = None) -> int:
    if sealed_by_policy():
        print(
            json.dumps(
                {
                    "status": "SEALED_MODEL_REFUSED",
                    "model": "dsv4f",
                    "reason": SEALED_STATUS,
                    "receipt": str(SEAL_RECEIPT),
                    "weights_reconstructed": False,
                }
            ),
            flush=True,
        )
        return 0
    from lab.operators.deepseek_v4_resource_supervisor import main as operator_main

    return int(operator_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
