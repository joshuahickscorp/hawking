#!/usr/bin/env python3.12
"""Deprecated 32/82-GiB long-run finalizer.

The sealed historical report remains in commit 2910050e.  The Gravity closure
campaign makes 5 GiB authoritative, so this entry point intentionally cannot
rewrite current status or emit obsolete Telegram policy.
"""
from __future__ import annotations

import json


def main() -> int:
    print(json.dumps({
        "status": "DEPRECATED",
        "reason": "SUPERSEDED_BY_KIMI_K26_GRAVITY_CLOSURE_5_GIB_POLICY",
        "historical_report_commit": "2910050ea93bd79309ba2e1af0b059a509e98b67",
        "authoritative_disk_floor_bytes": 5368709120,
    }, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
