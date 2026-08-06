#!/usr/bin/env python3.12
"""CLI entry point for bounded, header-only DeepSeek-V4 streaming admission.

This command never downloads a model or reads a tensor body.  Use ``plan`` to
seal the exact byte interval, have the separately approved transport capture
that header range, then use ``header`` to validate it and seal a receipt.
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.deepseek_v4_stream_executor import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
