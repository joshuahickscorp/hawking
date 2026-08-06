#!/usr/bin/env python3.12
"""CLI entry point for sustained DeepSeek-V4 public-path confirmation."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.deepseek_v4_xet_sustained import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
