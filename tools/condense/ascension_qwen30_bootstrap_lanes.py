#!/usr/bin/env python3
"""Installed-path wrapper for dedicated Qwen30 runtime and TG3 lanes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.ascension_qwen30_bootstrap_lanes import main


if __name__ == "__main__":
    raise SystemExit(main())
