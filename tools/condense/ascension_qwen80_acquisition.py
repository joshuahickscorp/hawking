#!/usr/bin/env python3
"""Installed-path wrapper for detached Qwen80 full source acquisition."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.ascension_qwen80_acquisition import main


if __name__ == "__main__":
    raise SystemExit(main())
