#!/usr/bin/env python3
"""Seal the candidate-only HQ30GR2 Qwen30 runtime integration handoff."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.ascension_qwen30_quality_repack_integration_handoff import main


if __name__ == "__main__":
    raise SystemExit(main())
