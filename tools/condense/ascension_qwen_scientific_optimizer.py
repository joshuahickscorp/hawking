#!/usr/bin/env python3
"""Launch the additive deterministic Qwen scientific optimizer lane."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.ascension_qwen_scientific_optimizer import main


if __name__ == "__main__":
    raise SystemExit(main())
