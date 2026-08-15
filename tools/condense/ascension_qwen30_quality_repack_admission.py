#!/usr/bin/env python3
"""Launch wrapper for the isolated Qwen30 quality-repack native admission watcher."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.ascension_qwen30_quality_repack_admission import main


if __name__ == "__main__":
    raise SystemExit(main())
