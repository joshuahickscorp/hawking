#!/usr/bin/env python3
"""Launch the candidate-only Qwen30 HQ30GR2 direct-packed CPU matvec harness."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.ascension_qwen30_quality_repack_packed_matvec_parity import main


if __name__ == "__main__":
    raise SystemExit(main())
