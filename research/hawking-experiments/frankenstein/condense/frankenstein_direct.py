#!/usr/bin/env python3.12
"""CLI entry for the ceremony-free Frankenstein direct fusion harness."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab.layout import ensure_experiment_imports  # noqa: E402
ensure_experiment_imports()

from frankenstein_direct import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
