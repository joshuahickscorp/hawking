#!/usr/bin/env python3.12
"""CLI entry for the architecture-config teacher-forced executor."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.frankenstein_teacher_forced_executor import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
