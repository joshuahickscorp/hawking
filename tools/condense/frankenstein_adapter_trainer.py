#!/usr/bin/env python3.12
"""CLI entry for reversible adapter trainer (paired activations)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators.frankenstein_adapter_trainer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
