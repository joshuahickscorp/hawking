#!/usr/bin/env python3
"""CLI for Qwen3.8 genesis children. See lab.operators.qwen38_genesis_children."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lab.operators.qwen38_genesis_children import main

if __name__ == "__main__":
    raise SystemExit(main())
