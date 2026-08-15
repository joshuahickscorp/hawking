#!/usr/bin/env python3
"""Installed-path entry point for the detached Ascension Telegram observer."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lab.operators.ascension_telegram_notifier import main


if __name__ == "__main__":
    raise SystemExit(main())
