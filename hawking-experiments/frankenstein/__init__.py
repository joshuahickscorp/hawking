"""Frankenstein experiment package.

The parent directory ``hawking-experiments`` is not a valid Python package
name (dash). This module puts ``operators`` and ``condense`` on ``sys.path``
so ``import frankenstein_ablation`` and the other 57 modules resolve.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT / "condense", _ROOT / "operators"):
    _s = str(_p)
    if _p.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)
