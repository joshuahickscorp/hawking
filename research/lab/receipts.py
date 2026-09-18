"""Internal compatibility view of Hawking's canonical receipt authority."""
from __future__ import annotations

from hawking import receipts as _canonical
from hawking.receipts import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_canonical, name)
