"""Internal compatibility view of Hawking's canonical semantic taxonomy."""
from __future__ import annotations

from hawking import semantic_taxonomy as _canonical
from hawking.semantic_taxonomy import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_canonical, name)
