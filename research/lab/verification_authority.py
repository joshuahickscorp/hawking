"""Internal compatibility view of Hawking's canonical verification authority."""
from __future__ import annotations

from hawking import verification_authority as _canonical
from hawking.verification_authority import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_canonical, name)
