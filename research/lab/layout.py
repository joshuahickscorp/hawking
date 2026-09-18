"""Internal compatibility view of Hawking's canonical workspace layout.

New runtime code imports :mod:`hawking.layout`.  Retained research modules keep
this import path so historical experiments can run without carrying a second
layout implementation.
"""
from __future__ import annotations

from hawking import layout as _canonical
from hawking.layout import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_canonical, name)
