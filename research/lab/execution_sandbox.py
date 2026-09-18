"""Internal compatibility view of Hawking's canonical sandbox policy."""
from __future__ import annotations

from hawking import execution_sandbox as _canonical
from hawking.execution_sandbox import *  # noqa: F401,F403


def __getattr__(name: str):
    return getattr(_canonical, name)
