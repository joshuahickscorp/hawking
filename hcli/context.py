"""Worker context packets.

The old ``Context`` class stored the full root goal and is gone. The only
worker-context type is ``WorkerPacket`` in ``goal.py``, next to the compiler
that produces it. This module re-exports that type so the class object is
the same.

Verified caller: ``tools/headless/startup_census.py`` lists this module.
Production code imports ``WorkerPacket`` from ``hcli.goal`` /
``hcli.agentos``. Not a second packet shape.
"""
from __future__ import annotations

from .goal import WorkerPacket, compile_worker_context

__all__ = ["WorkerPacket", "compile_worker_context"]
