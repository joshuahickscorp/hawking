"""Phase timings for one HCLI cycle, as a reversible probe.

Four external stopwatch readings of the same task spanned 177.6-267.0 s and
could not separate resident load from prompt compilation from generation. No
number of repetitions fixes that; only instrumentation does.

This patches at the TYPE (Python resolves methods there) and restores on exit,
so nothing in the production path changes. Import it, run a cycle, read the
report.
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

TIMINGS: dict[str, list[float]] = defaultdict(list)


def _wrap(owner: Any, name: str, label: str) -> tuple | None:
    fn = getattr(owner, name, None)
    if fn is None:
        return None

    def timed(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            TIMINGS[label].append(time.perf_counter() - t)

    setattr(owner, name, timed)
    return (owner, name, fn)


@contextmanager
def phase_timing():
    """Time the phases of whatever runs inside the block."""
    import subprocess

    from hcli import engine as E
    from hcli import hawking_native as N

    TIMINGS.clear()
    patches = [
        _wrap(E.Engine, "_call_model", "engine._call_model"),
        _wrap(E.Engine, "_post_completion", "engine._post_completion"),
        _wrap(E.Engine, "_build_model_payload", "engine._build_model_payload"),
        _wrap(subprocess, "run", "subprocess.run"),
        _wrap(subprocess.Popen, "__init__", "subprocess.Popen.__init__"),
    ]
    # Real class names, read from the module rather than guessed. The first
    # version of this probe guessed three names, matched none, and reported a
    # single opaque 95.5% block.
    for cls_name, methods in (
        ("HawkingNativeConnector",
         ("complete_payload", "start", "_run_one_shot", "_restart_resident")),
        ("ResidentProcess", ("start",)),
        ("_TokenizerRenderer", ("_load",)),
    ):
        cls = getattr(N, cls_name, None)
        if cls is not None:
            for m in methods:
                patches.append(_wrap(cls, m, f"{cls_name}.{m}"))
    try:
        yield TIMINGS
    finally:
        for p in patches:
            if p:
                setattr(p[0], p[1], p[2])


def report(total_wall: float | None = None) -> str:
    rows = sorted(((sum(v), len(v), k) for k, v in TIMINGS.items()), reverse=True)
    out = [f"  {'phase':34} {'calls':>6} {'total s':>9} {'share':>7}"]
    for tot, n, k in rows:
        share = f"{tot / total_wall * 100:6.1f}%" if total_wall else "      -"
        out.append(f"  {k:34} {n:>6} {tot:>9.2f} {share:>7}")
    if total_wall:
        named = sum(t for t, _, _ in rows if "." in _ or True)
        out.append(f"  {'(wall)':34} {'':>6} {total_wall:>9.2f}")
    return "\n".join(out)
