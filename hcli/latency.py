"""Canonical duration handling for HCLI's live control plane.

Wall-clock timestamps remain Unix seconds because they are provenance fields.
Durations are monotonic integer nanoseconds: they do not jump when the system
clock is adjusted and they do not lose sub-millisecond work to float rounding.
Readers accept the old ``*_s`` event fields as a compatibility path for
historical receipts, but new producers should emit ``*_ns`` only.
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Optional


def now_ns() -> int:
    """Return the monotonic clock used for duration measurements."""
    return time.perf_counter_ns()


def since_ns(started_ns: int) -> int:
    """Return a non-negative monotonic duration in integer nanoseconds."""
    return max(0, now_ns() - int(started_ns))


def event_elapsed_ns(data: Mapping[str, Any]) -> Optional[int]:
    """Read a duration from a new event, with a legacy receipt fallback.

    The fallback is intentionally read-only. It keeps old mission logs
    interpretable without allowing new code to continue writing float seconds.
    """
    value = data.get("elapsed_ns")
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    legacy = data.get("elapsed_s")
    if isinstance(legacy, (int, float)) and not isinstance(legacy, bool):
        return max(0, int(round(float(legacy) * 1_000_000_000)))
    return None


def wall_elapsed_ns(data: Mapping[str, Any]) -> Optional[int]:
    """Read a wall-duration field with a legacy seconds fallback."""
    value = data.get("wall_ns")
    if isinstance(value, int) and not isinstance(value, bool):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    legacy = data.get("wall_s")
    if isinstance(legacy, (int, float)) and not isinstance(legacy, bool):
        return max(0, int(round(float(legacy) * 1_000_000_000)))
    return None


def seconds_from_ns(value: Any) -> Optional[float]:
    """Convert an integer duration to seconds only at a presentation edge."""
    if isinstance(value, bool):
        return None
    try:
        return max(0.0, int(value) / 1_000_000_000.0)
    except (TypeError, ValueError):
        return None


def format_duration_ns(value: Any) -> str:
    """Format a duration compactly without throwing away its measured unit."""
    try:
        ns = max(0, int(value))
    except (TypeError, ValueError):
        return ""
    if ns < 1_000:
        return f"{ns}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f}µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.1f}ms"
    return f"{ns / 1_000_000_000:.1f}s"


__all__ = [
    "event_elapsed_ns",
    "format_duration_ns",
    "now_ns",
    "seconds_from_ns",
    "since_ns",
    "wall_elapsed_ns",
]
