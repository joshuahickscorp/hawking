"""Contract tests for the nanosecond Gravity tool benchmark."""

from tools.audit.gravity_tool_benchmark import _timed


def test_timed_emits_integer_nanoseconds_only():
    timing = _timed(lambda: None, repeats=3)

    assert set(timing) == {"min_ns", "median_ns", "max_ns"}
    assert all(isinstance(value, int) for value in timing.values())
    assert timing["min_ns"] <= timing["median_ns"] <= timing["max_ns"]
