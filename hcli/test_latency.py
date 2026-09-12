from __future__ import annotations

from hcli.latency import event_elapsed_ns, format_duration_ns, wall_elapsed_ns


def test_new_nanosecond_duration_wins_over_legacy_seconds():
    assert event_elapsed_ns({"elapsed_ns": 417, "elapsed_s": 99.0}) == 417


def test_legacy_duration_is_read_without_becoming_the_write_format():
    assert event_elapsed_ns({"elapsed_s": 0.125}) == 125_000_000


def test_wall_duration_prefers_exact_nanoseconds_and_reads_legacy_seconds():
    assert wall_elapsed_ns({"wall_ns": 417, "wall_s": 99.0}) == 417
    assert wall_elapsed_ns({"wall_s": 0.125}) == 125_000_000


def test_duration_formatter_keeps_small_measurements_visible():
    assert format_duration_ns(417) == "417ns"
    assert format_duration_ns(12_500) == "12.5µs"
    assert format_duration_ns(1_250_000) == "1.2ms"


def test_engine_model_scope_emits_nanoseconds():
    from hcli.engine import Engine

    engine = Engine.__new__(Engine)
    engine._active_goal_id = None
    engine._last_call_plan = {}
    events = []
    engine._emit = lambda event_type, data: events.append((event_type, data))

    with engine._model_call_scope():
        pass

    finished = [data for kind, data in events if kind == "model_call_finished"]
    assert len(finished) == 1
    assert isinstance(finished[0]["elapsed_ns"], int)
    assert "elapsed_s" not in finished[0]


def test_tool_result_keeps_exact_nanosecond_duration():
    from hcli.tool_registry import ToolResult

    result = ToolResult(
        tool="fixture",
        invocation_id="fixture-1",
        ok=True,
        started_ns=10_000,
        finished_ns=10_417,
    )
    payload = result.to_dict()
    assert payload["elapsed_ns"] == 417
    assert payload["timing_unit"] == "ns"
