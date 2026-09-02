"""Truth-bound tests for the durable EventSink / read_events tailer.

The worker's EventBus drops every event but "runtime_ready" today; EventSink
is the durable half of the fix, so these tests are the proof that a) events
really do land on disk, b) a reader can tail them incrementally by offset
without re-reading old lines or exploding on a half-written one, and c) a
write can never propagate an exception into the caller.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict
from unittest.mock import patch

from hcli.agentos.event_sink import EventSink, read_events


@dataclass
class _Event:
    type: str
    data: Dict[str, Any] = field(default_factory=dict)


def test_write_round_trips_event_object_and_dict(tmp_path):
    sink = EventSink(tmp_path)
    sink.write(_Event("tool_call_started", {"tool": "grep"}))
    sink.write({"type": "tool_call_finished", "data": {"tool": "grep", "ok": True}})

    events, offset = read_events(sink.path)
    assert [e["type"] for e in events] == ["tool_call_started", "tool_call_finished"]
    assert events[0]["data"] == {"tool": "grep"}
    assert events[0]["seq"] == 0
    assert events[1]["seq"] == 1
    assert "t" in events[0]
    assert offset == sink.path.stat().st_size


def test_incremental_tailing_by_offset_only_returns_new_events(tmp_path):
    sink = EventSink(tmp_path)
    sink.write(_Event("a"))
    first_batch, offset = read_events(sink.path)
    assert [e["type"] for e in first_batch] == ["a"]

    sink.write(_Event("b"))
    sink.write(_Event("c"))
    second_batch, offset2 = read_events(sink.path, offset=offset)
    assert [e["type"] for e in second_batch] == ["b", "c"]
    assert offset2 > offset

    # nothing new since offset2 - re-reading yields no events, same offset
    third_batch, offset3 = read_events(sink.path, offset=offset2)
    assert third_batch == []
    assert offset3 == offset2


def test_torn_trailing_line_is_not_returned_and_offset_does_not_advance_past_it(tmp_path):
    sink = EventSink(tmp_path)
    sink.write(_Event("complete"))
    complete_size = sink.path.stat().st_size

    # simulate a writer that appended a partial line (no trailing "\n" yet)
    with open(sink.path, "a", encoding="utf-8") as handle:
        handle.write('{"t": 1.0, "type": "torn", "data": {}')  # no closing brace, no newline

    events, offset = read_events(sink.path)
    assert [e["type"] for e in events] == ["complete"]
    assert offset == complete_size  # stopped before the torn line, not past it

    # completing the line on a later append makes it readable
    with open(sink.path, "a", encoding="utf-8") as handle:
        handle.write('}\n')
    events2, offset2 = read_events(sink.path, offset=offset)
    assert [e["type"] for e in events2] == ["torn"]
    assert offset2 == sink.path.stat().st_size


def test_missing_file_returns_empty_without_raising(tmp_path):
    events, offset = read_events(tmp_path / "no" / "such" / "events.jsonl")
    assert events == []
    assert offset == 0


def test_rotation_and_offset_recovery(tmp_path):
    sink = EventSink(tmp_path, max_bytes=10_000)
    for i in range(5):
        sink.write(_Event(f"evt{i}"))
    stale_offset = sink.path.stat().st_size  # a reader parked here, mid-file

    sink._max_bytes = 1  # force the next append to see itself over budget
    sink.write(_Event("after_rotation"))

    rotated_path = sink.path.with_name(sink.path.name + ".1")
    assert rotated_path.exists()
    first_rotated = json.loads(rotated_path.read_text().splitlines()[0])
    assert first_rotated["type"] == "evt0"

    # the fresh live file (one event) is well short of the stale offset
    assert sink.path.stat().st_size < stale_offset
    events, offset = read_events(sink.path, offset=stale_offset)
    assert [e["type"] for e in events] == ["after_rotation"]
    assert offset > 0


def test_nonserialisable_payload_is_coerced_not_dropped(tmp_path):
    sink = EventSink(tmp_path)

    class Weird:
        def __str__(self) -> str:
            return "weird-repr"

    sink.write(_Event("odd", {"blob": Weird(), "items": {1, 2, 3}}))
    assert sink.dropped == 0
    events, _ = read_events(sink.path)
    assert events[0]["data"]["blob"] == "weird-repr"
    # a set has no native JSON encoding either; json's default=str hook
    # stringifies it whole rather than the write being dropped.
    assert isinstance(events[0]["data"]["items"], str)
    assert events[0]["data"]["items"].startswith("{") and "1" in events[0]["data"]["items"]


def test_write_never_raises_and_counts_dropped(tmp_path):
    sink = EventSink(tmp_path)
    with patch("hcli.agentos.event_sink.json.dumps", side_effect=RuntimeError("boom")):
        sink.write(_Event("x"))  # must not raise
    assert sink.dropped == 1
    # sink recovers once the induced failure is gone
    sink.write(_Event("y"))
    assert sink.dropped == 1
    events, _ = read_events(sink.path)
    assert [e["type"] for e in events] == ["y"]
