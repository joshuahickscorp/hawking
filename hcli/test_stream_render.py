"""hcli/stream_render.py: the pure formatter shared by the TUI and any
resident watcher. See the module docstring for why this exists.
"""

from hcli.stream_render import event_phase, is_terminal_event, render_event

# event dict -> expected render_event() output, exact strings (not "truthy").
# One row per event type TUI._on_event handles today (31 arms in hcli/tui.py).
TABLE = [
    ({"type": "activity_started", "data": {"label": "compiling"}}, []),
    ({"type": "activity_completed", "data": {"label": "compiling"}}, ["✓ compiling"]),
    ({"type": "activity_completed", "data": {}}, ["✓ done"]),
    ({"type": "user_message", "data": {"text": "hello there"}}, ["You: hello there"]),
    ({"type": "runtime_loading", "data": {"model": "q80"}},
     ["… loading resident: q80"]),
    ({"type": "runtime_ready", "data": {"admitted": 12}},
     ["✓ resident ready (12 admitted)"]),
    ({"type": "runtime_ready", "data": {}}, ["✓ resident ready"]),
    ({"type": "workunit_started", "data": {}}, []),
    ({"type": "workunit_completed", "data": {}}, []),
    ({"type": "final_response", "data": {"content": "the answer is 4"}},
     ["the answer is 4"]),
    ({"type": "final_response", "data": {"content": ""}}, []),
    ({"type": "error", "data": {"message": "boom"}}, ["✗ boom"]),
    ({"type": "error", "data": {}}, ["✗ error"]),
    ({"type": "rollback", "data": {"reason": "bad diff"}},
     ["✗ rollback: bad diff"]),
    ({"type": "rollback", "data": {}}, ["✗ rollback: rollback"]),
    ({"type": "validation_failed", "data": {}}, ["✗ validation failed"]),
    ({"type": "goal_completed", "data": {"status": "failed"}},
     ["✗ goal failed"]),
    ({"type": "goal_completed", "data": {"status": "cancelled"}},
     ["✗ goal cancelled"]),
    ({"type": "goal_completed", "data": {"status": "success"}}, []),
    ({"type": "steer_queued", "data": {}}, ["✓ Steer queued"]),
    ({"type": "bank_queued", "data": {"id": "b1", "goal": "ship it"}},
     ["▣ Banked b1: ship it"]),
    ({"type": "bank_started", "data": {"id": "b1", "goal": "ship it"}},
     ["▶ Bank starting b1: ship it"]),
    ({"type": "bank_finished", "data": {"id": "b1", "status": "ok"}},
     ["✓ Bank finished b1 status=ok"]),
    ({"type": "bank_dropped", "data": {"id": "b1"}}, ["✓ Bank dropped b1"]),
    ({"type": "bank_cleared", "data": {"removed": 3}},
     ["✓ Bank cleared 3 goal(s)"]),
    ({"type": "bank_cleared", "data": {}}, ["✓ Bank cleared 0 goal(s)"]),
    ({"type": "transcript_cleared", "data": {}}, []),
    ({"type": "warning", "data": {"message": "low disk"}}, ["! low disk"]),
    ({"type": "warning", "data": {}}, ["! warning"]),
    ({"type": "evidence_gathering_started", "data": {}}, []),
    ({"type": "evidence_gathering_finished", "data": {"file_count": 5}},
     ["evidence  5 files"]),
    ({"type": "evidence_gathering_finished", "data": {"files": ["a", "b"]}},
     ["evidence  2 files"]),
    ({"type": "goal_compiled", "data": {"workunits": 7}},
     ["compiled  7 units"]),
    ({"type": "goal_compiled", "data": {"unit_count": 2}},
     ["compiled  2 units"]),
    ({"type": "model_call_started", "data": {"prompt_tokens": 500}}, []),
    ({"type": "model_call_finished", "data": {"elapsed_s": 1.2}}, []),
    ({"type": "heartbeat", "data": {"phase": "thinking"}}, []),
    ({"type": "tool_call_started", "data": {"tool": "fs.read"}},
     ["⏺ fs.read()"]),
    ({"type": "tool_call_started",
      "data": {"tool": "fs.read", "args": {"path": "a.py"}}},
     ["⏺ fs.read(path=a.py)"]),
    ({"type": "tool_call_finished",
      "data": {"tool": "fs.read", "ok": True, "elapsed_s": 0.4}},
     ["  ⎿ ok  0.4s"]),
    ({"type": "tool_call_finished", "data": {"tool": "fs.read", "ok": False}},
     ["  ⎿ failed"]),
    ({"type": "tool_invoked",
      "data": {"tool": "fs.read", "ok": True, "elapsed_s": 0.4}},
     ["  ⎿ ok  0.4s"]),
    ({"type": "validation_started", "data": {}}, []),
    ({"type": "mutation_prepared", "data": {}}, []),
    # unknown event type: silent, never raises
    ({"type": "some_future_event", "data": {"whatever": 1}}, []),
]


def test_table_has_enough_rows_to_mean_something():
    # A table with a handful of rows could "cover everything" by accident.
    # This repo's own house rule: a check over a near-empty collection that
    # still goes green is a bug, not a pass.
    assert len(TABLE) >= 20
    # And it must actually span distinct event types, not repeat one type
    # dressed up as many rows.
    assert len({ev["type"] for ev, _ in TABLE}) >= 20


def test_render_event_matches_table_exactly():
    for event, expected in TABLE:
        got = render_event(event)
        assert got == expected, f"{event['type']}: got {got!r}, want {expected!r}"


# --- malformed input: must never raise ---

# type itself is missing/unrecognizable -> nothing to render, no phase.
UNKNOWN_TYPE = [
    {},
    {"type": None},
    {"type": 123},
    {"data": {"text": "hi"}},  # no "type" key at all
    None,
    "not-an-event",
    [],
    42,
]

# type is a real, recognized event, but "data" itself is garbage -> must
# fall back to its empty-dict defaults rather than raising.
VALID_TYPE_BAD_DATA = [
    {"type": "tool_call_started", "data": None},
    {"type": "tool_call_started", "data": "not-a-dict"},
    {"type": "tool_call_started", "data": []},
    {"type": "final_response"},  # no "data" key at all
    {"type": "goal_completed", "data": None},
    {"type": "heartbeat", "data": 7},
]


def test_unknown_type_events_never_raise_and_render_nothing():
    for event in UNKNOWN_TYPE:
        assert render_event(event) == []
        assert event_phase(event) is None
        assert is_terminal_event(event) is False


def test_valid_type_bad_data_never_raises():
    for event in VALID_TYPE_BAD_DATA:
        lines = render_event(event)
        assert isinstance(lines, list) and all(isinstance(x, str) for x in lines)
        event_phase(event)  # must not raise
        assert isinstance(is_terminal_event(event), bool)


# --- the chain-of-thought leak this module exists to prevent ---

def test_model_text_never_leaks_onto_the_phase_word():
    thinking_content = "the secret reasoning is: the user's password is hunter2"
    event = {"type": "final_response", "data": {"content": thinking_content, "status": "ok"}}
    # The text is fine to show as a transcript line...
    assert render_event(event) == [thinking_content]
    # ...but "ok" is not terminal, so nothing should touch the phase word,
    # and in particular the free-text content must never come back as one.
    assert event_phase(event) is None

    # A heartbeat's "phase" field is meant to be a short status word, but if
    # something upstream ever stuffs prose into it, _status_word's regex
    # (no spaces, <=64 chars) must refuse to pass it through as a phase.
    leaked = {"type": "heartbeat", "data": {"phase": thinking_content}}
    assert event_phase(leaked) == "working"
    assert thinking_content not in (event_phase(leaked) or "")

    # Same guard for activity_started and tool_call_started labels/names.
    leaked_label = {"type": "activity_started", "data": {"label": thinking_content}}
    assert event_phase(leaked_label) == "working"


def test_tool_call_lines_shaped_like_claude_code():
    call = render_event({"type": "tool_call_started",
                          "data": {"tool": "fs.search", "args": {"pattern": "TODO"}}})
    outcome = render_event({"type": "tool_call_finished",
                             "data": {"tool": "fs.search", "ok": True, "elapsed_s": 0.05}})
    assert call == ["⏺ fs.search(pattern=TODO)"]
    assert outcome == ["  ⎿ ok  0.1s"]
    # outcome nests visually under the call: indented, marked with ⎿
    assert outcome[0].startswith("  ⎿ ")


def test_is_terminal_event_examples():
    assert is_terminal_event({"type": "error", "data": {}}) is True
    assert is_terminal_event({"type": "rollback", "data": {}}) is True
    assert is_terminal_event({"type": "goal_completed", "data": {"status": "failed"}}) is True
    assert is_terminal_event({"type": "goal_completed", "data": {"status": "ok"}}) is False
    assert is_terminal_event({"type": "heartbeat", "data": {}}) is False
