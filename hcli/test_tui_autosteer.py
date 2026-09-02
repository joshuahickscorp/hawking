"""D2/D3: a plain line auto-steers a running mission instead of racing it
with a second goal, and /quit (not /exit) is the primary kill verb.

`_wired_on_input` mimics app.py's `_handle_input`: it emits `user_message`
on the real event bus, and for a routed "/steer ..." line it also emits the
`final_response` that `_cmd_steer`'s own confirmation would produce. Without
that wiring these tests could pass on an empty transcript for the wrong
reason; with it, the transcript is what a real command dispatch would leave.

Runnable two ways:

    python3 -m pytest hcli/test_tui_autosteer.py -q
    python3 hcli/test_tui_autosteer.py
"""
from __future__ import annotations

import json
import sys

from hcli.command_registry import COMMANDS, handler_name
from hcli.commands import CommandHandler
from hcli.events import Event, EventBus
from hcli.mission import mission_state_path
from hcli.tui import TUI


def _tui(workspace="/tmp/ws-autosteer"):
    return TUI(EventBus(), workspace, "qwen3.8", 1)


def _one_shot(lines):
    it = iter(lines)

    def prompt(_msg):
        try:
            return next(it)
        except StopIteration:
            raise EOFError

    return prompt


def _wired_on_input(t, seen):
    def on_input(text):
        seen.append(text)
        t.bus.emit("user_message", {"text": text})
        if text.startswith("/steer "):
            arg = text[len("/steer "):]
            t.bus.emit("final_response", {"content": f"✓ Steer queued: {arg}"})

    return on_input


def test_plain_line_with_running_mission_queues_steer_not_a_goal():
    t = _tui()
    t._mission_running = lambda: True
    t._prompt_fn = _one_shot(["do the thing", "/quit"])
    seen = []
    assert t.run(_wired_on_input(t, seen)) == 0
    assert seen == ["/steer do the thing", "/quit"], seen


def test_plain_line_with_no_running_mission_starts_a_goal_as_today():
    t = _tui()
    t._mission_running = lambda: False
    t._prompt_fn = _one_shot(["do the thing", "/quit"])
    seen = []
    assert t.run(_wired_on_input(t, seen)) == 0
    assert seen == ["do the thing", "/quit"], seen
    assert any(row == "You: do the thing" for row in t.transcript), t.transcript


def test_auto_steer_note_replaces_the_echo_and_swallows_the_confirmation():
    t = _tui()
    t._mission_running = lambda: True
    t._prompt_fn = _one_shot(["do the thing", "/quit"])
    seen = []
    t.run(_wired_on_input(t, seen))
    assert t.transcript == ["✓ steering: do the thing", "You: /quit"], t.transcript


def test_auto_steer_note_clips_to_60_chars_but_routes_the_full_line():
    t = _tui()
    t._mission_running = lambda: True
    long_text = "x" * 90
    t._prompt_fn = _one_shot([long_text, "/quit"])
    seen = []
    t.run(_wired_on_input(t, seen))
    assert seen[0] == "/steer " + long_text, "the full line must still reach the mission"
    note = next(row for row in t.transcript if row.startswith("✓ steering:"))
    head = note[len("✓ steering: "):]
    assert len(head) == 60 and head.endswith("…"), note


def test_explicit_steer_command_still_works_as_a_hidden_alias():
    """Muscle memory holds: typing /steer directly is untouched by
    auto-steer, whether or not a mission happens to be running."""
    for running in (True, False):
        t = _tui()
        t._mission_running = lambda running=running: running
        t._prompt_fn = _one_shot(["/steer keep it simple", "/quit"])
        seen = []
        t.run(_wired_on_input(t, seen))
        assert seen[0] == "/steer keep it simple", (running, seen)
        assert any(row == "You: /steer keep it simple" for row in t.transcript), (
            running,
            t.transcript,
        )


def test_a_slash_command_never_gets_auto_steer_prefixed():
    t = _tui()
    t._mission_running = lambda: True
    t._prompt_fn = _one_shot(["/status", "/quit"])
    seen = []
    t.run(_wired_on_input(t, seen))
    assert seen[0] == "/status"


def test_mission_running_reads_the_real_checkpoint_shape(tmp_path):
    """Not a guessed shape: this is the literal field Mission.checkpoint()
    writes (mission.py), read cross-process."""
    t = TUI(EventBus(), str(tmp_path), "qwen3.8", 1)
    assert t._mission_running() is False, "no mission file yet"

    path = mission_state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"phase": "running"}))
    assert t._mission_running() is True

    path.write_text(json.dumps({"phase": "completed"}))
    assert t._mission_running() is False

    path.write_text("not json")
    assert t._mission_running() is False, "a torn/partial write must not raise or read as running"


def test_quit_is_the_primary_verb_and_exit_is_its_alias():
    quit_cmd = next(c for c in COMMANDS if c.name == "/quit")
    assert "/exit" in quit_cmd.aliases
    assert not any(c.name == "/exit" for c in COMMANDS), "must not exist twice"
    assert handler_name("/quit") == "_cmd_quit"
    assert handler_name("/exit") == "_cmd_exit"


class _StubController:
    def __init__(self):
        self.exits = 0

    def request_exit(self):
        self.exits += 1


def test_exit_resolves_to_the_same_handler_as_quit():
    for spelling in ("/quit", "/exit"):
        controller = _StubController()
        handler = CommandHandler(controller)
        handler.handle(spelling)
        assert controller.exits == 1, spelling


def test_status_line_never_carries_model_text_from_an_event_that_carries_some():
    t = _tui()
    secret = "the model's private chain of thought must never reach the status line"
    t._on_event(Event("final_response", {"content": secret, "status": "error"}))
    assert secret not in t.status, t.status
    assert t.status == "error"
    # The same text is fine in the transcript -- only the status word must
    # stay clean.
    assert any(secret in row for row in t.transcript)

    t2 = _tui()
    t2._begin_turn()
    t2._on_event(Event("heartbeat", {"phase": secret, "elapsed_s": 2}))
    assert secret not in t2.status, t2.status
    t2._end_turn()


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
