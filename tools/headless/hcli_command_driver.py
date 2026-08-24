#!/usr/bin/env python3
"""Drive HCLI's real command plane headlessly and capture what a user would see.

The command-plane bugs this exists to catch are *rendering* bugs: a handler that
returns a perfectly good string which never reaches the screen. So this driver
does not call handlers directly. It wires the same objects the interactive path
wires -- ``Controller`` -> ``CommandHandler`` -> ``EventBus`` -> ``TUI._on_event``
-> ``TUI.render_transcript`` -- and reports the rendered transcript delta per
command. If a command is blank in the real TUI it is blank here.

    python3 tools/headless/hcli_command_driver.py
    python3 tools/headless/hcli_command_driver.py --json
    python3 tools/headless/hcli_command_driver.py --commands /help,/status

Exit 0 when every driven command rendered non-empty output, 1 otherwise.
No model inference. No runtime spawn (the pool is lazy).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

from hcli.controller import Controller  # noqa: E402
from hcli.events import EventBus  # noqa: E402
from hcli.models import ModelRegistry  # noqa: E402
from hcli.tui import TUI  # noqa: E402
from hcli.workspace import Workspace  # noqa: E402

# Commands safe to drive without a mission, a model, or a network call.
# /exit and /quit terminate the loop and are checked separately.
DEFAULT_COMMANDS = (
    "/help",
    "/status",
    "/models",
    "/mission",
    "/context",
    "/ultragoal",
    "/steer probe steering line",
    "/grok status",
    "/clear",
)


class Harness:
    """The interactive wiring, minus the blocking prompt loop."""

    def __init__(self, workspace: str, runtime_count: int = 1) -> None:
        self.ws = Workspace(workspace)
        self.bus = EventBus()
        self.registry = ModelRegistry()
        self.controller = Controller(
            workspace=self.ws,
            runtime_count=runtime_count,
            model=None,
            bus=self.bus,
            registry=self.registry,
        )
        self.tui = TUI(
            event_bus=self.bus,
            workspace=self.ws.root,
            model_name=self.controller.model_name or "local",
            runtime_count=runtime_count,
        )
        # Same subscription TUI.run() makes before it starts prompting.
        self.bus.subscribe(self.tui._on_event)

    def send(self, line: str) -> Dict[str, Any]:
        """One turn through App._handle_input, then render like TUI.run does."""
        before = len(self.tui.transcript)
        started = time.perf_counter()
        error: Optional[str] = None
        returned: Any = None

        # App._handle_input emits user_message first, then dispatches.
        self.bus.emit("user_message", {"text": line})
        try:
            returned = self.controller.handle_command(line)
        except Exception as exc:  # a raising command is also a rendering failure
            error = f"{type(exc).__name__}: {exc}"

        wall = time.perf_counter() - started
        after = len(self.tui.transcript)
        # A clearing command truncates the transcript, so `before` no longer
        # indexes into it. Everything present afterwards is then new.
        cleared = after < before
        new_entries = self.tui.transcript if cleared else self.tui.transcript[before:]
        # The transcript entry for the echoed user line is not command output.
        rendered = [e for e in new_entries if not e.startswith("You: ")]
        rendered_text = "\n".join(rendered).strip()

        return {
            "command": line,
            "wall_s": round(wall, 4),
            "handler_returned_chars": len(str(returned)) if returned is not None else 0,
            "handler_returned_none": returned is None,
            "cleared_transcript": cleared,
            "rendered_entries": len(rendered),
            "rendered_chars": len(rendered_text),
            "rendered_text": rendered_text,
            "error": error,
            # A clearing command legitimately renders nothing into the
            # transcript: render_transcript() then shows "(no activity yet)"
            # and the acknowledgement sits on the status line. Judging it by
            # transcript bytes would demand the wrong behaviour.
            "status_line": self.tui.status,
            "visible": (bool(rendered_text) or cleared) and error is None,
        }


def drive(workspace: str, commands: List[str]) -> Dict[str, Any]:
    h = Harness(workspace)
    turns: List[Dict[str, Any]] = []
    try:
        for line in commands:
            turns.append(h.send(line))
    finally:
        try:
            h.controller.shutdown()
        except Exception:
            pass

    invisible = [t["command"] for t in turns if not t["visible"]]
    return {
        "gate": "HCLI_COMMAND_INGRESS_UNIFIED",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workspace": workspace,
        "path_exercised": (
            "Controller.handle_command -> CommandHandler.handle -> EventBus -> "
            "TUI._on_event -> TUI.render_transcript  (the same wiring App._run_interactive uses)"
        ),
        "turns": turns,
        "invisible_commands": invisible,
        "all_visible": not invisible,
    }


def clear_canary(tmp_workspace: str) -> Dict[str, Any]:
    """/clear must clear the transcript and nothing else.

    Run in a throwaway workspace so the canary cannot damage the repo's own
    durable mission state.
    """
    h = Harness(tmp_workspace)
    ws = Path(tmp_workspace)
    goal_md = ws / ".hcli" / "GOAL.md"

    def durable() -> Dict[str, Any]:
        ctrl = h.controller
        session = getattr(ctrl, "session", None)
        return {
            "goal": getattr(session, "goal", None) if session else None,
            "mission_id": getattr(session, "mission_id", None) if session else None,
            "goal_md_bytes": goal_md.stat().st_size if goal_md.is_file() else None,
            "goal_md_sha": (
                __import__("hashlib").sha256(goal_md.read_bytes()).hexdigest()
                if goal_md.is_file() else None
            ),
            # Steering lives on the session, not the controller. A count that
            # is always zero would make this canary comfortable and useless.
            "steer_count": len(getattr(session, "steering", []) or []) if session else 0,
        }

    try:
        h.send("/ultragoal keep this durable goal alive across a transcript clear")
        h.send("/steer this steer must survive /clear")
        before = durable()
        transcript_before = len(h.tui.transcript)
        turn = h.send("/clear")
        after = durable()
        transcript_after = len(h.tui.transcript)
    finally:
        try:
            h.controller.shutdown()
        except Exception:
            pass

    preserved = {k: before[k] == after[k] for k in before}
    return {
        "canary": "clear_preserves_mission",
        "workspace": tmp_workspace,
        "durable_before": before,
        "durable_after": after,
        "preserved": preserved,
        "transcript_entries_before": transcript_before,
        "transcript_entries_after": transcript_after,
        "transcript_was_cleared": bool(turn.get("cleared_transcript")),
        "clear_rendered_chars": turn.get("rendered_chars"),
        # The acknowledgement lives on the status line, because the transcript
        # is the thing being emptied. Requiring transcript output here would
        # demand that /clear leave behind exactly what it was asked to remove.
        "clear_status_line": turn.get("status_line"),
        "ok": all(preserved.values())
        and bool(turn.get("cleared_transcript"))
        and transcript_after < transcript_before
        and bool(turn.get("status_line")),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", default=str(REPO_ROOT))
    ap.add_argument("--commands", default=None, help="comma-separated override")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", default=None, help="also write the report here")
    ap.add_argument(
        "--clear-canary",
        action="store_true",
        help="also run the /clear preservation canary in a throwaway workspace",
    )
    args = ap.parse_args(argv)

    commands = (
        [c.strip() for c in args.commands.split(",") if c.strip()]
        if args.commands
        else list(DEFAULT_COMMANDS)
    )

    rep = drive(args.workspace, commands)

    if args.clear_canary:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="hcli-clear-canary-") as tmp:
            rep["clear_canary"] = clear_canary(tmp)
        rep["all_visible"] = rep["all_visible"] and rep["clear_canary"]["ok"]

    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        for t in rep["turns"]:
            mark = "ok " if t["visible"] else "BLANK"
            print(
                f"{mark} {t['command']:<32} rendered={t['rendered_chars']:>6}ch "
                f"entries={t['rendered_entries']} returned={t['handler_returned_chars']}ch"
                + (f"  ERROR {t['error']}" if t["error"] else "")
            )
        cc = rep.get("clear_canary")
        if cc:
            print(
                f"{'ok ' if cc['ok'] else 'FAIL'} /clear canary            "
                f"transcript {cc['transcript_entries_before']}->{cc['transcript_entries_after']}, "
                f"preserved={cc['preserved']}"
            )
        if rep["invisible_commands"]:
            print("\nBLANK IN THE REAL TUI: " + ", ".join(rep["invisible_commands"]))

    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"receipt: {p}")

    return 0 if rep["all_visible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
