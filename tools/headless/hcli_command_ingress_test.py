#!/usr/bin/env python3
"""Protected live-ingress checks. Enters through App._handle_input / Controller.

Run:
    python3 tools/headless/hcli_command_ingress_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("GROK_DRYRUN", "1")

REPO = Path(__file__).resolve().parents[2]

from hcli.app import App  # noqa: E402
from hcli.commands import REQUIRED_COMMANDS  # noqa: E402
from hcli.controller import Controller  # noqa: E402
from hcli.events import EventBus  # noqa: E402
from hcli.grok_bridge import GrokBridge, GrokRunHandle  # noqa: E402
from hcli.tui import TUI  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}: {detail}")
        FAILS.append(f"{name}: {detail}")


def check_required_commands() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = Controller(workspace=tmp, runtime_count=1)
        handler = ctrl.dispatcher()
        missing = [
            cmd
            for cmd in REQUIRED_COMMANDS
            if not callable(getattr(handler, f"_cmd_{cmd[1:]}", None))
        ]
        check("required-wired", missing == [], f"missing={missing}")
        unknown = ctrl.handle_command("/help")
        check(
            "help-via-controller",
            isinstance(unknown, str) and "/grok" in unknown and "/clear" in unknown,
            repr(unknown)[:200],
        )


def check_tui_grok() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        app = App(workspace=tmp, runtime_count=1)
        seen = []

        def consult(self, prompt, **kwargs):
            seen.append(prompt)
            return GrokRunHandle(
                task_id="tui-consult",
                command_run=["grok-run", "consult"],
                started_at="now",
                mode="consult",
                dry_run=True,
            )

        with patch.object(GrokBridge, "consult", consult):
            out = app._handle_input("/grok consult nonce-ingress")
        check("tui-grok-reached", seen == ["nonce-ingress"], f"seen={seen!r} out={out!r}")


def check_clear() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        bus = EventBus()
        tui = TUI(bus, str(ws), "m", 1)
        bus.subscribe(tui._on_event)
        ctrl = Controller(workspace=str(ws), runtime_count=1, bus=bus)
        ctrl.session.messages = [{"role": "user", "content": "x"}]
        tui.transcript = ["stale"]
        created = ctrl.start_ultragoal("clear must not forget this goal")
        (ws / ".hcli" / "receipts").mkdir(parents=True, exist_ok=True)
        keep = ws / ".hcli" / "receipts" / "r.json"
        keep.write_text("{}", encoding="utf-8")
        ctrl.handle_command("/clear")
        check("clear-messages", ctrl.session.messages == [], repr(ctrl.session.messages))
        check("clear-tui", tui.transcript == [], repr(tui.transcript))
        check("clear-goal", ctrl.session.goal == created["goal"], ctrl.session.goal)
        check(
            "clear-mission",
            ctrl.mission is not None and ctrl.mission.id == created["mission_id"],
            getattr(ctrl.mission, "id", None),
        )
        check("clear-receipt", keep.is_file())
        check("clear-dag", (ws / ".hcli" / "dag.json").is_file())


CHECKS = [
    ("required-commands", check_required_commands),
    ("tui-grok", check_tui_grok),
    ("clear", check_clear),
]


def main() -> int:
    FAILS.clear()
    for name, fn in CHECKS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            FAILS.append(f"{name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    if FAILS:
        print(f"\n{len(FAILS)} FAILED")
        for item in FAILS:
            print("  " + item)
        return 1
    print("\nall hcli command ingress checks passed")
    return 0


def test_hcli_command_ingress() -> None:
    rc = main()
    assert rc == 0, f"{len(FAILS)} ingress checks failed: {FAILS}"


if __name__ == "__main__":
    sys.exit(main())
