#!/usr/bin/env python3.12
"""CLI wiring tests: every documented successor command is parseable and dispatched."""
import pathlib
import sys

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import succ_cli as cli  # noqa: E402

# The master goal (section 6.2 / 20) mandates these commands exist and do something real.
REQUIRED = ["audit", "compile", "start", "status", "explain-next", "ping", "resume", "drain",
            "verify", "transition-status", "gc-plan", "queue", "arm-transition", "telegram"]


def test_all_required_commands_are_dispatched():
    for cmd in REQUIRED:
        assert cmd in cli.DISPATCH, f"command {cmd} declared but not wired to a handler"
        assert callable(cli.DISPATCH[cmd])


def test_parser_accepts_every_command():
    parser = cli.build_parser()
    for cmd in ("audit", "compile", "start", "status", "explain-next", "ping", "resume",
                "drain", "verify", "transition-status", "gc-plan"):
        ns = parser.parse_args([cmd])
        assert ns.cmd == cmd
    assert parser.parse_args(["queue", "--model", "72B"]).model == "72B"
    assert parser.parse_args(["arm-transition", "--intent", "/x.json"]).intent == "/x.json"
    assert parser.parse_args(["telegram", "status"]).telegram_action == "status"


def test_unknown_command_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["frobnicate"])


def test_no_documented_command_missing_from_parser():
    # every DISPATCH key must be a real subparser choice (no orphan handlers)
    parser = cli.build_parser()
    subparser_action = next(a for a in parser._actions if hasattr(a, "choices") and a.choices
                            and "compile" in a.choices)
    for cmd in cli.DISPATCH:
        assert cmd in subparser_action.choices, f"handler {cmd} has no parser subcommand"
