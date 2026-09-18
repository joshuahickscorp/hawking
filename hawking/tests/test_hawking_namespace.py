"""The public control and import vocabulary is Hawking-first."""
from __future__ import annotations

import argparse
import importlib.util

import hawking
from hawking.control import CONTROL_VERBS, build_parser


def _top_level_commands(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("control parser has no subcommands")


def test_hawking_root_forwards_the_orchestration_surface_lazily():
    from hawking import AgentOS, Hawking, ResidentDaemon, run_matrix

    assert Hawking is AgentOS
    assert callable(ResidentDaemon)
    assert callable(run_matrix)
    assert "Hawking" in dir(hawking)


def test_control_parser_and_root_dispatch_share_one_command_vocabulary():
    assert _top_level_commands(build_parser()) == set(CONTROL_VERBS)


def test_retired_parallel_perception_names_have_no_importable_package():
    assert importlib.util.find_spec("hawking.vmcp") is None
    assert importlib.util.find_spec("hawking.agentos.vmcp") is None
