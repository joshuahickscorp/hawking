"""Regression coverage for the retired Grok command adapter.

The historical bridge remains readable for old receipts, but it is not a
public Hawking command surface. These tests protect the consolidation so
future command changes do not accidentally resurrect the old owner.
"""
from __future__ import annotations

from hawking.commands import CommandHandler


class _Controller:
    workspace_root = "."


def test_grok_command_is_retired_and_does_not_dispatch():
    handler = CommandHandler(_Controller())

    result = handler.handle("/grok delegate legacy-task contract.md")

    assert "adapter is retired" in result
    assert "Hawking owns cognition" in result
    assert "/mission" in result
    assert "/cognition" in result


def test_help_does_not_advertise_grok_as_a_current_command():
    result = CommandHandler(_Controller()).handle("/help")

    assert "/grok" not in result
    assert "/mission" in result
    assert "/cognition" in result
