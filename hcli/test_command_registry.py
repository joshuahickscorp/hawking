"""The command surface has one source of truth and it cannot drift again.

The defect this locks down: ``REQUIRED_COMMANDS`` and the ``/help`` text were
two hand-maintained lists. Help advertised ``/tools``, ``/provider`` and
``/flash-next`` while the tuple listed none of them, and ``/flash-next`` did
not even dispatch -- the dash cannot appear in the ``_cmd_<name>`` identifier
the dispatcher looks up.

Runnable two ways:

    python3 -m pytest hcli/test_command_registry.py -q
    python3 hcli/test_command_registry.py
"""
from __future__ import annotations

from hcli.command_registry import COMMANDS, command_names, handler_name, help_text
from hcli.commands import REQUIRED_COMMANDS, CommandHandler


def test_help_and_completion_cannot_disagree():
    """The drift the registry exists to prevent, asserted in both directions."""
    advertised = {
        line.strip().split()[0]
        for line in help_text().splitlines()
        if line.startswith("  /")
    }
    completable = {name for name in REQUIRED_COMMANDS}
    aliases = {alias for command in COMMANDS for alias in command.aliases}
    assert advertised, help_text()
    # Aliases are completable without being advertised on their own line;
    # everything else must appear on both sides.
    assert advertised == completable - aliases, sorted(
        advertised.symmetric_difference(completable - aliases)
    )


def test_every_registered_command_dispatches():
    """Including /flash-next, whose dash used to make it unreachable."""
    handler = CommandHandler(None)
    missing = [
        name
        for name in command_names()
        if not callable(getattr(handler, handler_name(name), None))
    ]
    assert missing == [], missing


def test_handler_name_matches_the_real_dispatcher():
    """A test over handler_name is only useful if it is the same lookup."""
    handler = CommandHandler(None)
    assert handler.handle("/nonesuch") == "Unknown command: /nonesuch"
    assert handler_name("/flash-next") == "_cmd_flash-next"
    assert getattr(handler, "_cmd_flash-next") == handler._cmd_flash_next


def test_no_handler_is_unregistered():
    """A command that exists but is in no list is drift in the other direction."""
    wired = {handler_name(name) for name in command_names()}
    # `_cmd_flash_next` is the same function object as `_cmd_flash-next`, not a
    # second command, so compare functions rather than attribute names.
    reachable = {getattr(CommandHandler, attr) for attr in wired}
    orphans = [
        attr
        for attr in dir(CommandHandler)
        if attr.startswith("_cmd_")
        and attr not in wired
        and getattr(CommandHandler, attr) not in reachable
    ]
    assert orphans == [], orphans


def test_every_command_carries_its_metadata():
    for command in COMMANDS:
        assert command.name.startswith("/"), command
        assert command.help and not command.help.endswith("."), command
        assert isinstance(command.mutates, bool), command
        assert command.example.startswith(command.name), command


def test_authority_uses_the_tool_registry_vocabulary():
    """One scale for commands and tools, not a second private one."""
    from hcli.tool_registry import MUTATION_CLASSES

    unknown = sorted({c.authority for c in COMMANDS} - set(MUTATION_CLASSES))
    assert unknown == [], unknown


def test_read_only_commands_do_not_claim_to_mutate():
    for command in COMMANDS:
        if command.authority == "read_only":
            assert not command.mutates, command
        if command.mutates:
            assert command.authority != "read_only", command


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all green")
