"""One machine-readable table of the HCLI slash commands.

Both the completion list (``hcli.commands.REQUIRED_COMMANDS``) and ``/help``
render from ``COMMANDS``. Before this module there were two hand-maintained
lists and they had already drifted: the tuple was missing ``/tools``,
``/provider`` and ``/flash-next`` while the help text advertised all three --
and ``/flash-next`` did not dispatch at all, because a dash cannot appear in
the ``_cmd_<name>`` identifier the dispatcher looks up.

``authority`` reuses the mutation-class vocabulary of
``hcli.tool_registry.MUTATION_CLASSES`` so a command and a tool are graded on
one scale. A command with several verbs is graded at its *highest* verb:
``/grok`` is ``repo_write`` because ``/grok delegate`` writes code, even
though ``/grok status`` only reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Command:
    """One slash command. ``aliases`` are extra spellings of the same handler."""

    name: str
    help: str
    mutates: bool
    authority: str
    example: str
    aliases: Tuple[str, ...] = ()


COMMANDS: Tuple[Command, ...] = (
    Command("/help", "show this help", False, "read_only", "/help"),
    Command(
        "/status",
        "show session and machine status",
        False,
        "read_only",
        "/status",
    ),
    Command("/models", "list available models", False, "read_only", "/models"),
    Command(
        "/model",
        "select model",
        True,
        "reversible_runtime",
        "/model 2",
    ),
    Command("/tools", "list typed AgentOS tools", False, "read_only", "/tools"),
    Command(
        "/provider",
        "show the selected provider profile",
        False,
        "read_only",
        "/provider",
    ),
    Command(
        "/flash-next",
        "show the pinned Flash-Next acquisition identity",
        False,
        "read_only",
        "/flash-next",
    ),
    Command(
        "/receipts",
        "list durable run receipts newest first",
        False,
        "read_only",
        "/receipts 5",
    ),
    Command(
        "/processes",
        "what every live Hawking process is, and which are safe to stop",
        False,
        "read_only",
        "/processes",
    ),
    Command("/goal", "set active goal", True, "workspace_write", "/goal ship X"),
    Command(
        "/bank",
        "queue a future goal; it starts after the active goal completes",
        True,
        "workspace_write",
        "/bank prepare the overnight production report",
        aliases=("\\bank",),
    ),
    Command(
        "/ultragoal",
        "create or show the durable Goal + ledger + DAG",
        True,
        "workspace_write",
        "/ultragoal ship X with evidence",
    ),
    Command(
        "/mission",
        "run a persistent mission",
        True,
        "repo_write",
        "/mission ship X",
    ),
    Command(
        "/steer",
        "queue steering instruction",
        True,
        "workspace_write",
        "/steer prefer the smaller diff",
    ),
    Command(
        "/grok",
        "delegate, audit, consult, or inspect a Grok task",
        True,
        "repo_write",
        "/grok consult is this contract testable",
    ),
    Command(
        "/cancel",
        "cancel the active mission",
        True,
        "reversible_runtime",
        "/cancel",
        aliases=("/stop",),
    ),
    Command(
        "/context",
        "show context and prior knowledge; manage cached pastes",
        True,
        "destructive",
        "/context list",
    ),
    Command("/compact", "compact context", True, "workspace_write", "/compact"),
    Command(
        "/clear",
        "clear transcript (does not forget the mission)",
        True,
        "workspace_write",
        "/clear",
    ),
    Command("/resume", "resume session", True, "workspace_write", "/resume"),
    Command(
        "/quit",
        "exit HCLI",
        True,
        "reversible_runtime",
        "/quit",
        aliases=("/exit",),
    ),
    Command(
        "/land",
        "commit accumulated work via the governed landing service "
        "(push/merge are separate: /land push, /land merge <branch>)",
        True,
        "repo_write",
        "/land",
    ),
)


def command_names() -> Tuple[str, ...]:
    """Every spelling the dispatcher must answer, aliases included."""
    names = []
    for command in COMMANDS:
        names.append(command.name)
        names.extend(command.aliases)
    return tuple(names)


def handler_name(name: str) -> str:
    """The ``CommandHandler`` attribute a command name dispatches to.

    Identical to the lookup in ``CommandHandler.handle``, so a test over this
    is a test of the real dispatch path.
    """
    return f"_cmd_{name.lstrip('/\\')}"


def help_text() -> str:
    """The body of ``/help``. The only place command help is worded."""
    width = max(len(command.name) for command in COMMANDS)
    lines = ["Commands:"]
    for command in COMMANDS:
        alias = f"  (also {' '.join(command.aliases)})" if command.aliases else ""
        lines.append(f"  {command.name:<{width}} - {command.help}{alias}")
    return "\n".join(lines)
