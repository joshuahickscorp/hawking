"""Every tool the registry discovers must be invokable.

A capability nothing can call does not exist. This repo once had 41 registered
typed tools with zero call sites. This check is narrower and mechanical:
``registry.invoke(name, {})`` must not return ``UNKNOWN_TOOL`` for any name
that ``registry.discover()`` returned.

``INVALID_ARGUMENTS`` and ``PERMISSION_DENIED`` are not failures of this check
— they prove the tool exists and its contract is being enforced. Only
``UNKNOWN_TOOL`` means a registered name has no reachable handler.

COSTLY tools are detected as registered without their handlers running.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from hcli.tool_registry import COSTLY, ToolContext, ToolSpec, default_tool_registry

REPO = Path(__file__).resolve().parents[2]
TEST_PATH = "tools/future/test_tool_reachability.py"
RECEIPT_PATH = REPO / "receipts" / "future" / "BENCH_B_TOOL_REACHABILITY.json"
GHOST = "ghost.no_handler"


def _registry():
    return default_tool_registry(REPO, repo_root=REPO)


def probe(registry, name: str):
    """``invoke(name, {})`` without running a live handler.

    UNKNOWN_TOOL is decided before schema and permission checks. Clearing the
    permission set makes a reachable tool fail ``PERMISSION_DENIED`` (or
    ``INVALID_ARGUMENTS`` if the schema already rejects ``{}``) rather than
    executing. COSTLY is therefore detected as registered and never run.
    """
    saved = registry.context
    registry.context = ToolContext(
        saved.workspace,
        saved.repo_root,
        saved.mission_root,
        permissions=frozenset(),
    )
    try:
        return registry.invoke(name, {})
    finally:
        registry.context = saved


def unknown_tool_failures(registry, names):
    """Names that discover listed and invoke cannot find."""
    unknown = []
    for name in names:
        result = probe(registry, name)
        if result.failure_class == "UNKNOWN_TOOL":
            unknown.append(name)
    return unknown


def check_discovered_tools(registry, names=None):
    """The load-bearing check. Fails if any discovered name is UNKNOWN_TOOL."""
    if names is None:
        names = [item["name"] for item in registry.discover()]
    unknown = unknown_tool_failures(registry, names)
    assert unknown == [], unknown
    return names


def test_every_discovered_tool_is_invokable():
    """``discover()`` is not a catalogue. Every name must survive ``invoke``."""
    registry = _registry()
    discovered = registry.discover()
    names = [item["name"] for item in discovered]
    assert names, "the default registry must discover tools"

    costly = []
    for item in discovered:
        spec = registry.get(item["name"])
        assert spec is not None, item["name"]
        if spec.mutation == COSTLY:
            costly.append(item["name"])

    names = check_discovered_tools(registry, names)

    for name in costly:
        result = probe(registry, name)
        assert result.ok is False, name
        assert result.failure_class != "UNKNOWN_TOOL", (name, result.failure_class)
        assert result.failure_class in {"INVALID_ARGUMENTS", "PERMISSION_DENIED"}, (
            name,
            result.failure_class,
        )

    print(f"checked {len(names)} tools")
    _write_receipt(
        tools_checked=len(names),
        unknown_tool_failures=[],
        costly_registered_not_executed=costly,
        names=names,
    )


def test_the_check_fails_when_a_discovered_name_has_no_handler():
    """Negative control: a catalogue entry with no handler is red, not green.

    Register a tool so discover lists it, then remove it from the handler map.
    The check must then fail with UNKNOWN_TOOL for that name. A check that
    stayed green here would not be evidence.
    """
    registry = _registry()
    registry.register(
        ToolSpec(
            GHOST,
            "constructed: discoverable, then the handler mapping is removed",
            {"type": "object", "additionalProperties": False, "properties": {}},
        )
    )
    names = [item["name"] for item in registry.discover()]
    assert GHOST in names

    del registry._tools[GHOST]
    assert registry.get(GHOST) is None
    result = registry.invoke(GHOST, {})
    assert result.ok is False
    assert result.failure_class == "UNKNOWN_TOOL"

    fired = False
    try:
        check_discovered_tools(registry, names)
    except AssertionError as exc:
        fired = True
        assert GHOST in str(exc)
    assert fired, "the check stayed green with a missing handler"

    unknown = unknown_tool_failures(registry, names)
    assert GHOST in unknown


def _write_receipt(*, tools_checked, unknown_tool_failures, costly_registered_not_executed, names):
    recorded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc = {
        "commands": [
            f"python3 -m pytest {TEST_PATH} -q",
            f"python3 {TEST_PATH}",
        ],
        "costly_registered_not_executed": costly_registered_not_executed,
        "names": names,
        "negative_control": (
            f"register {GHOST}, snapshot discover(), delete it from the handler "
            "map, then assert check_discovered_tools raises because invoke "
            "returns UNKNOWN_TOOL for that name"
        ),
        "recorded_at": recorded_at,
        "test_path": TEST_PATH,
        "tools_checked": tools_checked,
        "unknown_tool_failures": unknown_tool_failures,
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all green")
