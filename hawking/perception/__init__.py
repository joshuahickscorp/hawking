"""Hawking-native perception kernel.

Host/store tools and local perception organs live together here.  The exports
are lazy so importing Hawking's namespace does not initialize file parsing,
terminal probing, or the project store until a caller needs one.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "Host": (".host", "Host"),
    "RegisteredTool": (".host", "RegisteredTool"),
    "ToolError": (".host", "ToolError"),
    "ToolManager": (".host", "ToolManager"),
    "ProjectStore": (".store", "ProjectStore"),
    "TOOL_NAMES": (".tools", "TOOL_NAMES"),
    "create_server": (".tools", "create_server"),
    "list_profiles": (".tools", "list_profiles"),
    "public_api_versions": (".tools", "public_api_versions"),
    "observe": (".file_eye", "observe"),
    "classify_bytes": (".file_eye", "classify_bytes"),
    "capture": (".terminal", "capture"),
    "probe": (".terminal", "probe"),
    "profile": (".doctor", "profile"),
    "report": (".doctor", "report"),
    "see": (".acts", "see"),
    "hold": (".acts", "hold"),
    "know": (".acts", "know"),
    "check": (".acts", "check"),
    "prove": (".acts", "prove"),
    "compact_surface": (".acts", "compact_surface"),
    "disposition": (".acts", "disposition"),
    "selftest": (".acts", "selftest"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
