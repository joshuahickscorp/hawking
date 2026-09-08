"""A minimal in-process tool host, hawking-native.

This replaces the one thing hcli/vmcp_adapter.py needed from the foreign
package: `create_server(projects_root, profile)` returning an object whose
`_tool_manager.get_tool(name)` yields something with an awaitable
`run(arguments)`. That is the entire calling convention -- 40 lines of it, not
a 299k-line dependency.

Deliberately not an MCP server. Nothing here speaks stdio or JSON-RPC, because
nothing in hawking ever asked it to: call_vmcp only ever constructed a host in
process and called one tool on it.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


class ToolError(RuntimeError):
    """Raised when a tool body fails. Preserves the original as __cause__."""


@dataclass
class RegisteredTool:
    fn: Callable[..., Any]
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    is_async: bool = False
    inputSchema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.inputSchema is None:
            self.inputSchema = self.parameters
        self.is_async = inspect.iscoroutinefunction(self.fn)

    async def run(self, arguments: dict[str, Any] | None = None,
                  context: Any = None, convert_result: bool = False) -> Any:
        args = dict(arguments or {})
        try:
            out = self.fn(**args)
            if inspect.isawaitable(out):
                out = await out
            return out
        except Exception as exc:  # noqa: BLE001 - re-raised with the original attached
            raise ToolError(f"{self.name}: {exc}") from exc


class ToolManager:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def add_tool(self, tool: RegisteredTool) -> None:
        self._tools[tool.name] = tool

    def remove_tool(self, name: str) -> None:
        self._tools.pop(name, None)

    def get_tool(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[RegisteredTool]:
        return list(self._tools.values())


class Host:
    """What create_server() returns. `_tool_manager` is the accessed surface."""

    def __init__(self, projects_root, profile: str = "core") -> None:
        self.projects_root = projects_root
        self.profile = profile
        self._tool_manager = ToolManager()

    def tool(self, name: str, description: str = "",
             parameters: dict[str, Any] | None = None) -> Callable:
        def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._tool_manager.add_tool(RegisteredTool(
                fn=fn, name=name, description=description or (fn.__doc__ or "").strip(),
                parameters=parameters or {}))
            return fn
        return wrap

    def tool_names(self) -> list[str]:
        return sorted(t.name for t in self._tool_manager.list_tools())
