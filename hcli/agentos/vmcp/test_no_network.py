"""Negative control: hcli/agentos/vmcp must not grow a network client."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORBIDDEN_MODULES = frozenset(
    {"socket", "urllib", "urllib.request", "urllib.parse", "requests", "http.client", "http.server", "aiohttp"}
)
FORBIDDEN_NAMES = frozenset({"socket", "urllib", "requests", "http"})


def test_no_network_imports_in_tools_vmcp():
    offenders: list[str] = []
    for path in ROOT.glob("*.py"):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if alias.name in FORBIDDEN_MODULES or top in FORBIDDEN_NAMES:
                        offenders.append(f"{path.name}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if node.module in FORBIDDEN_MODULES or top in FORBIDDEN_NAMES:
                    offenders.append(f"{path.name}:{node.lineno}:from {node.module}")
    assert offenders == []
