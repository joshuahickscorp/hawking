"""No network egress, credentials, scanning, or payload generation in the engine."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools.theia.bounty import BountyClass
from tools.theia.labs import LABS, LabKind

HERE = Path(__file__).resolve().parent

FORBIDDEN_IMPORTS = frozenset(
    {
        "socket",
        "http",
        "http.client",
        "http.server",
        "urllib",
        "urllib.request",
        "urllib.client",
        "urllib3",
        "requests",
        "httpx",
        "aiohttp",
        "paramiko",
        "ftplib",
        "smtplib",
        "ssl",
        "nacl",
    }
)


def _py_files():
    return [p for p in HERE.glob("*.py") if not p.name.startswith("test_")]


def test_seventeen_closed_bounty_classes():
    assert len(BountyClass) == 17
    values = {c.value for c in BountyClass}
    assert "Hawking internal self-bounty" in values
    assert "authorized bug-bounty program" in values
    assert "CTF / intentionally vulnerable lab" in values


def test_six_laboratories_registered():
    assert set(LABS) == set(LabKind)
    sec = LABS[LabKind.AUTHORIZED_SECURITY]
    assert "ACTIVE_TEST" in sec.refused_work
    assert "scan" in sec.refused_work
    assert "payload_generation" in sec.refused_work
    assert "network_egress" in sec.refused_work
    assert "credential_handling" in sec.refused_work
    assert "ACTIVE_TEST" not in sec.executable_work
    assert LABS[LabKind.MATH_FORMAL].executable_work
    assert LABS[LabKind.SYSTEMS_COMPILER].executable_work
    assert LABS[LabKind.HAWKING_SELF_BOUNTY].executable_work
    assert LABS[LabKind.PHYSICS_QUANTUM].executable_work == ()
    assert LABS[LabKind.OPEN_SOURCE].executable_work == ()
    assert "ACTIVE_TEST" not in LABS[LabKind.MATH_FORMAL].executable_work
    assert "ACTIVE_TEST" not in LABS[LabKind.SYSTEMS_COMPILER].executable_work


def test_package_has_no_network_or_scan_imports():
    found = []
    for path in _py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if alias.name in FORBIDDEN_IMPORTS or root in FORBIDDEN_IMPORTS:
                        found.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module in FORBIDDEN_IMPORTS or node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                    found.append(f"{path.name}: from {node.module}")
    assert found == []


def test_no_subprocess_or_urlopen_calls():
    found = []
    for path in _py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {
                "urlopen",
                "urlretrieve",
                "create_connection",
                "getaddrinfo",
            }:
                found.append(f"{path.name}:{node.lineno}:{node.attr}")
            if isinstance(node, ast.Name) and node.id in {"Popen", "urlopen"}:
                found.append(f"{path.name}:{node.lineno}:{node.id}")
    assert found == []
