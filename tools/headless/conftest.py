"""Headless pytest collection.

HCLI tests import `hcli` from tools/haider. This worktree is a sparse checkout
and that path is often not materialized; collecting the directory then dies
with ModuleNotFoundError before any test runs. Skip those modules only when
the package is actually absent, so `pytest tools/headless -q` can still
exercise the tests that do live here.
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_HAIDER = _REPO / "tools" / "haider" / "hcli"

# Files that import hcli in the test body even when collection succeeds.
_HCLI_BODY = {
    "rollback_integrity_test.py",
    "handoff_cold_read_test.py",
}


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    if _HAIDER.is_dir():
        return False
    try:
        name = collection_path.name
    except AttributeError:
        return False
    if name.startswith("hcli_") and name.endswith(".py"):
        return True
    if name in _HCLI_BODY:
        return True
    return False
