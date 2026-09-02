"""Headless pytest collection.

Some tests here import `hcli`. In a sparse-checkout worktree the package may
not be materialized; collecting the directory then dies with
ModuleNotFoundError before any test runs. Skip those modules only when the
package is genuinely unimportable, so `pytest tools/headless -q` still
exercises the tests that do live here.

This used to gate on the directory `tools/haider/hcli` existing. That path is
gone, so the check would have been permanently False and every module below
would have been skipped forever -- green, and testing nothing. Ask the real
question instead: can `hcli` be imported?
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _hcli_importable() -> bool:
    try:
        return importlib.util.find_spec("hcli") is not None
    except (ImportError, ValueError):
        return False

# Files that import hcli in the test body even when collection succeeds.
_HCLI_BODY = {
    "rollback_integrity_test.py",
    "handoff_cold_read_test.py",
}


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    # firstresult=True: False would stop parent hooks (tools/conftest.py)
    # from ignoring name-colliding modules under `pytest tools/`. None
    # means "no opinion" so a parent can still vote.
    if _hcli_importable():
        return None
    try:
        name = collection_path.name
    except AttributeError:
        return None
    if name.startswith("hcli_") and name.endswith(".py"):
        return True
    if name in _HCLI_BODY:
        return True
    return None
