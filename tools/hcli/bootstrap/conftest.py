"""Ignore historical haider bootstrap snapshots.

Those files are sealed science (names like ``.stale.`` and
``pre-fast-selfhost``). They are not live tests; collecting them under
``pytest tools/hcli/bootstrap`` raises ImportError on fossil module names.
"""
from __future__ import annotations


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    try:
        parts = collection_path.parts
    except AttributeError:
        return None
    if "bootstrap_snapshots" in parts:
        return True
    return None
