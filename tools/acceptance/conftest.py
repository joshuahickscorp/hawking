"""Put the primary checkout on sys.path so sparse worktrees can import hcli."""
from __future__ import annotations

from tools.acceptance.lake.common import ensure_hcli_path, ensure_tools_path


def pytest_configure(config):  # noqa: ARG001
    ensure_tools_path()
    try:
        ensure_hcli_path()
    except FileNotFoundError:
        pass
