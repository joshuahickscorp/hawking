#!/usr/bin/env python3
"""Opt-in xdist runner for tools/ tests.

Not wired into pytest addopts: addopts applies to every interpreter that
runs this repo, including ones without pytest-xdist. See pyproject.toml.

Usage:
    python3 tools/run_tools_pytest.py tools/roadmap
    python3 tools/run_tools_pytest.py tools
    PYTEST_XDIST_N=8 python3 tools/run_tools_pytest.py tools/audit
"""
from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        args = ["tools"]
    n = os.environ.get("PYTEST_XDIST_N") or str(os.cpu_count() or 8)
    try:
        import xdist  # noqa: F401
    except ImportError:
        print(
            "pytest-xdist is not installed in this interpreter; "
            "running without -n (see pyproject.toml)",
            file=sys.stderr,
        )
        xdist_args: list[str] = []
    else:
        xdist_args = ["-n", n, "--dist", "worksteal"]
    import pytest

    return int(pytest.main([*xdist_args, *args]))


if __name__ == "__main__":
    raise SystemExit(main())
