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
from pathlib import Path

# The interpreters on this host (Framework python, /usr/bin/python3) have no
# pytest-xdist and PEP 668 refuses to install it into them, so this runner used
# to fall back to SERIAL for everyone -- the documented fast path never ran.
# workspace/ops/.venv is a --system-site-packages venv that adds xdist and
# nothing else; re-exec into it rather than asking each caller to remember.
_XDIST_VENV = Path(__file__).resolve().parents[1] / "workspace/ops/.venv/bin/python"


def _reexec_into_xdist_venv() -> None:
    """Hand off to the venv that has xdist. No-op if we are already there."""
    if os.environ.get("HAWKING_NO_XDIST_VENV"):
        return
    if not _XDIST_VENV.exists() or Path(sys.executable).resolve() == _XDIST_VENV.resolve():
        return
    try:
        import xdist  # noqa: F401
    except ImportError:
        os.execv(str(_XDIST_VENV), [str(_XDIST_VENV), __file__, *sys.argv[1:]])


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        args = ["tools"]
    n = os.environ.get("PYTEST_XDIST_N") or str(os.cpu_count() or 8)
    _reexec_into_xdist_venv()
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
        # loadfile, NOT worksteal. worksteal scatters one file's tests across
        # workers, and every xdist worker is its own process -- so a module-scoped
        # cache is rebuilt once per worker instead of once. Measured on the three
        # heaviest tools/future files at n=6: loadfile 435s, worksteal >600s (budget
        # exceeded) on the identical selection. There are 569 test files under tools/
        # and at most a few dozen workers, so per-file granularity still balances.
        dist = os.environ.get("PYTEST_XDIST_DIST", "loadfile")
        xdist_args = ["-n", n, "--dist", dist]
    import pytest

    return int(pytest.main([*xdist_args, *args]))


if __name__ == "__main__":
    raise SystemExit(main())
