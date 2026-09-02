"""Session artifacts for the roadmap suite.

The capability graph and python-facts dump are built once per pytest
session (and shared across xdist workers via a temp dir keyed by the
controller pid). A mutated auditor.py changes the code digest and the
cache is not reused — that is the load-bearing property the mutation
check relies on.
"""
from __future__ import annotations

import os


def pytest_configure(config):  # noqa: ARG001
    os.environ.setdefault("ROADMAP_ARTIFACT_SESSION", str(os.getpid()))
