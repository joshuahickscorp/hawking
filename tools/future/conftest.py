"""Sparse-checkout collection guard and session git artifacts.

``pytest tools/future`` imports every test module before ``-k`` runs.
Two modules fail at import when receipts/ or hcli/ are not materialized.
Ignore them only while those paths are absent so the noetic/ebpw filter
can run; they collect again when the tree is widened.

Session start prefetches HEAD's path list and receipts/future blobs with
one ls-tree + one cat-file --batch. Every producer was independently
`git show`ing the same receipts (the rebuild-the-world-per-test pattern);
the bytes are HEAD's, not a fixture standing in for a run. A mutated
_common.py in a new process does not reuse the memo.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

# Isolate the GPU lane lock from the live daemon before any test module
# imports a producer that would park 25 minutes on a live holder.
os.environ.setdefault(
    "HAWKING_GPU_LANE_LOCK",
    str(Path(tempfile.gettempdir()) / f"hawking-gpu-lane-pytest-{os.getpid()}.lock"),
)
os.environ.setdefault("FUTURE_ARTIFACT_SESSION", str(os.getpid()))

# Skip reports from THIS session. attack_actual_skips used to shell out to a
# nested `pytest tools/future/` (the whole suite, 20 minutes) to see whether
# any skip actually fired. The outer session is that suite; recording its
# skips is the same evidence, not a fixture standing in for a run.
SESSION_SKIP_LINES: list[str] = []


def pytest_configure(config):  # noqa: ARG001
    from tools.future._common import prefetch_session_artifacts

    prefetch_session_artifacts()


def pytest_runtest_logreport(report) -> None:
    if not getattr(report, "skipped", False):
        return
    if getattr(report, "wasxfail", None):
        return
    node = getattr(report, "nodeid", "") or ""
    detail = ""
    longrepr = getattr(report, "longrepr", None)
    if longrepr is not None:
        detail = str(longrepr).replace("\n", " ")[:180]
    SESSION_SKIP_LINES.append(f"SKIPPED {node} {detail}".strip())


def pytest_collection_modifyitems(session, config, items):  # noqa: ARG001
    """Run the integration-attack aggregator last so SESSION_SKIP_LINES is complete."""
    last = []
    rest = []
    for item in items:
        nodeid = getattr(item, "nodeid", "") or ""
        if nodeid.endswith("test_runs_and_emits_sealed_receipt"):
            last.append(item)
        else:
            rest.append(item)
    items[:] = rest + last


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    name = getattr(collection_path, "name", None) or Path(str(collection_path)).name
    if name == "test_path_to_71.py":
        a = _REPO / "receipts/future/SEALED_DEFAULT_ABSOLUTE.json"
        b = _REPO / "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"
        if not a.is_file() and not b.is_file():
            return True
    if name == "test_status_causality_gates.py":
        if not (_REPO / "hcli/agentos/resident_gate.py").is_file():
            return True
    return None
