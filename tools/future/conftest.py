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


# Longest-processing-time-first. Under `--dist loadfile` an xdist worker claims a
# whole file and workers pull as they free up, so a file that takes nine minutes
# and is claimed LAST runs alone against idle cores. Hoisting the known-heavy
# files to the front lets one worker start the long pole immediately while the
# other twenty-three chew through everything else beside it.
#
# This is packing, not coverage: the same tests run, in the same processes, with
# the same assertions. Measured: the suite wall was 670s while its longest single
# file was 559s, i.e. ~110s where the long pole was not yet running.
#
# Names, not durations, because a duration table goes stale silently and would
# have to be regenerated to stay honest. A file that is no longer slow merely
# loses a scheduling hint; nothing breaks.
# ONE file, not eight. Hoisting eight heavy files made the suite SLOWER --
# 687.96s against 670.42s -- because eight workers then start heavy files
# simultaneously and rival_codec's own eight-thread eigh pool contends with all
# of them; its own duration rose 558.89s -> 565.81s. Only the true long pole
# needs the head start; everything else schedules better on its own.
_SLOW_FIRST = ("test_rival_codec_screen.py",)


def _slow_rank(nodeid: str) -> int:
    for i, name in enumerate(_SLOW_FIRST):
        if name in nodeid:
            return i
    return len(_SLOW_FIRST)


def pytest_collection_modifyitems(session, config, items):  # noqa: ARG001
    """Heavy files first, the integration-attack aggregator last.

    The aggregator must stay last so SESSION_SKIP_LINES is complete when it runs;
    everything else is ordered longest-first so parallel packing does not leave
    the long pole for the tail.
    """
    last = []
    rest = []
    for item in items:
        nodeid = getattr(item, "nodeid", "") or ""
        if nodeid.endswith("test_runs_and_emits_sealed_receipt"):
            last.append(item)
        else:
            rest.append(item)
    # Stable sort: files keep their internal order, which loadfile relies on.
    rest.sort(key=lambda it: _slow_rank(getattr(it, "nodeid", "") or ""))
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
