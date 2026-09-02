"""Parity between the AST reachability path and hawking-index python-facts.

A fast path that silently changes a verdict is worse than a slow one.
These tests pin the index extractor to the same adversarial rules as
capability_reachability, on a synthetic overlay that does not need a 45s
repo walk.
"""
from __future__ import annotations

import os

import pytest

from tools.roadmap.catalog import GATES
from tools.roadmap.gitfs import REPO, SourceView
from tools.roadmap import index_client
from tools.roadmap import reach


SYN_DEF = """\
class Scheduler:
    pass

NO_PROGRESS = 3

def abort():
    pass
"""

SYN_CALLER = """\
from syn.mod import Scheduler, abort
import syn.mod as m

def run():
    s = Scheduler()
    abort()
    m.abort()
    x = Scheduler  # name-only, not a call
"""

SYN_SUB = """\
import subprocess
subprocess.run(["syn/mod.py"])
subprocess.run(["tools/hcli/bootstrap/syn/mod.py"])
"""


@pytest.fixture
def overlay_view():
    view = SourceView()
    view.overlay["syn/mod.py"] = SYN_DEF
    view.overlay["syn/caller.py"] = SYN_CALLER
    view.overlay["syn/launch.py"] = SYN_SUB
    return view


def _probe():
    return {
        "code_paths": ["syn/mod.py"],
        "modules": ["syn.mod"],
        "symbols": [{"module": "syn.mod", "symbol": "Scheduler"}],
        "receipt_globs": [],
    }


def test_index_client_classifies_assignment_vs_class():
    ff = {
        "definitions": [
            {"name": "NO_PROGRESS", "kind": "assignment", "line": 4, "scope": "module"},
            {"name": "Scheduler", "kind": "class", "line": 1, "scope": "module"},
            {"name": "abort", "kind": "function", "line": 6, "scope": "module"},
        ]
    }
    kind, line = index_client.classify_from_facts(ff, "NO_PROGRESS")
    assert kind == "assignment" and line == 4
    kind, line = index_client.classify_from_facts(ff, "Scheduler")
    assert kind == "class" and line == 1
    kind, line = index_client.classify_from_facts(ff, "abort")
    assert kind == "function" and line == 6


def test_exact_cli_path_still_rejects_suffix():
    assert reach.is_exact_cli_path("syn/mod.py", "syn/mod.py")
    assert not reach.is_exact_cli_path("syn/mod.py", "tools/hcli/bootstrap/syn/mod.py")


@pytest.mark.skipif(index_client.find_index_bin() is None, reason="hawking-index-query not built")
def test_index_and_ast_agree_on_synthetic_overlay(overlay_view):
    probe = _probe()
    unique = {"syn/mod.py"}
    os.environ["ROADMAP_REACH_BACKEND"] = "ast"
    try:
        ast_look = reach.scan_probe_ast(overlay_view, probe, unique_paths=unique)
        os.environ["ROADMAP_REACH_BACKEND"] = "index"
        # Overlay-only dump: this test's corpus is the three synthetic files,
        # not a 20s HEAD walk. The rust extractor still runs on that corpus.
        overlay_view._python_facts = None  # type: ignore[attr-defined]
        index_client.load_python_facts(overlay_view, git_head=False)
        idx_look = reach.scan_probe(overlay_view, probe, unique_paths=unique)
    finally:
        os.environ.pop("ROADMAP_REACH_BACKEND", None)

    ast_calls = {(s["file"], s["line"], s["kind"]) for s in ast_look["runtime_caller"]}
    idx_calls = {(s["file"], s["line"], s["kind"]) for s in idx_look["runtime_caller"]}
    assert ast_calls == idx_calls, (ast_look["runtime_caller"], idx_look["runtime_caller"])
    assert ast_look["defined"] == idx_look["defined"]
    ast_imp = {(s["file"], s["line"]) for s in ast_look["import_sites"]}
    idx_imp = {(s["file"], s["line"]) for s in idx_look["import_sites"]}
    assert ast_imp == idx_imp, (ast_look["import_sites"], idx_look["import_sites"])


def test_catalog_still_has_71_gates():
    assert len(GATES) == 71


def test_sibling_resolution_uses_known_files_not_disk():
    """Sibling-import idiom must not stat the worktree (sparse hcli/ is absent)."""
    imp = {
        "form": "from",
        "module": "_common",
        "level": 0,
        "names": [{"name": "REPO", "asname": None}],
    }
    targets, binds = index_client.import_targets_and_binds(
        "syn/caller.py",
        imp,
        known_files={"syn/caller.py", "syn/_common.py"},
    )
    assert "syn._common" in targets
    assert "syn._common.REPO" in targets
    assert ("REPO", "syn._common.REPO") in binds
    # A sibling that is not in known_files is not invented from disk.
    targets_none, _ = index_client.import_targets_and_binds(
        "syn/caller.py",
        imp,
        known_files={"syn/caller.py"},
    )
    assert "syn._common" not in targets_none


def test_sparse_hcli_absent_from_disk():
    """HEAD blob for hcli/scheduler.py counts as defined without a worktree file.

    The first assertion used to require *this process's worktree* to be sparse
    (`assert not (REPO / "hcli").is_dir()`). That is an environment check, not
    a SourceView check: on a full checkout (the primary tree, CI) hcli/ is on
    disk and the test failed even when HEAD-backed reads were correct. The
    contract is the blob, not the cone.
    """
    rel = "hcli/scheduler.py"
    view = SourceView()
    assert view.exists(rel), "HEAD blob for hcli/scheduler.py must still count as defined"
    text = view.read(rel)
    assert "class Scheduler" in text or "Scheduler" in text
    if not (REPO / rel).is_file():
        assert not (REPO / "hcli").is_dir()


@pytest.mark.skipif(index_client.find_index_bin() is None, reason="hawking-index-query not built")
def test_python_facts_carry_head_commit_and_stay_inside_blob():
    """Facts are parsed from HEAD blobs. A line past EOF at HEAD is a bug."""
    from tools.roadmap.gitfs import blob_text, head_commit

    view = SourceView()
    view._python_facts = None  # type: ignore[attr-defined]
    dump = index_client.load_python_facts(view)
    sha = head_commit()
    assert dump.get("commit") == sha, dump.get("commit")
    rel = "hcli/controller.py"
    ff = index_client.file_facts(dump, rel)
    assert ff is not None, "HEAD blob for hcli/controller.py must be indexed"
    assert ff.get("commit") == sha
    text = blob_text(sha, rel) or ""
    n = len(text.splitlines())
    assert n > 0
    for d in ff.get("definitions") or []:
        line = int(d.get("line") or 0)
        assert 1 <= line <= n, f"{rel}:{line} exceeds {n} lines at {sha} ({d})"
    for c in ff.get("calls") or []:
        line = int(c.get("line") or 0)
        assert 1 <= line <= n, f"{rel} call {line} exceeds {n} lines at {sha}"
