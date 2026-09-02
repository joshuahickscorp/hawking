"""Tests for tools/future/capability_reachability.py.

Two layers: a hermetic engine test against a synthetic three-file fixture
(so the "own test doesn't count" rule is checked against a known-exact
answer, not against whatever the live repo happens to contain today), and a
handful of live-repo invariants that hold regardless of what code lands.

The hermetic fixture is the one that matters most: it is built to fail if
the analyzer ever reports a capability as callable when its only call site
is its own test -- exactly the modellake_events-shaped trap this module
exists to catch.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.future import capability_reachability as cr
from tools.future import _common as cr_common


# --------------------------------------------------------------------------
# Hermetic engine fixture: three synthetic files, a known-exact answer.
# --------------------------------------------------------------------------


def _write(root: Path, rel_path: str, content: str) -> Path:
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


@pytest.fixture
def fixture_repo(tmp_path, monkeypatch):
    """A tiny fake repo: tools/future/widget.py, its own test, and a real
    outside caller. Patches cr.REPO so rel()/module_name_of() resolve
    against tmp_path instead of the live repo, and clears the per-run text
    cache so fixtures from an earlier test never leak in."""
    monkeypatch.setattr(cr, "REPO", tmp_path)
    cr._TEXT_CACHE.clear()
    cr._TRACKED_PY = None
    cr._GIT_CHECKOUT = None
    cr._REL_MEMO.clear()
    widget = _write(
        tmp_path, "tools/future/widget.py",
        "def gadget():\n    return 1\n",
    )
    own_test = _write(
        tmp_path, "tools/future/test_widget.py",
        "from tools.future.widget import gadget\n\n"
        "def test_gadget():\n    assert gadget() == 1\n",
    )
    outside_caller = _write(
        tmp_path, "tools/future/caller.py",
        "from tools.future.widget import gadget\n\n"
        "def use_it():\n    return gadget() + 1\n",
    )
    return {
        "root": tmp_path,
        "widget": widget,
        "own_test": own_test,
        "caller": outside_caller,
    }


def test_only_own_test_calls_it_is_not_callable(fixture_repo):
    """The exact shape the wave named: a module imported by nothing but its
    own test must NOT be reported callable, even though it plainly has a
    call site (inside that test)."""
    files = [fixture_repo["widget"], fixture_repo["own_test"]]
    idx = cr.build_repo_index(files=files)
    sites = cr.find_symbol_call_sites(idx, "tools.future.widget", "gadget")
    cap = cr.build_capability(
        "tools.future.widget.gadget", "function",
        defined=True, registered=False, resident_visible=False, sites=sites,
    )
    assert cap["tested"] is True, "the own-test call site must still count toward tested"
    assert cap["callable"] is False, (
        "callable must be false when the only call site is the capability's own test"
    )
    assert cap["call_sites"] == [], "callable=false must carry zero call_sites"
    assert len(cap["test_only_sites"]) == 1


def test_a_real_outside_caller_makes_it_callable(fixture_repo):
    """Add one caller outside the test tree; now it is a live capability."""
    files = [fixture_repo["widget"], fixture_repo["own_test"], fixture_repo["caller"]]
    idx = cr.build_repo_index(files=files)
    sites = cr.find_symbol_call_sites(idx, "tools.future.widget", "gadget")
    cap = cr.build_capability(
        "tools.future.widget.gadget", "function",
        defined=True, registered=False, resident_visible=False, sites=sites,
    )
    assert cap["tested"] is True
    assert cap["callable"] is True
    assert len(cap["call_sites"]) == 1
    assert cap["call_sites"][0]["file"] == "tools/future/caller.py"


def test_module_level_import_is_also_a_call_site(fixture_repo):
    """The module-granularity question (does anything import widget.py at
    all) must agree with the function-granularity one for this fixture."""
    files = [fixture_repo["widget"], fixture_repo["own_test"], fixture_repo["caller"]]
    idx = cr.build_repo_index(files=files)
    sites = cr.find_module_import_sites(idx, "tools.future.widget")
    prod = [s for s in sites if not cr.is_test_path(Path(s.file))]
    assert len(prod) == 1
    assert prod[0].file == "tools/future/caller.py"


def test_relative_import_binds_to_the_fully_dotted_symbol(tmp_path, monkeypatch):
    """Regression: `from .helper import fn` inside a package must bind `fn`
    to `pkg.helper.fn`, not the raw `helper.fn` -- a real bug caught while
    building this analyzer (hcli/escalation.py's three functions all showed
    zero call sites through hcli/tool_registry.py's relative imports until
    this was fixed)."""
    monkeypatch.setattr(cr, "REPO", tmp_path)
    cr._TEXT_CACHE.clear()
    _write(tmp_path, "pkg/__init__.py", "")
    helper = _write(tmp_path, "pkg/helper.py", "def do_it():\n    return 1\n")
    caller = _write(
        tmp_path, "pkg/caller.py",
        "from .helper import do_it\n\ndef run():\n    return do_it()\n",
    )
    idx = cr.build_repo_index(files=[helper, caller])
    sites = cr.find_symbol_call_sites(idx, "pkg.helper", "do_it")
    assert len(sites) == 1
    assert sites[0].file == "pkg/caller.py"


def test_subprocess_string_mention_without_a_real_launch_is_not_a_call_site(tmp_path, monkeypatch):
    """A path string sitting in a metadata dict is not a launch. Regression
    for a real false positive: hcli/agentos/handoff.py names
    'tools/odyssey_ctl.py' in a plain dict value, and that is not evidence
    anything runs it."""
    monkeypatch.setattr(cr, "REPO", tmp_path)
    cr._TEXT_CACHE.clear()
    mentions_only = _write(
        tmp_path, "notes.py",
        'DOC = {"unrelated_preserved_edit": "tools/odyssey_ctl.py"}\n',
    )
    really_launches = _write(
        tmp_path, "launcher.py",
        "import subprocess\n"
        'subprocess.run(["python3", "tools/odyssey_ctl.py", "cycle"])\n',
    )
    sites = cr._subprocess_path_sites(
        "tools/odyssey_ctl.py", [mentions_only, really_launches], exclude_files=()
    )
    files_hit = {s.file for s in sites}
    assert files_hit == {"launcher.py"}, (
        "a dict-value mention must not count; only the real subprocess.run call site should"
    )


def test_tool_name_requires_dispatch_context_not_bare_string(tmp_path, monkeypatch):
    """Two unrelated systems can reuse the same short tool name
    ('git.status' in both hcli/tool_registry.py and
    tools/hcli/bootstrap/p0_tool_bridge.py's own ALL_TOOLS set). A bare set literal
    must not count as a call site for THIS registry."""
    monkeypatch.setattr(cr, "REPO", tmp_path)
    cr._TEXT_CACHE.clear()
    unrelated_bridge = _write(
        tmp_path, "other_bridge.py",
        'ALL_TOOLS = {"git.status", "fs.read"}\n',
    )
    real_dispatch = _write(
        tmp_path, "gate.py",
        'CATALOG = [{"label": "x", "tool": "git.status", "arguments": {}}]\n',
    )
    sites = cr.find_literal_sites("git.status", [unrelated_bridge, real_dispatch])
    files_hit = {s.file for s in sites}
    assert files_hit == {"gate.py"}


def test_using_reader_does_not_leak_after_context(tmp_path, monkeypatch):
    """An overlay reader is scoped: after the block, on-disk bytes win again."""
    monkeypatch.setattr(cr, "REPO", tmp_path)
    cr._TEXT_CACHE.clear()
    cr._TRACKED_PY = None
    cr._GIT_CHECKOUT = None
    cr._REL_MEMO.clear()
    path = _write(tmp_path, "notes.py", "hello from disk\n")

    def stolen(_path):
        return "STOLEN_OVERLAY"

    with cr.using_reader(stolen):
        assert cr.read_text(path) == "STOLEN_OVERLAY"
    assert cr.read_text(path) == "hello from disk\n"
    assert cr._reader_var.get() is None


def test_roadmap_view_does_not_blind_later_fixture_reads(fixture_repo):
    """tools.roadmap used to replace cr.read_text with a SourceView closure.

    After that assignment, hermetic fixture files (not in the real repo HEAD)
    read as empty, so the own-test call site vanished and
    test_only_own_test_calls_it_is_not_callable failed whenever roadmap ran
    first in the same process. HCLI is that same process for real callers.
    """
    from tools.roadmap import reach
    from tools.roadmap.gitfs import SourceView

    original = cr.read_text
    view = SourceView()
    view.overlay["syn/mod.py"] = "def abort():\n    pass\n"
    view.overlay["syn/caller.py"] = (
        "from syn.mod import abort\n\ndef run():\n    abort()\n"
    )
    look = reach.scan_probe_ast(
        view,
        {
            "code_paths": ["syn/mod.py"],
            "modules": ["syn.mod"],
            "symbols": [{"module": "syn.mod", "symbol": "abort"}],
        },
        unique_paths={"syn/mod.py"},
    )
    assert look["defined"] is True
    assert cr.read_text is original, (
        "roadmap.install_view must not replace capability_reachability.read_text; "
        "a long-lived process calls both surfaces"
    )
    assert cr._reader_var.get() is None

    files = [fixture_repo["widget"], fixture_repo["own_test"]]
    idx = cr.build_repo_index(files=files)
    sites = cr.find_symbol_call_sites(idx, "tools.future.widget", "gadget")
    cap = cr.build_capability(
        "tools.future.widget.gadget", "function",
        defined=True, registered=False, resident_visible=False, sites=sites,
    )
    assert cap["tested"] is True, "the own-test call site must still count toward tested"
    assert cap["callable"] is False
    assert len(cap["test_only_sites"]) == 1


# --------------------------------------------------------------------------
# Live-repo invariants: hold regardless of what code lands next.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_doc():
    out = cr.build()
    return cr.load_json(out)


def test_live_receipt_schema(live_doc):
    assert live_doc["schema"] == cr.SCHEMA
    assert live_doc["seal_sha256"]
    assert live_doc["bench"]["gpu_authority"] is False
    assert live_doc["bench"]["measurement_state"] == "STATIC_ONLY"


def test_live_every_capability_has_the_five_fields_and_consistent_evidence(live_doc):
    caps = live_doc["capabilities"]
    assert len(caps) >= 44, "at least the 44+ typed tools plus named/sidecar capabilities"
    for name, row in caps.items():
        for field_name in cr.FIELDS:
            assert isinstance(row[field_name], bool), f"{name}.{field_name}"
        assert row["callable"] == bool(row["call_sites"]), name
        assert not row["callable"] or not row["registered"] or row["resident_visible"], (
            f"{name}: registered ToolSpecs are always resident_visible"
        )


def test_live_dead_surface_is_the_non_callable_defined_set(live_doc):
    caps = live_doc["capabilities"]
    dead_names = {d["name"] for d in live_doc["DEAD_SURFACE"]}
    expected = {n for n, c in caps.items() if c["defined"] and not c["callable"]}
    assert dead_names == expected
    assert dead_names, "there is always at least one dead typed tool in this codebase"


def test_live_tool_registry_is_a_live_production_entry_point(live_doc):
    row = live_doc["capabilities"]["hcli.tool_registry.default_tool_registry"]
    assert row["callable"] is True
    assert row["registered"] is True
    assert row["resident_visible"] is True


def test_live_escalation_handlers_are_wired_through_tool_registry(live_doc):
    """hcli/tool_registry.py's frontier.escalate / grok.swarm.* handlers are
    the sole callers of these three functions -- a call site in
    hcli/tool_registry.py, not in a test file."""
    caps = live_doc["capabilities"]
    for symbol in ("escalate_to_frontier", "propose_swarm", "launch_swarm"):
        row = caps[f"hcli.escalation.{symbol}"]
        assert row["callable"] is True, symbol
        assert any(s["file"] == "hcli/tool_registry.py" for s in row["call_sites"]), symbol


def test_live_sidecar_modules_are_overwhelmingly_not_registered(live_doc):
    """The settled finding this wave rests on: AgentOS cannot discover the
    tools/future sidecar. Not a hardcoded count -- just that it is not
    (close to) all of them."""
    sidecars = [c for c in live_doc["capabilities"].values() if c["kind"] == "sidecar_module"]
    assert sidecars
    registered = sum(1 for c in sidecars if c["registered"])
    assert registered < len(sidecars) // 2


def test_live_selftest_passes():
    cr.selftest()


# --------------------------------------------------------------------------
# Rust index path: fallback, hermetic facts, live-repo parity gate.
# --------------------------------------------------------------------------


def test_assemble_falls_back_when_binary_missing(fixture_repo, monkeypatch):
    """A missing binary must degrade to the Python scan, never crash."""
    monkeypatch.setenv("HAWKING_INDEX_BIN", str(fixture_repo["root"] / "no-such-hawking-index"))
    monkeypatch.delenv("HAWKING_REACHABILITY_FORCE_PYTHON", raising=False)
    doc = cr.assemble()
    assert doc["facts_source"] == "python-ast"
    assert "schema" in doc
    assert isinstance(doc["capabilities"], dict)
    assert any(n.startswith("tools.future.") for n in doc["capabilities"])


def test_repo_index_from_facts_drives_call_sites(fixture_repo):
    """The assembler must honour precomputed facts (what the rust CLI emits)
    without re-walking the tree: import is not a call, a Name call is."""
    files = [fixture_repo["widget"], fixture_repo["own_test"], fixture_repo["caller"]]
    facts = {
        "schema": cr.RUST_FACTS_SCHEMA,
        "files": ["tools/future/widget.py", "tools/future/test_widget.py", "tools/future/caller.py"],
        "import_sites": {
            "tools.future.widget": [
                {"file": "tools/future/test_widget.py", "line": 1, "kind": "import"},
                {"file": "tools/future/caller.py", "line": 1, "kind": "import"},
            ],
            "tools.future.widget.gadget": [
                {"file": "tools/future/test_widget.py", "line": 1, "kind": "import"},
                {"file": "tools/future/caller.py", "line": 1, "kind": "import"},
            ],
        },
        "bound_names": {
            "tools/future/test_widget.py": [["gadget", "tools.future.widget.gadget"]],
            "tools/future/caller.py": [["gadget", "tools.future.widget.gadget"]],
        },
        "calls": [
            {"file": "tools/future/test_widget.py", "line": 4, "name": "gadget"},
            {"file": "tools/future/caller.py", "line": 4, "name": "gadget"},
        ],
        "subprocess": [],
        "literals": [],
    }
    idx = cr.repo_index_from_facts(facts)
    assert idx.facts_source == "hawking-index"
    sites = cr.find_symbol_call_sites(idx, "tools.future.widget", "gadget")
    cap = cr.build_capability(
        "tools.future.widget.gadget", "function",
        defined=True, registered=False, resident_visible=False, sites=sites,
    )
    assert cap["tested"] is True
    assert cap["callable"] is True
    assert cap["call_sites"] == [
        {"file": "tools/future/caller.py", "line": 4, "kind": "call"}
    ]
    # And the files we passed aren't needed for the lookup — the table is.
    assert set(cr.rel(p) for p in files) == set(facts["files"])


def test_parity_gate_rust_matches_python_on_this_repo():
    """Both paths over the real repo: identical keys and identical capability objects.

    A missing hawking-index binary must fail this test, not skip it. Skipping
    is how rust/python divergence (21 verdicts after the indexer moved to HEAD)
    landed without a red test.
    """
    report = cr.run_parity()
    assert report["status"] != "BINARY_MISSING", (
        "hawking-index binary is required for rust/python parity; "
        "a skip here would let the analyzers diverge silently again"
    )
    assert report["status"] == "IDENTICAL", (
        f"parity {report['status']} compared={report.get('compared')} "
        f"key_sets_equal={report.get('key_sets_equal')} "
        f"diffs={report.get('diffs', [])[:20]}"
    )
    assert report.get("key_sets_equal") is True
    assert report["compared"] >= 44
    assert report["rust_count"] == report["python_count"] == report["compared"]
    assert not report.get("diffs")


def test_head_source_ignores_dirty_worktree(tmp_path, monkeypatch):
    """source='head' (default) reads committed blobs; source='worktree' sees dirt.

    This is the named opt-in the assembler exposes. No production caller
    passes worktree; the parameter exists so the two behaviours cannot be
    silent variants of the same function.
    """
    repo = tmp_path
    monkeypatch.setattr(cr, "REPO", repo)
    monkeypatch.setattr(cr_common, "REPO", repo)
    cr._TEXT_CACHE.clear()
    cr._TRACKED_PY = None
    cr._GIT_CHECKOUT = None
    cr._REL_MEMO.clear()

    def git_in(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
        )

    git_in("init", "-b", "main")
    git_in("config", "user.email", "r6@test")
    git_in("config", "user.name", "r6")
    git_in("config", "commit.gpgsign", "false")

    widget = repo / "tools" / "future" / "widget.py"
    caller = repo / "tools" / "future" / "caller.py"
    widget.parent.mkdir(parents=True)
    widget.write_text("def gadget():\n    return 1\n")
    caller.write_text(
        "from tools.future.widget import gadget\n\n"
        "def use_it():\n    return gadget() + 1\n"
    )
    git_in("add", "-A")
    git_in("commit", "-m", "base")

    caller.write_text(
        "from tools.future.widget import gadget\n\n"
        "def use_it():\n    return gadget() + 1\n"
        "def extra_only_on_disk():\n    return gadget()\n"
    )
    (repo / "tools" / "future" / "untracked.py").write_text(
        "from tools.future.widget import gadget\n\n"
        "def ghost():\n    return gadget()\n"
    )

    files = [widget, caller, repo / "tools" / "future" / "untracked.py"]
    cr._TEXT_CACHE.clear()
    idx_head = cr.build_repo_index(files=files, source="head")
    sites_head = cr.find_symbol_call_sites(idx_head, "tools.future.widget", "gadget")
    head_files = {s.file for s in sites_head}
    head_lines = {(s.file, s.line) for s in sites_head}
    assert ("tools/future/caller.py", 4) in head_lines
    assert ("tools/future/caller.py", 6) not in head_lines
    assert "tools/future/untracked.py" not in head_files

    cr._TEXT_CACHE.clear()
    idx_wt = cr.build_repo_index(files=files, source="worktree")
    sites_wt = cr.find_symbol_call_sites(idx_wt, "tools.future.widget", "gadget")
    wt_lines = {(s.file, s.line) for s in sites_wt}
    wt_files = {s.file for s in sites_wt}
    assert ("tools/future/caller.py", 6) in wt_lines
    assert "tools/future/untracked.py" in wt_files
