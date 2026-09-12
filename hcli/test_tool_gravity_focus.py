from hcli.tool_registry import ToolContext, ToolRegistry, ToolSpec, default_tool_registry


def test_authorized_defensive_security_focus_reaches_existing_primitives(tmp_path):
    registry = default_tool_registry(tmp_path, repo_root=tmp_path)
    result = registry.describe("authorized defensive security", max_results=12)
    names = set(result["names"])
    assert result["shown"] > 0
    assert {"audit", "shell.readonly", "tests.run"} <= names
    # Retrieval must not erase the authority class; the model still sees the
    # distinction between observation, reversible work, and destructive work.
    by_name = {row["name"]: row for row in result["matches"]}
    assert by_name["shell.readonly"]["mutation"] == "read_only"


def test_unrelated_focus_remains_bounded_and_does_not_return_everything(tmp_path):
    registry = default_tool_registry(tmp_path, repo_root=tmp_path)
    result = registry.describe("quantum gardening", max_results=12)
    assert result["names"] == []
    assert result["match_count"] == 0


def test_hcli_agentos_focus_discovers_the_live_capability_plane(tmp_path):
    """A product-level capability question must not look like zero tools."""
    registry = default_tool_registry(tmp_path, repo_root=tmp_path)
    result = registry.describe("HCLI/AgentOS", max_results=21)
    assert result["shown"] > 0
    names = set(result["names"])
    assert {"tools.catalog", "processes", "receipt"} <= names
    assert result["match_count"] >= result["shown"]


def test_catalog_index_invalidates_for_late_application_tools(tmp_path):
    registry = ToolRegistry(ToolContext(tmp_path, tmp_path))
    registry.register(ToolSpec(
        "base.inspect",
        "Inspect the base artifact.",
        {"type": "object", "additionalProperties": False},
    ))

    # Force both derived indexes to materialize before the extension arrives.
    assert registry.discover()[0]["name"] == "base.inspect"
    assert registry.describe("late")["names"] == []

    registry.register(ToolSpec(
        "late.inspect",
        "Inspect the late application artifact.",
        {"type": "object", "additionalProperties": False},
    ))
    assert {row["name"] for row in registry.discover()} == {
        "base.inspect", "late.inspect"
    }
    assert registry.describe("late")["names"] == ["late.inspect"]


def test_native_catalog_ranking_keeps_python_contracts_and_schema(monkeypatch, tmp_path):
    """Rust ranks names only; Python remains the schema/authority owner."""
    registry = ToolRegistry(ToolContext(tmp_path, tmp_path))
    registry.register(ToolSpec(
        "lake.catalog", "Read indexed ModelLake metadata.",
        {"type": "object", "additionalProperties": False},
    ))

    def native(_root, *, entries, terms, max_results):
        assert entries[0]["name"] == "lake.catalog"
        assert terms == ("catalog",)
        assert max_results == 12
        return {"names": ["lake.catalog"], "match_count": 1, "elapsed_ns": 42}

    monkeypatch.setattr("hcli.repo_context.native_tool_catalog", native)
    result = registry.describe("catalog")
    assert result["provenance"] == "hawking-gravityd.tool-catalog.v1"
    assert result["native_elapsed_ns"] == 42
    assert result["matches"][0]["input_schema"] == {
        "type": "object", "additionalProperties": False,
    }


def test_native_dispatch_denial_is_final_before_python_handler(monkeypatch, tmp_path):
    registry = ToolRegistry(ToolContext(tmp_path, tmp_path))
    called = False

    def handler(_context, _args):
        nonlocal called
        called = True
        return {"unexpected": True}

    registry.register(ToolSpec(
        "dangerous.fixture", "A destructive fixture.",
        {"type": "object", "additionalProperties": False},
        mutation="destructive", handler=handler,
    ))
    monkeypatch.setattr(
        "hcli.repo_context.native_tool_dispatch_admit",
        lambda *_args, **_kwargs: {"admitted": False, "reason": "native fixture denial"},
    )
    result = registry.invoke("dangerous.fixture")
    assert not result.ok
    assert result.failure_class == "PERMISSION_DENIED"
    assert result.provenance["source"] == "hawking-gravityd.tool-dispatch.v1"
    assert not called


def test_native_dispatch_snapshot_is_reused_until_registry_changes(monkeypatch, tmp_path):
    registry = ToolRegistry(ToolContext(tmp_path, tmp_path))
    registry.register(ToolSpec(
        "fixture.read", "Read a fixture.",
        {"type": "object", "additionalProperties": False},
        handler=lambda _context, _args: {"ok": True},
    ))
    entry_ids = []

    def native(_root, *, entries, **_kwargs):
        entry_ids.append(id(entries))
        return {"admitted": True}

    monkeypatch.setattr("hcli.repo_context.native_tool_dispatch_admit", native)
    assert registry.invoke("fixture.read").ok
    assert registry.invoke("fixture.read").ok
    registry.register(ToolSpec(
        "fixture.read2", "Read a second fixture.",
        {"type": "object", "additionalProperties": False},
        handler=lambda _context, _args: {"ok": True},
    ))
    assert registry.invoke("fixture.read").ok
    assert entry_ids[0] == entry_ids[1]
    assert entry_ids[2] != entry_ids[1]


def test_native_filesystem_list_preserves_content_free_discovery_contract(monkeypatch, tmp_path):
    from hcli.tool_registry import ToolContext, _list_files

    context = ToolContext(tmp_path, tmp_path)
    native_result = {
        "files": [{
            "path": "src/main.rs", "filename": "main.rs", "type": "file",
            "kind": "file", "size": 17, "bytes": 17,
        }],
        "directories": [{
            "path": "src", "filename": "src", "type": "directory",
            "kind": "directory", "size": None, "bytes": None,
        }],
        "truncated": False,
        "directories_seen": 1,
        "elapsed_ns": 81,
    }
    seen = {}

    def native(root, path, *, glob, max_results, recursive):
        seen.update(root=root, path=path, glob=glob, max_results=max_results,
                    recursive=recursive)
        return dict(native_result)

    monkeypatch.setattr("hcli.repo_context.native_filesystem_list", native)
    result = _list_files(context, {"path": ".", "glob": "*.rs", "max_results": 4})
    assert result["provenance"] == "hawking-gravityd.list.v1"
    assert result["elapsed_ns"] == 81
    assert "content" not in result["files"][0]
    assert seen["root"] == tmp_path.resolve()
    assert seen["path"] == tmp_path.resolve()
    assert seen["glob"] == "*.rs"
    assert seen["max_results"] == 4
    assert seen["recursive"] is False


def test_native_filesystem_search_preserves_scoped_match_contract(monkeypatch, tmp_path):
    from hcli.tool_registry import ToolContext, _search_files

    context = ToolContext(tmp_path, tmp_path)
    seen = {}

    def native(root, path, *, needle, glob, max_results, max_per_file):
        seen.update(root=root, path=path, needle=needle, glob=glob,
                    max_results=max_results, max_per_file=max_per_file)
        return {
            "root": str(path), "pattern": needle,
            "matches": [{"path": "hcli/main.py", "line": 9, "text": "needle"}],
            "files_seen": 1, "skipped_large": 0, "truncated": False,
            "elapsed_ns": 73,
        }

    monkeypatch.setattr("hcli.repo_context.native_filesystem_search", native)
    result = _search_files(context, {
        "root": ".", "pattern": "needle", "glob": "*.py",
        "max_results": 7, "max_per_file": 2,
    })
    assert result["provenance"] == "hawking-gravityd.search-files.v1"
    assert result["elapsed_ns"] == 73
    assert result["matches"][0]["path"] == str(tmp_path / "hcli/main.py")
    assert seen["root"] == tmp_path.resolve()
    assert seen["path"] == tmp_path.resolve()
    assert seen["needle"] == "needle"
    assert seen["glob"] == "*.py"
    assert seen["max_results"] == 7
    assert seen["max_per_file"] == 2


def test_native_file_body_transport_is_lab_opt_in_until_it_earns_promotion(monkeypatch, tmp_path):
    from hcli.tool_registry import ToolContext, _read_file

    path = tmp_path / "sample.py"
    path.write_text("python handler\n", encoding="utf-8")
    called = False

    def native(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"content": "native handler", "sha256": "native"}

    monkeypatch.setenv("HCLI_NATIVE_GRAVITY", "1")
    monkeypatch.delenv("HCLI_NATIVE_GRAVITY_READ", raising=False)
    monkeypatch.setattr("hcli.repo_context.native_filesystem_read", native)
    context = ToolContext(tmp_path, tmp_path)

    result = _read_file(context, {"path": "sample.py"})
    assert result["content"] == "python handler\n"
    assert not called

    monkeypatch.setenv("HCLI_NATIVE_GRAVITY_READ", "1")
    result = _read_file(context, {"path": "sample.py"})
    assert result["content"] == "native handler"
    assert called


def test_repo_context_accepts_native_direct_path_rank_without_full_inventory(monkeypatch, tmp_path):
    from hcli.repo_context import RepoContext

    context = RepoContext(root=tmp_path, name=tmp_path.name, is_git=False)
    monkeypatch.setattr(
        "hcli.repo_context.native_rank_code_paths",
        lambda root, *, suffixes, words, max_results: {
            "paths": ["hcli/serve.py"],
            "files_seen": 10_604,
            "complete": True,
            "elapsed_ns": 91,
        },
    )
    monkeypatch.setattr(
        context,
        "_candidate_files",
        lambda: (_ for _ in ()).throw(AssertionError("full path inventory should not run")),
    )

    assert context.paths_for("inspect hcli/serve.py", limit=4) == ["hcli/serve.py"]


def test_native_code_path_validation_keeps_only_canonical_relative_paths():
    from hcli.repo_context import _validated_native_code_path

    assert _validated_native_code_path("hcli/serve.py") == "hcli/serve.py"
    assert _validated_native_code_path("src/.hidden.py") == "src/.hidden.py"
    for path in ("/tmp/serve.py", "../serve.py", "hcli/../serve.py",
                 "hcli/./serve.py", "hcli\\serve.py", "hcli/serve.txt", ".py"):
        assert _validated_native_code_path(path) is None


def test_repo_context_accepts_native_multi_term_rank_without_inventory(monkeypatch, tmp_path):
    from hcli.repo_context import RepoContext

    context = RepoContext(root=tmp_path, name=tmp_path.name, is_git=False)
    seen = {}

    def native(root, *, terms, scope_prefix, test_focus, max_results):
        seen.update(
            root=root,
            terms=terms,
            scope_prefix=scope_prefix,
            test_focus=test_focus,
            max_results=max_results,
        )
        return {
            "paths": ["hcli/session_transport.py"],
            "files_seen": 1,
            "terms_used": 2,
            "elapsed_ns": 73,
        }

    monkeypatch.setattr("hcli.repo_context.native_rank_content_paths", native)
    monkeypatch.setattr(
        context,
        "_candidate_files",
        lambda: (_ for _ in ()).throw(AssertionError("content ranking should not walk")),
    )

    assert context.paths_for(
        "Determine whether HCLI streaming preserves durable session state",
        limit=4,
    ) == ["hcli/session_transport.py"]
    assert seen["scope_prefix"] == "hcli/"
    assert seen["test_focus"] is False
