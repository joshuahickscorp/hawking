"""Constrained read tools must not follow a checkout symlink out of scope."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

from hawking import tool_registry as tr
from hawking.tool_registry import (
    _MAX_READ_BYTES,
    ToolContext,
    default_tool_registry,
    suppress_native_helpers,
    suppress_restricted_worker_capabilities,
)


def _registry(workspace: Path):
    return default_tool_registry(workspace, repo_root=workspace)


def _invoke(registry, name: str, args: dict):
    # Restricted workers force this path; keep the regression independent of
    # whether a local Gravity native helper happens to be installed.
    with suppress_native_helpers():
        return registry.invoke(name, args)


def test_fs_search_and_list_skip_a_file_symlink_outside_read_roots(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = "BOUNDARY_ESCAPE_TOKEN"
    (outside / "secret.txt").write_text(secret, encoding="utf-8")
    (workspace / "linked.txt").symlink_to(outside / "secret.txt")
    registry = _registry(workspace)

    search = _invoke(registry, "fs.search", {"path": ".", "pattern": secret})
    listing = _invoke(registry, "fs.list", {"path": "."})

    assert search.ok, search.error
    assert search.value["matches"] == []
    assert listing.ok, listing.error
    assert "linked.txt" not in {
        row["filename"] for row in listing.value["files"]
    }


def test_named_roadmap_is_readable_without_granting_civilization_tree(tmp_path):
    workspace = tmp_path / "workspace"
    roadmap = workspace / "civilization" / "ROADMAP_STATE.json"
    roadmap.parent.mkdir(parents=True)
    roadmap.write_text('{"status":"CURRENT"}', encoding="utf-8")
    (roadmap.parent / "other.json").write_text('{"status":"PRIVATE"}', encoding="utf-8")
    registry = _registry(workspace)

    roadmap_read = _invoke(registry, "roadmap.read", {})
    roadmap_inspect = _invoke(registry, "roadmap.inspect", {})
    broad = _invoke(registry, "receipt.read", {"path": "civilization/other.json"})

    assert roadmap_read.ok, roadmap_read.error
    assert roadmap_read.value["document"] == {"status": "CURRENT"}
    assert roadmap_inspect.ok, roadmap_inspect.error
    assert not broad.ok
    assert "receipt path must be under" in str(broad.error)


def test_static_campaign_and_prior_readers_do_not_follow_outside_symlinks(tmp_path):
    workspace = tmp_path / "workspace"
    civilization = workspace / "civilization"
    receipts = workspace / "receipts" / "future"
    civilization.mkdir(parents=True)
    receipts.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = "STATIC_READER_ESCAPE_TOKEN"
    (outside / "goal.txt").write_text(
        f"PRIMARY OBJECTIVE: {secret}\n", encoding="utf-8"
    )
    (outside / "laws.json").write_text(
        json.dumps({"laws": [{"law_id": "outside", "statement": secret}]}),
        encoding="utf-8",
    )
    (civilization / "sovereign-goal.txt").symlink_to(outside / "goal.txt")
    (receipts / "ODYSSEY2_LAW_STORE.json").symlink_to(outside / "laws.json")
    registry = _registry(workspace)

    state = _invoke(registry, "campaign.state", {})
    priors = _invoke(registry, "odyssey.priors", {})

    assert state.ok, state.error
    assert priors.ok, priors.error
    assert secret not in json.dumps(state.value, sort_keys=True)
    assert secret not in json.dumps(priors.value, sort_keys=True)
    assert any("outside read roots" in item for item in priors.value["sources_missing"])


def test_large_allowed_file_is_prefix_read_without_a_full_file_allocation(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    large = workspace / "large.txt"
    large.write_bytes(b"x" * (_MAX_READ_BYTES + 1))
    registry = _registry(workspace)

    # The old handler called Path.read_bytes() before clipping. If that comes
    # back, this fails before a multi-megabyte fixture can become a memory
    # regression in a constrained worker.
    with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read")):
        result = _invoke(registry, "fs.read", {"path": "large.txt"})

    assert result.ok, result.error
    assert result.value["bytes"] == _MAX_READ_BYTES + 1
    assert result.value["truncated"] is True
    assert result.value["sha256"] is None
    assert result.value["sha256_scope"] == "not_computed_file_exceeds_bounded_read_limit"


def test_large_receipt_is_refused_before_whole_file_read(tmp_path):
    workspace = tmp_path / "workspace"
    receipt = workspace / "receipts" / "large.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"{" + b"x" * _MAX_READ_BYTES)
    registry = _registry(workspace)

    with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read")):
        result = _invoke(registry, "receipt.read", {"path": "receipts/large.json"})

    assert not result.ok
    assert "bounded read limit" in str(result.error)


def test_restricted_research_does_not_attach_ambient_github_or_hf_tokens(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context = ToolContext(workspace=workspace, repo_root=workspace)
    headers_seen = []

    def fake_fetch(_context, args, *, headers=None, **_kwargs):
        headers_seen.append(dict(headers or {}))
        if "api.github.com" in str(args["url"]):
            content = json.dumps({"items": [], "total_count": 0})
        else:
            content = json.dumps({"sha": "public", "siblings": []})
        return {"content": content, "provenance": {}}

    with mock.patch.object(tr, "_fetch", side_effect=fake_fetch), \
            mock.patch.dict(os.environ, {"GH_TOKEN": "not-for-worker", "HF_TOKEN": "not-for-worker"}):
        with suppress_restricted_worker_capabilities():
            tr._github_search(context, {"query": "public evidence"})
            tr._huggingface_resolve(context, {"repo": "org/public-model"})

    assert headers_seen
    assert all("Authorization" not in headers for headers in headers_seen)


def test_restricted_research_disables_ambient_proxy_discovery(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context = ToolContext(workspace=workspace, repo_root=workspace)
    handlers_seen = []

    class _Response:
        status = 200
        headers = {}

        def read(self, _limit):
            return b"public evidence"

        def geturl(self):
            return "https://public.example/"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _Opener:
        def open(self, _request, timeout):
            assert timeout > 0
            return _Response()

    def build_opener(*handlers):
        handlers_seen.extend(handlers)
        return _Opener()

    with mock.patch.object(tr, "_host_is_public", return_value=True), \
            mock.patch.object(tr.urllib.request, "build_opener", side_effect=build_opener), \
            mock.patch.dict(os.environ, {"HTTPS_PROXY": "https://user:secret@proxy.invalid"}):
        with suppress_restricted_worker_capabilities():
            tr._fetch(context, {"url": "https://public.example/"})

    proxy_handlers = [
        handler for handler in handlers_seen
        if isinstance(handler, tr.urllib.request.ProxyHandler)
    ]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}
