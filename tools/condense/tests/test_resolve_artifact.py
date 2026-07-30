#!/usr/bin/env python3.12
"""Tests for the single campaign artifact search path."""
from __future__ import annotations
import sys
from pathlib import Path as _Path_repo
_REPO = _Path_repo(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import os
import sys
from pathlib import Path

import pytest

CONDENSE = Path(__file__).resolve().parents[1]
REPO_ROOT = CONDENSE.parents[1]

from lab.operators.glm52_common import (  # noqa: E402
    Glm52Error,
    resolve_artifact,
)

GRAPH = "GLM52_SHARD_DEPENDENCY_GRAPH.json"
CENSUS = "GLM52_ROUTE_POPULATION_CENSUS.json"
ALLOCATION = "PROMETHEUS_MATH_ALLOCATION_MANIFEST.json"

def test_resolves_from_repo_root() -> None:
    for name in (GRAPH, CENSUS, ALLOCATION):
        path = resolve_artifact(name)
        assert path == REPO_ROOT / name
        assert path.is_file()

def test_resolves_from_hawking_artifact_root_when_repo_copy_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    name = "TEST_ONLY_CAMPAIGN_ARTIFACT.json"
    payload = b'{"schema":"test","ok":true}\n'
    (external / name).write_bytes(payload)

    # Hide any accidental repo-root copy of this test-only basename.
    monkeypatch.setattr(
        "glm52_common.REPO_ROOT",
        tmp_path / "empty_repo",
    )
    (tmp_path / "empty_repo").mkdir()
    monkeypatch.setenv("HAWKING_ARTIFACT_ROOT", str(external))

    path = resolve_artifact(name)
    assert path == external / name
    assert path.read_bytes() == payload

def test_raises_actionable_error_when_neither_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_repo = tmp_path / "repo"
    empty_external = tmp_path / "external"
    empty_repo.mkdir()
    empty_external.mkdir()
    monkeypatch.setattr("glm52_common.REPO_ROOT", empty_repo)
    monkeypatch.setenv("HAWKING_ARTIFACT_ROOT", str(empty_external))

    name = "MISSING_CAMPAIGN_ARTIFACT.json"
    with pytest.raises(Glm52Error) as excinfo:
        resolve_artifact(name)

    message = str(excinfo.value)
    assert name in message
    assert f"git checkout HEAD -- {name}" in message
    assert str(empty_repo / name) in message
    assert str(empty_external / name) in message

def test_repo_root_wins_over_external(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    external = tmp_path / "external"
    repo.mkdir()
    external.mkdir()
    name = "PREFERENCE_TEST_ARTIFACT.json"
    (repo / name).write_text("from-repo\n", encoding="utf-8")
    (external / name).write_text("from-external\n", encoding="utf-8")
    monkeypatch.setattr("glm52_common.REPO_ROOT", repo)
    monkeypatch.setenv("HAWKING_ARTIFACT_ROOT", str(external))

    path = resolve_artifact(name)
    assert path == repo / name
    assert path.read_text(encoding="utf-8") == "from-repo\n"

def test_rejects_non_basename() -> None:
    with pytest.raises(Glm52Error, match="basename"):
        resolve_artifact("subdir/artifact.json")
