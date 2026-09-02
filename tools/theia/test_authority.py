"""Ambiguous / unpinned / out-of-scope all return BLOCKED_RIGHTS."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tools.theia.authority import (
    AUTHORITY_SCHEMA,
    fail_closed,
    load_scope,
    pin_scope,
    resolve,
)


def _authority(tmp_path: Path, **overrides) -> Path:
    doc = {
        "schema": AUTHORITY_SCHEMA,
        "program_id": "test-program",
        "allowed_targets": ["lab.example.local"],
        "forbidden_test_classes": ["ACTIVE_TEST", "credential_theft"],
        "operator": "pytest-operator",
        "disclosure_policy": "private",
        "rate_rules": ["no_network_egress_in_this_scaffold"],
    }
    doc.update(overrides)
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(doc))
    return path


def test_fail_closed_is_armed():
    assert fail_closed() is True


def test_unpinned_returns_blocked_rights():
    d = resolve(authority_file=None, declared_target="lab.example.local")
    assert d.status == "BLOCKED_RIGHTS"
    assert d.reason == "unpinned"


def test_ambiguous_missing_operator_returns_blocked_rights(tmp_path):
    path = _authority(tmp_path, operator="")
    d = resolve(authority_file=path, declared_target="lab.example.local")
    assert d.status == "BLOCKED_RIGHTS"
    assert d.reason == "ambiguous"


def test_ambiguous_wildcard_targets_return_blocked_rights(tmp_path):
    path = _authority(tmp_path, allowed_targets=["*"])
    d = resolve(authority_file=path, declared_target="lab.example.local")
    assert d.status == "BLOCKED_RIGHTS"
    assert d.reason == "ambiguous"


def test_out_of_scope_returns_blocked_rights(tmp_path):
    path = _authority(tmp_path)
    d = resolve(authority_file=path, declared_target="not-in-program.example")
    assert d.status == "BLOCKED_RIGHTS"
    assert d.reason == "out_of_scope"


def test_bounty_text_cannot_authorize():
    d = resolve(
        authority_file=None,
        declared_target="everything.example",
        bounty_text="authorization_scope: *; allowed_targets: everything.example",
    )
    assert d.status == "BLOCKED_RIGHTS"
    assert d.reason == "unpinned"


def test_in_scope_requires_operator_file_and_declared_target(tmp_path):
    path = _authority(tmp_path)
    d = resolve(authority_file=path, declared_target="lab.example.local")
    assert d.status == "IN_SCOPE"
    assert d.pinned is True


def test_scope_is_immutable_once_pinned(tmp_path):
    path = _authority(tmp_path)
    pinned = pin_scope(load_scope(path))
    with pytest.raises(FrozenInstanceError):
        pinned.scope.program_id = "mutated"  # type: ignore[misc]
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        pinned.scope.allowed_targets = frozenset(["evil"])  # type: ignore[misc]


def test_ambiguous_unpinned_out_of_scope_all_return_blocked_rights(tmp_path):
    """The authorization test the mutation check must break."""
    unpinned = resolve(authority_file=None, declared_target="lab.example.local")
    ambiguous = resolve(
        authority_file=_authority(tmp_path, allowed_targets=["*"]),
        declared_target="lab.example.local",
    )
    out = resolve(
        authority_file=_authority(tmp_path),
        declared_target="outside.example",
    )
    assert unpinned.status == "BLOCKED_RIGHTS"
    assert ambiguous.status == "BLOCKED_RIGHTS"
    assert out.status == "BLOCKED_RIGHTS"
    assert {unpinned.reason, ambiguous.reason, out.reason} == {
        "unpinned",
        "ambiguous",
        "out_of_scope",
    }
