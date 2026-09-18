"""Focused regression for the P18 tunnel operation permission gate.

Exercises ``tunnel_operation_allowed`` in ``hawking.share_bridge``: the
predicate that a tunnel handler must consult before dispatching an operation.
This is the executable behavior added by the P18 tunnel implementation
contract v2 tranche; it is not a marker, import-only, or version assertion.
"""
from __future__ import annotations

import pytest

from hawking.share_bridge import (
    ShareError,
    tunnel_operation_allowed,
    tunnel_operation_permission,
)


def test_operation_permission_mapping_is_stable() -> None:
    assert tunnel_operation_permission("list_chats") == "read"
    assert tunnel_operation_permission("stream_prompt") == "prompt"
    assert tunnel_operation_permission("scoped_build") == "build"


def test_unknown_operation_fails_closed() -> None:
    with pytest.raises(ShareError) as excinfo:
        tunnel_operation_permission("delete_everything")
    assert excinfo.value.status == 400


def test_read_only_capability_cannot_reach_prompt_or_build() -> None:
    record = {"permissions": ["read"]}
    assert tunnel_operation_allowed(record, "list_chats") is True
    assert tunnel_operation_allowed(record, "retrieve_result") is True
    assert tunnel_operation_allowed(record, "stream_prompt") is False
    assert tunnel_operation_allowed(record, "scoped_build") is False


def test_prompt_capability_cannot_reach_build() -> None:
    record = {"permissions": ["read", "prompt"]}
    assert tunnel_operation_allowed(record, "stream_prompt") is True
    assert tunnel_operation_allowed(record, "scoped_build") is False


def test_build_capability_reaches_all_declared_operations() -> None:
    record = {"permissions": ["read", "prompt", "build"]}
    assert tunnel_operation_allowed(record, "list_models") is True
    assert tunnel_operation_allowed(record, "stream_prompt") is True
    assert tunnel_operation_allowed(record, "scoped_build") is True


def test_missing_or_malformed_permissions_default_to_read_only() -> None:
    assert tunnel_operation_allowed({}, "list_chats") is True
    assert tunnel_operation_allowed({}, "stream_prompt") is False
    assert tunnel_operation_allowed({"permissions": None}, "scoped_build") is False
    assert tunnel_operation_allowed({"permissions": "bogus"}, "stream_prompt") is False


def test_non_mapping_record_is_denied() -> None:
    assert tunnel_operation_allowed(None, "list_chats") is False
    assert tunnel_operation_allowed(["read"], "list_chats") is False


def test_unknown_operation_denied_even_with_broad_permissions() -> None:
    record = {"permissions": ["read", "prompt", "build"]}
    with pytest.raises(ShareError):
        tunnel_operation_allowed(record, "not_a_real_operation")