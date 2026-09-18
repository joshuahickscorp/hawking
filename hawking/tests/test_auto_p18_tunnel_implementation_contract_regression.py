"""Focused regression for the P18 tunnel operation permission contract.

Exercises the production helper ``tunnel_operation_permission`` in
``hawking.share_bridge``: declared operations resolve to their declared
permission, and unknown/non-string operations fail closed with an explicit
``ShareError`` (HTTP 400) rather than defaulting to a permissive scope.

Boundary: this is a unit-level contract check only.  It performs no network
I/O, no tunnel session creation, and no hardware qualification, and it makes
no release or phase-completion claim.
"""
from __future__ import annotations

import pytest

from hawking.share_bridge import (
    TUNNEL_OPERATION_PERMISSIONS,
    ShareError,
    tunnel_operation_permission,
)


def test_declared_operations_resolve_to_declared_permission():
    for operation, permission in TUNNEL_OPERATION_PERMISSIONS.items():
        assert tunnel_operation_permission(operation) == permission


def test_unknown_operation_fails_closed_with_400():
    with pytest.raises(ShareError) as excinfo:
        tunnel_operation_permission("delete_everything")
    assert excinfo.value.status == 400
    assert "delete_everything" in str(excinfo.value)


def test_non_string_operation_fails_closed_with_400():
    with pytest.raises(ShareError) as excinfo:
        tunnel_operation_permission(None)
    assert excinfo.value.status == 400


def test_unknown_operation_does_not_leak_a_permission():
    with pytest.raises(ShareError):
        tunnel_operation_permission("scoped_build_extra")