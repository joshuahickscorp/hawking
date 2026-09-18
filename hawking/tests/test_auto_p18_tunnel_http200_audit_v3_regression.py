"""Focused regression for P18_TUNNEL_HTTP200_AUDIT_V3.

Exercises the production predicate ``tunnel_operation_allowed`` in
``hawking/share_bridge.py``: a capability whose granted permission matches the
operation must additionally be unexpired before a non-read tunnel operation is
allowed, while read operations remain available so an expired capability can
still be inspected.  This is a bounded behavior check, not a release or
hardware qualification.
"""
from __future__ import annotations

import time

from hawking.share_bridge import tunnel_operation_allowed


def _record(permissions, expires_at):
    return {"permissions": permissions, "expires_at": expires_at}


def test_expired_capability_denies_prompt_operation():
    record = _record(["read", "prompt"], time.time() - 60)
    assert tunnel_operation_allowed(record, "stream_prompt") is False


def test_unexpired_capability_allows_prompt_operation():
    record = _record(["read", "prompt"], time.time() + 60)
    assert tunnel_operation_allowed(record, "stream_prompt") is True


def test_expired_capability_still_allows_read_operation():
    record = _record(["read"], time.time() - 60)
    assert tunnel_operation_allowed(record, "list_chats") is True


def test_missing_permission_still_denied_regardless_of_expiry():
    record = _record(["read"], time.time() + 60)
    assert tunnel_operation_allowed(record, "scoped_build") is False


def test_malformed_expiry_fails_closed_for_non_read():
    record = _record(["read", "prompt"], "not-a-timestamp")
    assert tunnel_operation_allowed(record, "stream_prompt") is False


def test_absent_expiry_preserves_prior_behavior():
    record = {"permissions": ["read", "prompt"]}
    assert tunnel_operation_allowed(record, "stream_prompt") is True