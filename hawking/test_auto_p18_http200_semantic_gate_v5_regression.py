"""Focused regression for P18_HTTP200_SEMANTIC_GATE_V5.

Exercises the production gate http200_semantic_gate_v5 in hawking/backends.py:
a 200 status is only accepted when the decoded body carries a non-empty
completion payload. No release or hardware qualification is claimed here.
"""
from __future__ import annotations

import pytest

from hawking.backends import http200_semantic_gate_v5


@pytest.mark.parametrize(
    "status,body,expected_ok,expected_reason",
    [
        (200, {"choices": [{"text": "hi"}]}, True, "ok"),
        (200, "hello", True, "ok"),
        (200, {"content": "answer"}, True, "ok"),
        (200, {"error": "model not loaded"}, False, "error_envelope"),
        (200, {"choices": []}, False, "empty_completion"),
        (200, {"choices": None}, False, "empty_completion"),
        (200, {"content": "   "}, False, "empty_completion"),
        (200, {}, False, "missing_completion"),
        (200, None, False, "empty_body"),
        (200, "   ", False, "empty_body"),
        (500, {"choices": [{"text": "hi"}]}, False, "http_status_500"),
        (200, 42, False, "unexpected_body_type"),
    ],
)
def test_http200_semantic_gate_v5(status, body, expected_ok, expected_reason):
    ok, reason = http200_semantic_gate_v5(status, body)
    assert ok is expected_ok
    assert reason == expected_reason


def test_http200_semantic_gate_v5_rejects_error_envelope_despite_200():
    ok, reason = http200_semantic_gate_v5(200, {"error": {"message": "boom"}})
    assert ok is False
    assert reason == "error_envelope"