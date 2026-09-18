"""Focused regression for the P2_HWEB_IMPLEMENTATION_CONTRACT request-body bound.

This exercises the production behavior added to `hawking/serve.py`: the
chat-completions surface must refuse an oversized or malformed request body
with a JSON error status instead of reading it into memory or raising. It is a
behavioral check on the changed source, not a marker or import assertion.
"""
from __future__ import annotations

import io
import json
import unittest

from hawking import serve


class _FakeHandler(serve.HawkingOpenAIHandler):
    """A handler whose socket is replaced by an in-memory request body."""

    def __init__(self, body: bytes, declared_length: str | None = None):
        self._body = body
        self._declared_length = (
            str(len(body)) if declared_length is None else declared_length
        )
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.headers = {"Content-Length": self._declared_length}
        self.status = None
        self.sent_headers = []

    def send_response(self, code, message=None):
        self.status = code

    def send_header(self, key, value):
        self.sent_headers.append((key, value))

    def end_headers(self):
        pass


class RequestBodyBoundTest(unittest.TestCase):
    def test_oversized_body_is_rejected_with_413(self):
        limit = serve.MAX_REQUEST_BODY_BYTES
        body = b"x" * (limit + 1)
        handler = _FakeHandler(body)
        payload = handler._read_json_body()
        self.assertIsNone(payload)
        self.assertEqual(handler.status, 413)

    def test_malformed_length_is_rejected_with_400(self):
        handler = _FakeHandler(b"{}", declared_length="not-a-number")
        payload = handler._read_json_body()
        self.assertIsNone(payload)
        self.assertEqual(handler.status, 400)

    def test_valid_body_is_parsed(self):
        handler = _FakeHandler(json.dumps({"model": "sealed-3.14"}).encode())
        payload = handler._read_json_body()
        self.assertEqual(payload, {"model": "sealed-3.14"})
        self.assertIsNone(handler.status)


if __name__ == "__main__":
    unittest.main()