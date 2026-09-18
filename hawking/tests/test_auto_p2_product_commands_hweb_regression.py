"""Focused regression for the P2_PRODUCT_COMMANDS_HWEB request-body bound.

This exercises the observable behavior added to `hawking/serve.py`: a
chat-completions request whose `Content-Length` header is missing or not a
number is refused with a JSON error body instead of being handed to an
unbounded `rfile.read()`. The check is deliberately narrow -- it drives the
real handler over a socket pair and asserts the status line and the JSON
error shape -- so it fails if the bound is removed and passes only when the
refusal is actually on the request path.

No release or hardware qualification is claimed here; this is a source-level
behavioral check only.
"""
from __future__ import annotations

import json
import socket
import threading

import pytest

from hawking import serve


def _drive_raw_request(raw: bytes) -> bytes:
    """Send `raw` to the serve handler over a socket pair, return the reply."""
    client, server = socket.socketpair()
    try:
        done = threading.Event()

        def _run() -> None:
            try:
                serve._handle_connection(server)
            except Exception:  # pragma: no cover - surfaced via the reply
                pass
            finally:
                done.set()

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        client.sendall(raw)
        client.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            piece = client.recv(65536)
            if not piece:
                break
            chunks.append(piece)
        done.wait(timeout=10.0)
        return b"".join(chunks)
    finally:
        client.close()
        server.close()


def _status_line(reply: bytes) -> str:
    return reply.split(b"\r\n", 1)[0].decode("latin-1")


def _json_body(reply: bytes) -> dict:
    head, _, body = reply.partition(b"\r\n\r\n")
    assert head, "reply had no header block"
    return json.loads(body.decode("utf-8"))


@pytest.mark.parametrize(
    "raw",
    [
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n",
        (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Length: not-a-number\r\n"
            b"\r\n"
        ),
    ],
    ids=["missing-content-length", "non-numeric-content-length"],
)
def test_chat_completions_refuses_unbounded_body(raw: bytes) -> None:
    reply = _drive_raw_request(raw)
    status = _status_line(reply)
    assert " 400 " in status or " 411 " in status, status
    payload = _json_body(reply)
    assert "error" in payload, payload
    assert payload["error"], payload


def test_body_byte_cap_constant_is_named() -> None:
    """The cap the refusal enforces is a named constant, not a magic number."""
    assert isinstance(serve.MAX_REQUEST_BODY_BYTES, int)
    assert serve.MAX_REQUEST_BODY_BYTES > 0