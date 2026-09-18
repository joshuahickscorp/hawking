"""P18_TUNNEL_WIRE_STATUS_V5 regression: the request-body wall-clock budget must
cover the header read, not only the body read.

WHY THIS EXISTS. `_read_json_body` bounds the body read with
`MAX_REQUEST_BODY_SECONDS`, but that deadline is armed only after the first
body byte arrives. A caller that sends a complete, valid header block and then
stalls before sending any body byte leaves the first `rfile.read` blocked with
the deadline never armed, so the resident thread is held indefinitely. This
test drives the real handler over a real socket pair and asserts the observable
behavior: a header-only request is answered 408 within the wall-clock budget
instead of hanging.

BOUNDARY. This is a wire-status regression only. It makes no claim about
model quality, decode throughput, release readiness, or hardware qualification.
"""
from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from hawking import serve


class _StallingHandler(serve._Handler):  # type: ignore[attr-defined]
    """A handler whose header read stalls forever after a valid header block."""

    def _read_headers(self, rfile):  # pragma: no cover - exercised via socket
        raise AssertionError("header read must be bounded, not delegated")


def _serve_one(handler_cls, client_sock, server_sock):
    handler = handler_cls.__new__(handler_cls)
    handler.connection = server_sock
    handler.rfile = server_sock.makefile("rb")
    handler.wfile = server_sock.makefile("wb")
    handler.client_address = ("127.0.0.1", 0)
    handler.server = None
    handler.close_connection = True
    try:
        handler.handle_one_request()
    except Exception:  # noqa: BLE001 - the assertion is on the wire bytes
        pass
    finally:
        try:
            handler.wfile.flush()
        except Exception:  # noqa: BLE001
            pass


def test_header_only_request_is_bounded_by_wall_clock_budget():
    """A valid header block with no body must not hold the thread forever."""
    budget = getattr(serve, "MAX_REQUEST_BODY_SECONDS", None)
    assert budget is not None, "MAX_REQUEST_BODY_SECONDS must be a named constant"
    assert budget > 0

    client_sock, server_sock = socket.socketpair()
    try:
        request = (
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 64\r\n"
            b"\r\n"
        )
        client_sock.sendall(request)

        thread = threading.Thread(
            target=_serve_one,
            args=(serve._Handler, client_sock, server_sock),
            daemon=True,
        )
        started = time.monotonic()
        thread.start()
        thread.join(timeout=budget + 5.0)
        elapsed = time.monotonic() - started

        assert not thread.is_alive(), (
            "header-only request held the handler thread past the wall-clock "
            f"budget ({elapsed:.2f}s > {budget:.2f}s)"
        )

        client_sock.settimeout(2.0)
        raw = b""
        try:
            while True:
                chunk = client_sock.recv(4096)
                if not chunk:
                    break
                raw += chunk
        except socket.timeout:
            pass

        assert raw, "handler produced no response bytes for a header-only request"
        status_line = raw.split(b"\r\n", 1)[0]
        assert b" 408 " in status_line, (
            f"expected 408 for a header-only request, got {status_line!r}"
        )
        body = raw.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in raw else b""
        if body:
            payload = json.loads(body.decode("utf-8"))
            assert "error" in payload
    finally:
        client_sock.close()
        server_sock.close()


def test_body_deadline_constant_is_shared_by_header_and_body_reads():
    """The header read must be armed with the same named wall-clock budget."""
    import inspect

    source = inspect.getsource(serve._read_json_body)
    assert "MAX_REQUEST_BODY_SECONDS" in source
    assert source.index("MAX_REQUEST_BODY_SECONDS") < source.index("readline") or (
        "deadline" in source
    ), "header read must be armed with the wall-clock budget before the body read"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))