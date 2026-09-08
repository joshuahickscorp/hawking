"""Reachability was accepted as readiness, and a corpse passed the check.

An orphaned mlx_lm.server whose model directory had been deleted answered
/v1/models with HTTP 200 in 0.004s while /v1/chat/completions returned nothing
at all after 45 seconds, holding 127.0.0.1:9999. The reachability loop accepts
any status from 200 to 499 -- 404 and 405 included -- so it declared that corpse
READY and every caller would hang on its first real call.
"""
import http.server, json, sys, threading, pathlib, time
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from hcli.backends import OpenAICompatibleBackend  # noqa: E402


class _Corpse(http.server.BaseHTTPRequestHandler):
    """Answers the readiness probe. Cannot generate. Exactly the orphan."""
    def do_GET(self):                                    # noqa: N802
        body = json.dumps({"object": "list", "data": [{"id": "ghost"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):                                   # noqa: N802
        time.sleep(30)                                   # hangs, like the corpse

    def log_message(self, *a):                           # silence
        pass


class _Live(_Corpse):
    def do_POST(self):                                   # noqa: N802
        body = json.dumps({"choices": [{"message": {"content": "1"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(handler):
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def _connector(url):
    c = OpenAICompatibleBackend(url)
    c.spawn()
    return c


def test_a_server_that_cannot_generate_is_not_ready():
    srv, url = _serve(_Corpse)
    try:
        assert _connector(url).ready(timeout=3.0) is False, (
            "a server that answers /v1/models but never completes was called READY")
    finally:
        srv.shutdown()


def test_a_server_that_can_generate_is_ready():
    srv, url = _serve(_Live)
    try:
        assert _connector(url).ready(timeout=5.0) is True
    finally:
        srv.shutdown()


def test_capability_is_proven_once_not_on_every_call():
    """The generate probe must not run on every readiness check.

    ready() is called repeatedly -- backends.py does so with timeout 0.0 in two
    places -- so a probe per call would put a generation on every status query.
    """
    posts = []

    class _Counting(_Live):
        def do_POST(self):                               # noqa: N802
            posts.append(1)
            _Live.do_POST(self)

    srv, url = _serve(_Counting)
    try:
        c = _connector(url)
        assert c.ready(timeout=5.0) is True
        assert c._capability_proven is True
        assert len(posts) == 1, posts
        for _ in range(4):
            assert c.ready(timeout=5.0) is True
        assert len(posts) == 1, f"the probe re-ran {len(posts)} times"
    finally:
        srv.shutdown()


def test_the_escape_hatch_still_works():
    import os
    srv, url = _serve(_Corpse)
    os.environ["HCLI_REMOTE_SKIP_CAPABILITY_PROBE"] = "1"
    try:
        assert _connector(url).ready(timeout=3.0) is True
    finally:
        del os.environ["HCLI_REMOTE_SKIP_CAPABILITY_PROBE"]
        srv.shutdown()


class _Broken(_Corpse):
    """Answers the readiness probe, then errors on generation."""
    def do_POST(self):                                   # noqa: N802
        self.send_response(500)
        self.send_header("Content-Length", "0")
        self.end_headers()


def test_a_server_that_errors_on_generation_is_not_ready():
    """A hang is not the only way to be unable to serve."""
    srv, url = _serve(_Broken)
    try:
        assert _connector(url).ready(timeout=3.0) is False, (
            "a server returning HTTP 500 on generation was called READY")
    finally:
        srv.shutdown()


class _Unauthorized(_Corpse):
    """401 proves a live provider that simply wants a credential."""
    def do_POST(self):                                   # noqa: N802
        self.send_response(401)
        self.send_header("Content-Length", "0")
        self.end_headers()


def test_a_credential_error_still_counts_as_alive():
    """The server answered with an opinion, so it is serving."""
    srv, url = _serve(_Unauthorized)
    try:
        assert _connector(url).ready(timeout=3.0) is True
    finally:
        srv.shutdown()
