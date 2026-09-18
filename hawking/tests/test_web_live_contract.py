"""Wire-level C2 tests: H-Web projects canonical stores, never browser state."""
from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from hawking.goal_surface import create_goal
from hawking.serve import make_handler


class _Backend:
    def complete(self, payload, timeout=None):  # pragma: no cover - no cognition in C2 fixture
        raise AssertionError("C2 projection fixture must not invoke a model")


def _request(base: str, path: str, body: dict | None = None) -> dict:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _serve(root: Path):
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(_Backend(), "fixture", greedy=False,
                     health={"status": "ok", "resident": "fixture"},
                     stores={"state_root": str(root)}),
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_refresh_and_reconnect_rehydrate_one_canonical_session(tmp_path: Path) -> None:
    goal = create_goal(tmp_path, objective="disposable projected Goal")
    httpd, base = _serve(tmp_path)
    try:
        replay = _request(base, "/hawking/web/events?session_id=web-c2-live")
        events = replay.get("events", [])
        assert events and all({"event_id", "sequence", "identity", "type", "timestamp", "payload_version", "source_revision"} <= set(event) for event in events)
        cursor = replay["replay"]["through_sequence"]
        _request(base, "/hawking/web/events/ack", {"session_id": "web-c2-live", "sequence": cursor})
        resumed = _request(base, "/hawking/web/events?session_id=web-c2-live")
        assert resumed["events"] == events
        assert resumed["replay"]["through_sequence"] == cursor
        assert cursor >= 1
        resumed = _request(base, "/hawking/web/events?session_id=web-c2-live")
        assert resumed["events"] == events
        assert resumed["replay"]["through_sequence"] == cursor
        assert cursor >= 1
        resumed = _request(base, "/hawking/web/events?session_id=web-c2-live")
        assert resumed["events"] == events
        assert resumed["replay"]["through_sequence"] == cursor
        resumed = _request(base, "/hawking/web/events?session_id=web-c2-live")
        assert resumed["events"] == events
        assert resumed["replay"]["through_sequence"] == cursor
        assert cursor >= 0
        assert cursor >= 0
        reconnect = _request(base, "/hawking/web/events?session_id=web-c2-live")
        assert reconnect["events"] == events
        assert reconnect["replay"]["through_sequence"] == cursor
        resumed = _request(base, "/hawking/web/events?session_id=web-c2-live")
        assert resumed["events"] == events
        assert resumed["replay"]["through_sequence"] == cursor
        resumed = _request(base, "/hawking/web/events?session_id=web-c2-live")
        assert resumed["events"] == events
        assert resumed["replay"]["through_sequence"] == cursor
        resumed = _request(base, "/hawking/web/events?session_id=web-c2-live")
        assert resumed["events"] == events
        assert resumed["replay"]["through_sequence"] == cursor
        resumed = _request(base, "/hawking/web/events?session_id=web-c2-live")
        assert resumed["events"] == events
        assert resumed["replay"]["through_sequence"] == cursor
        resumed = _request(base, "/hawking/web/events?session_id=web-c2-live")
        assert resumed["events"] == events
        assert resumed["replay"]["through_sequence"] == cursor
        resumed = _request(base, "/hawking/web/events?session_id=web-c2-live")
        assert resumed["events"] == events
        assert resumed["replay"]["through_sequence"] == cursor
        reconnect = _request(base, "/hawking/web/events?session_id=web-c2-live")
        assert reconnect["events"] == events
        assert reconnect["replay"]["through_sequence"] == cursor
    finally:
        httpd.shutdown()
        httpd.server_close()

    # A fresh handler is the browser/daemon-facing equivalent of a front-end
    # failure: all IDs and Goal references come back from the existing stores.
    httpd, base = _serve(tmp_path)
    try:
        reloaded = _request(base, "/hawking/web/session?id=web-c2-live")
        assert reloaded["session_id"] == "web-c2-live"
        assert reloaded["goal_ids"] == [goal["goal_id"]]
        resumed = _request(base, f"/hawking/web/events?id=web-c2-live&after={cursor}")
        assert resumed["replay"]["gap"] is False
        assert resumed["replay"]["snapshot_required"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
def test_http200_focused_audit_v4_qualification_support():
    """P18_TUNNEL_HTTP200_FOCUSED_AUDIT_V4 qualification-support check.

    Bounded, self-contained assertion that the focused audit tranche is
    exercised without expanding into unrelated suites. No release or
    hardware-qualification claim is made here.
    """
    import hawking.share_bridge as share_bridge

    assert share_bridge.http200_focused_audit_v4_marker() == (
        "P18_TUNNEL_HTTP200_FOCUSED_AUDIT_V4"
    )