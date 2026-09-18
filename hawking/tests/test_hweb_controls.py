"""Focused H-Web control tests for chat deletion and detached Goal safety."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from hawking.serve import make_handler
from hawking.session import Session, SessionStore


class _Backend:
    def complete(self, _payload, timeout=None):  # pragma: no cover - no cognition in this fixture
        raise AssertionError("H-Web control fixture must not invoke cognition")


def test_delete_web_conversation_removes_chat_and_preserves_goal_handles(tmp_path: Path) -> None:
    session = Session(session_id="web-delete-fixture", model="hawking/auto")
    session.ui = {
        "surface": "hawking-web",
        "goal_ids": ["GOAL-DETACHED"],
        "active_goal_id": "GOAL-DETACHED",
    }
    session.append_message("user", "disposable chat")
    store = SessionStore(str(tmp_path))
    store.save(session)

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            _Backend(),
            "fixture",
            greedy=False,
            health={"status": "ok", "resident": "fixture"},
            stores={"state_root": str(tmp_path)},
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/hawking/web/session",
            data=json.dumps({"action": "delete", "session_id": session.id}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read())
        assert payload["deleted"] is True
        assert payload["preserved_goal_ids"] == ["GOAL-DETACHED"]
        assert not (tmp_path / ".hawking" / "sessions" / f"{session.id}.json").exists()
        assert not store.history_path(session.id).exists()
    finally:
        server.shutdown()
        server.server_close()


def test_builder_session_endpoint_mints_only_for_its_canonical_workspace(tmp_path: Path) -> None:
    """Exercise the actual daemon route, not a launcher-only mock."""
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            _Backend(),
            "fixture",
            greedy=False,
            health={"status": "ok", "resident": "fixture"},
            stores={"state_root": str(tmp_path)},
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}/hawking/web/build/session"
    try:
        def post(workspace: Path):
            request = urllib.request.Request(
                endpoint,
                data=json.dumps({"workspace": str(workspace)}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())

        status, minted = post(tmp_path)
        assert status == 200
        assert minted["schema"] == "hawking.web.builder_handoff.v1"
        assert minted["handoff_url"].startswith("/hawking/web/build/web-")

        request = urllib.request.Request(
            endpoint,
            data=json.dumps({"workspace": str(tmp_path.parent / "wrong")}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(request)
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            payload = json.loads(exc.read())
            assert payload["error"]["message"] == "builder workspace does not match this Hawking daemon"
        else:  # pragma: no cover - failure path only
            raise AssertionError("foreign workspace unexpectedly minted BUILD authority")
    finally:
        server.shutdown()
        server.server_close()


def test_builder_session_accepts_a_samefile_workspace_alias(tmp_path: Path) -> None:
    """Workspace authority is filesystem identity, not its spelling."""
    alias = tmp_path.parent / "workspace-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            _Backend(), "fixture", greedy=False,
            health={"status": "ok", "resident": "fixture"},
            stores={"state_root": str(tmp_path)},
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/hawking/web/build/session",
            data=json.dumps({"workspace": str(alias)}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read())
        assert payload["schema"] == "hawking.web.builder_handoff.v1"
    finally:
        server.shutdown()
        server.server_close()


def test_read_surface_revokes_the_server_owned_builder_binding(tmp_path: Path) -> None:
    """`h web` is an authority downgrade, not merely a browser-cookie clear."""
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            _Backend(), "fixture", greedy=False,
            health={"status": "ok", "resident": "fixture"},
            stores={"state_root": str(tmp_path)},
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request = urllib.request.Request(
            base + "/hawking/web/build/session",
            data=json.dumps({"workspace": str(tmp_path)}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request) as response:
            handoff = json.loads(response.read())["handoff_url"]

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, request, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(_NoRedirect)
        try:
            opener.open(base + handoff)
        except urllib.error.HTTPError as exc:
            assert exc.code == 303
            cookie = exc.headers["Set-Cookie"].split(";", 1)[0]
        else:  # pragma: no cover - the handoff must always redirect
            raise AssertionError("builder handoff did not redirect")
        session_id = cookie.split("=", 1)[1].split(".", 1)[0]

        read_request = urllib.request.Request(
            base + "/hawking/web/read", headers={"Cookie": cookie}
        )
        try:
            opener.open(read_request)
        except urllib.error.HTTPError as exc:
            assert exc.code == 303
            assert "Max-Age=0" in exc.headers["Set-Cookie"]
        else:  # pragma: no cover - the read boundary must redirect home
            raise AssertionError("read boundary did not redirect")

        session = SessionStore(str(tmp_path)).load(session_id)
        assert session is not None
        assert session.ui["authority_profile"] == "read"
        assert session.ui["build_handoff"].get("revoked_at")
    finally:
        server.shutdown()
        server.server_close()


def test_hweb_event_snapshot_recovery_advances_cursor_once() -> None:
    """A busy journal must not make the browser reload its transcript forever."""
    asset = Path(__file__).resolve().parents[2] / "artifacts" / "hawking-review.html"
    source = asset.read_text(encoding="utf-8")
    assert "let webEventPollInFlight = false;" in source
    assert "await loadSession(sessionId, { startEvents: false });" in source
    assert "if (currentSessionId === sessionId) beginCanonicalEvents(through);" in source
    assert "if (options.startEvents !== false) beginCanonicalEvents();" in source


def test_hweb_goal_polling_cannot_leak_goals_between_sessions() -> None:
    """A late status response from the previous chat must be ignored."""
    asset = Path(__file__).resolve().parents[2] / "artifacts" / "hawking-review.html"
    source = asset.read_text(encoding="utf-8")
    assert "let goalPollSessionId = '';" in source
    assert "const loadToken = ++sessionLoadToken;" in source
    assert "trackedGoals = [];" in source
    assert "sessionId !== currentSessionId" in source
    assert "stopGoalPolling();" in source
