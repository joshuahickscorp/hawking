from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from hawking.goal_surface import create_goal, load_goal
from hawking.serve import make_handler
from hawking.session import Session, SessionStore
from hawking.share_bridge import (
    ShareError,
    act_share,
    create_share,
    create_tunnel_invitation,
    decode_tunnel_frame,
    disconnect_tunnel,
    encode_tunnel_frame,
    manifest_share,
    pair_tunnel,
    reconnect_tunnel,
    request_tunnel,
    revoke_share,
    snapshot_share,
)
from hawking.workunit_owner import WorkunitOwner, WorkunitRecord


class _Backend:
    def complete(self, _payload, timeout=None):  # pragma: no cover - tunnel is control-plane only
        raise AssertionError("Tunnel fixture must not invoke cognition")


def _post(base: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _post_with_status(base: str, path: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read())


def _goal_workspace(tmp_path):
    goal = create_goal(
        tmp_path,
        objective="Review Bearer do-not-leak-sk-0123456789012",
        worker_policy="auto",
    )
    goal_id = goal["goal_id"]
    owner = WorkunitOwner(tmp_path)
    record = WorkunitRecord(
        workunit_id=goal_id,
        goal_id=goal_id,
        objective=goal["objective"],
        staging_workspace=str(tmp_path),
        checkpoint_path=str(tmp_path / "checkpoint.json"),
    )
    owner.save(record)
    return goal_id


def test_goal_share_is_short_hashed_and_redacted(tmp_path):
    goal_id = _goal_workspace(tmp_path)
    created = create_share(
        tmp_path,
        target_type="goal",
        target_id=goal_id,
        permissions=["read"],
    )
    token = created["share_id"]
    assert len(token) <= 13
    files = list((tmp_path / ".hawking" / "shares").glob("*.json"))
    assert len(files) == 1
    assert token not in files[0].name
    assert token not in files[0].read_text(encoding="utf-8")
    snapshot = snapshot_share(tmp_path, token)
    encoded = json.dumps(snapshot)
    assert "Bearer [redacted]" in encoded
    assert "sk-0123456789012" not in encoded
    assert manifest_share(tmp_path, token)["actions"] == {}


def test_share_permissions_idempotency_steer_and_revoke(tmp_path):
    goal_id = _goal_workspace(tmp_path)
    created = create_share(
        tmp_path,
        target_type="goal",
        target_id=goal_id,
        permissions=["read", "comment", "steer"],
    )
    token = created["share_id"]
    with pytest.raises(ShareError) as denied:
        act_share(
            tmp_path,
            token,
            action="comment",
            capability="wrong-token",
            message="comment",
            idempotency_key="a",
        )
    assert denied.value.status == 403
    first = act_share(
        tmp_path,
        token,
        action="comment",
        capability=token,
        message="keep the route margin",
        idempotency_key="comment-1",
    )
    duplicate = act_share(
        tmp_path,
        token,
        action="comment",
        capability=token,
        message="different text is ignored",
        idempotency_key="comment-1",
    )
    assert first["accepted"] is True
    assert duplicate["duplicate"] is True
    assert load_goal(tmp_path, goal_id)["external_comments"][0]["message"] == "keep the route margin"
    steered = act_share(
        tmp_path,
        token,
        action="steer",
        capability=token,
        message="finish the focused check",
        idempotency_key="steer-1",
    )
    assert steered["accepted"] is True
    assert load_goal(tmp_path, goal_id)["last_operator_action"] == "share_steer"
    revoked = revoke_share(tmp_path, token)
    assert revoked["revoked"] is True
    with pytest.raises(ShareError) as gone:
        snapshot_share(tmp_path, token)
    assert gone.value.status == 410


def test_expired_share_is_not_readable(tmp_path):
    goal_id = _goal_workspace(tmp_path)
    created = create_share(tmp_path, target_type="goal", target_id=goal_id)
    token = created["share_id"]
    path = next((tmp_path / ".hawking" / "shares").glob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["expires_at"] = time.time() - 1
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ShareError) as expired:
        snapshot_share(tmp_path, token)
    assert expired.value.status == 410


def test_conversation_share_is_general_chat_and_persists_annotations(tmp_path):
    session = Session(session_id="web-share-test", model="hawking/auto")
    session.ui = {"surface": "hawking-web", "goal_ids": []}
    session.append_message("user", "Keep this chat scoped to the Hawking workspace.")
    session.append_message("assistant", "The conversation remains in Hawking.")
    SessionStore(str(tmp_path)).save(session)

    created = create_share(
        tmp_path,
        target_type="conversation",
        target_id=session.id,
        permissions=["read", "comment", "steer"],
    )
    token = created["share_id"]
    snapshot = snapshot_share(tmp_path, token)
    assert snapshot["kind"] == "hawking.conversation.share"
    assert snapshot["messages"][-1]["role"] == "assistant"
    assert snapshot["external_comments"] == []

    commented = act_share(
        tmp_path,
        token,
        action="comment",
        capability=token,
        message="Keep the shared chat space concise.",
        idempotency_key="conversation-comment-1",
    )
    steered = act_share(
        tmp_path,
        token,
        action="steer",
        capability=token,
        message="Continue from the current chat state.",
        idempotency_key="conversation-steer-1",
    )
    assert commented["scope"] == "conversation"
    assert steered["scope"] == "conversation"
    reloaded = SessionStore(str(tmp_path)).load(session.id)
    assert reloaded is not None
    assert reloaded.ui["external_comments"][-1]["message"] == "Keep the shared chat space concise."
    assert reloaded.ui["external_steering"][-1]["message"] == "Continue from the current chat state."
    assert reloaded.steering[-1] == "Continue from the current chat state."


def test_tunnel_pair_intersects_grants_and_reconnects_idempotently(tmp_path):
    goal_id = _goal_workspace(tmp_path)
    created = create_share(
        tmp_path,
        target_type="goal",
        target_id=goal_id,
        permissions=["read", "prompt", "build"],
    )
    token = created["share_id"]
    invitation = create_tunnel_invitation(
        tmp_path,
        token,
        capability=token,
        direct_available=True,
        relay_available=True,
    )
    paired = pair_tunnel(
        tmp_path,
        token,
        invitation=invitation["invitation"],
        device_id="browser.test-device",
        requested_permissions=["read", "prompt", "build"],
        session_permissions=["read", "prompt"],
    )
    assert paired["transport"] == "direct_p2p"
    frame = encode_tunnel_frame(
        paired["reconnect_secret"],
        request_id="frame-1",
        sequence=1,
        payload={"message": "secret prompt"},
    )
    assert "secret prompt" not in json.dumps(frame)
    assert decode_tunnel_frame(paired["reconnect_secret"], frame) == {
        "message": "secret prompt"
    }
    with pytest.raises(ShareError) as bad_frame:
        decode_tunnel_frame("wrong-secret-0123456789", frame)
    assert bad_frame.value.status == 403
    assert paired["granted_permissions"] == ["prompt", "read"]
    assert "stream_prompt" in paired["operations"]
    assert "scoped_build" not in paired["operations"]
    with pytest.raises(ShareError) as reused_invitation:
        pair_tunnel(
            tmp_path,
            token,
            invitation=invitation["invitation"],
            device_id="second-device",
        )
    assert reused_invitation.value.status == 410

    models = request_tunnel(
        tmp_path,
        token,
        reconnect_secret=paired["reconnect_secret"],
        request_id="models-1",
        operation="list_models",
    )
    duplicate = request_tunnel(
        tmp_path,
        token,
        reconnect_secret=paired["reconnect_secret"],
        request_id="models-1",
        operation="list_models",
    )
    assert models["models"]
    assert duplicate["duplicate"] is True
    prompt = request_tunnel(
        tmp_path,
        token,
        reconnect_secret=paired["reconnect_secret"],
        request_id="prompt-1",
        operation="stream_prompt",
        payload={"message": "Continue the scoped Goal review."},
    )
    assert prompt["accepted"] is True
    assert prompt["execution_state"] == "ADMITTED_TO_HWEB"
    with pytest.raises(ShareError) as build_denied:
        request_tunnel(
            tmp_path,
            token,
            reconnect_secret=paired["reconnect_secret"],
            request_id="build-1",
            operation="scoped_build",
            payload={"message": "edit the repository"},
        )
    assert build_denied.value.status == 403

    disconnected = disconnect_tunnel(
        tmp_path,
        token,
        reconnect_secret=paired["reconnect_secret"],
        request_id="disconnect-1",
    )
    duplicate_disconnect = disconnect_tunnel(
        tmp_path,
        token,
        reconnect_secret=paired["reconnect_secret"],
        request_id="disconnect-1",
    )
    assert disconnected["connected"] is False
    assert duplicate_disconnect["duplicate"] is True
    resumed = reconnect_tunnel(
        tmp_path,
        token,
        reconnect_secret=paired["reconnect_secret"],
        after_sequence=paired["cursor"],
    )
    assert resumed["connection"]["connected"] is True
    assert resumed["events"]
    assert resumed["gap"] is False

    record_path = next((tmp_path / ".hawking" / "shares").glob("*.json"))
    record_text = record_path.read_text(encoding="utf-8")
    assert invitation["invitation"] not in record_text
    assert paired["reconnect_secret"] not in record_text


def test_tunnel_relay_fallback_and_share_revocation_fail_closed(tmp_path):
    goal_id = _goal_workspace(tmp_path)
    created = create_share(
        tmp_path,
        target_type="goal",
        target_id=goal_id,
        permissions=["read"],
    )
    token = created["share_id"]
    invitation = create_tunnel_invitation(
        tmp_path,
        token,
        capability=token,
        direct_available=False,
        relay_available=True,
    )
    paired = pair_tunnel(
        tmp_path,
        token,
        invitation=invitation["invitation"],
        device_id="relay-browser",
    )
    assert paired["transport"] == "encrypted_relay"
    revoke_share(tmp_path, token)
    with pytest.raises(ShareError) as revoked:
        reconnect_tunnel(
            tmp_path,
            token,
            reconnect_secret=paired["reconnect_secret"],
            after_sequence=0,
        )
    assert revoked.value.status == 410


def test_tunnel_hweb_host_and_public_pair_routes_use_one_share_owner(tmp_path: Path):
    goal_id = _goal_workspace(tmp_path)
    created = create_share(
        tmp_path,
        target_type="goal",
        target_id=goal_id,
        permissions=["read"],
    )
    token = created["share_id"]
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
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        invite_status, invitation = _post_with_status(base, "/hawking/web/tunnel", {
            "action": "invite",
            "share_id": token,
            "capability": token,
            "direct_available": False,
            "relay_available": True,
        })
        pair_status, paired = _post_with_status(base, f"/s/{token}/tunnel/pair", {
            "invitation": invitation["invitation"],
            "device_id": "browser-http",
            "requested_permissions": ["read"],
            "session_permissions": ["read"],
        })
        request_status, models = _post_with_status(base, f"/s/{token}/tunnel/request", {
            "reconnect_secret": paired["reconnect_secret"],
            "request_id": "http-models-1",
            "operation": "list_models",
        })
        assert invite_status == 201
        assert pair_status == 200
        assert request_status == 200
        assert paired["transport"] == "encrypted_relay"
        assert models["models"]
    finally:
        server.shutdown()
        server.server_close()
def test_tunnel_implementation_ready_derives_from_health():
    from hawking.share_bridge import (
        tunnel_implementation_ready,
        with_tunnel_implementation_ready,
    )

    assert tunnel_implementation_ready({"connected": True, "degraded": False}) is True
    assert tunnel_implementation_ready({"connected": True, "degraded": True}) is False
    assert tunnel_implementation_ready({"connected": False, "degraded": False}) is False

    enriched = with_tunnel_implementation_ready({"connected": True, "degraded": False})
    assert enriched["tunnel_implementation_ready"] is True
    assert enriched["connected"] is True