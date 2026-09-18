"""H-Web markdown attachment admission and durable-reference contract tests."""
from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from hawking.auto_orchestration import worker_packet
from hawking.goal_surface import create_goal
from hawking.perception.store import ProjectStore
from hawking.serve import (
    _web_conversation_messages,
    _web_public_messages,
    make_handler,
)
from hawking.session import Session, SessionStore
from hawking.tool_registry import default_tool_registry
from hawking.web_artifacts import (
    MAX_WEB_ARTIFACT_BYTES,
    WebArtifactError,
    admit_bytes,
    read,
)


class _Backend:
    identity = "fixture"
    canonical_unbound = False

    def complete(self, payload, timeout=None):  # pragma: no cover - no remote call
        raise AssertionError("attachment endpoint fixture must not invoke cognition")


def _request(base: str, path: str, body: dict | None = None):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def _serve(root: Path):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            _Backend(),
            "fixture",
            greedy=False,
            health={"status": "ok", "resident": "fixture"},
            stores={"state_root": str(root)},
        ),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_markdown_bytes_use_the_existing_content_addressed_artifact_owner(
    tmp_path: Path,
) -> None:
    payload = b"# Attached notes\n\nThe requested outcome is harmless.\n"
    ref = admit_bytes(tmp_path, filename="../notes.md", payload=payload)

    assert ref["artifact_id"].startswith("artifact-")
    assert ref["filename"] == "notes.md"
    assert ref["media_type"] == "text/markdown"
    assert ref["status"] == "ADMITTED"
    assert ref["digest"]
    assert (tmp_path / ref["read_path"]).is_file()

    loaded = read(tmp_path, ref["artifact_id"])
    assert loaded["bytes"] == payload
    assert loaded["text"] == payload.decode("utf-8")
    assert loaded["malformed_utf8"] is False

    # A detached worker must consume the admitted reference through the real
    # Hawking fs.read boundary, not by reaching around the registry.
    tool_result = default_tool_registry(tmp_path, repo_root=tmp_path).invoke(
        "fs.read", {"path": ref["read_path"]}
    )
    assert tool_result.ok
    assert tool_result.to_dict()["value"]["content"] == payload.decode("utf-8")

    # The web adapter is not a second blob database: its record is the same
    # ProjectStore artifact row used by the existing perception owner.
    row = ProjectStore.open(tmp_path / ".hawking" / "web-artifacts").artifact(ref["digest"])
    assert row is not None
    assert row["media_type"] == "text/markdown"
    assert row["source_name"] == "notes.md"


def test_markdown_admission_is_bounded_and_decodes_malformed_utf8_explicitly(
    tmp_path: Path,
) -> None:
    malformed = admit_bytes(tmp_path, filename="broken.markdown", payload=b"# x\xff\n")
    loaded = read(tmp_path, malformed["artifact_id"])
    assert malformed["encoding"] == "utf-8-replacement"
    assert loaded["malformed_utf8"] is True
    assert "\ufffd" in loaded["text"]

    with pytest.raises(WebArtifactError, match="only .md"):
        admit_bytes(tmp_path, filename="notes.txt", payload=b"no")
    with pytest.raises(WebArtifactError, match="2 MiB"):
        admit_bytes(
            tmp_path,
            filename="large.md",
            payload=b"x" * (MAX_WEB_ARTIFACT_BYTES + 1),
        )
    with pytest.raises(WebArtifactError, match="not a Hawking artifact id"):
        read(tmp_path, "../../etc/passwd")


def test_session_and_worker_packet_keep_attachment_refs_separate_from_text(
    tmp_path: Path,
) -> None:
    ref = admit_bytes(tmp_path, filename="spec.md", payload=b"Requested outcome: one sentence.")
    session = Session("web-attachment", model="hawking/auto")
    session.workspace = str(tmp_path)
    session.append_message("user", "Read the attached specification.", attachments=[ref])
    SessionStore(str(tmp_path)).save(session)

    reloaded = SessionStore(str(tmp_path)).load(session.id)
    assert reloaded is not None
    reloaded.workspace = str(tmp_path)
    public = _web_public_messages(reloaded)
    assert public == [{
        "role": "user",
        "content": "Read the attached specification.",
        "attachments": [ref],
    }]
    projected = _web_conversation_messages(reloaded)
    assert projected[0]["content"].startswith("Read the attached specification.")
    assert "Requested outcome: one sentence." in projected[0]["content"]

    goal = create_goal(
        tmp_path,
        objective="Read the attached specification.",
        attachments=[ref],
        goal_mode="worker_research",
    )
    assert goal["attachments"] == [ref]
    packet = worker_packet(
        goal,
        {
            "goal_id": goal["goal_id"],
            "workunit_id": "WU-ATTACHMENT",
            "worker_model": "deepseek/deepseek-v4.1-flash",
            "worker_id": "hawking-worker-attachment",
            "state": "QUEUED",
        },
        authority={"capabilities": ["fs.read"]},
    )
    assert packet["attachments"] == [ref]
    assert ref["artifact_id"] in json.dumps(packet)


def test_live_local_admission_and_verified_read_endpoint_are_hawking_owned(
    tmp_path: Path,
) -> None:
    server, base = _serve(tmp_path)
    try:
        payload = b"# live fixture\n"
        code, _headers, raw = _request(
            base,
            "/hawking/web/attachments",
            {
                "filename": "live.md",
                "content_base64": base64.b64encode(payload).decode("ascii"),
            },
        )
        assert code == 201
        ref = json.loads(raw)["attachment"]
        assert ref["filename"] == "live.md"

        code, _headers, raw = _request(
            base,
            "/hawking/web/attachments?id=" + ref["artifact_id"],
        )
        assert code == 200
        assert json.loads(raw)["attachment"] == ref

        code, headers, raw = _request(
            base,
            "/hawking/web/attachments?id=" + ref["artifact_id"] + "&content=1",
        )
        assert code == 200
        assert headers.get_content_type() == "text/markdown"
        assert raw == payload

        code, _headers, raw = _request(
            base,
            "/hawking/web/attachments?id=../../etc/passwd",
        )
        assert code == 400
        assert "not a Hawking artifact id" in json.loads(raw)["error"]["message"]
    finally:
        server.shutdown()
        server.server_close()


def test_plan_export_is_a_durable_downloadable_markdown_result(
    tmp_path: Path,
) -> None:
    session = Session("web-plan-export", model="hawking/auto")
    session.workspace = str(tmp_path)
    session.append_message("user", "Write a plan.")
    session.append_message("assistant", "## Bounded plan\n\n1. Verify.")
    SessionStore(str(tmp_path)).save(session)

    server, base = _serve(tmp_path)
    try:
        code, _headers, raw = _request(
            base,
            "/hawking/web/markdown",
            {
                "session_id": session.id,
                "filename": "hawking-plan.md",
                "content": "## Bounded plan\n\n1. Verify.\n",
            },
        )
        assert code == 201
        exported = json.loads(raw)
        assert exported["schema"] == "hawking.web.markdown.v1"
        ref = exported["attachment"]
        assert ref["filename"] == "hawking-plan.md"
        assert exported["download_url"].endswith(
            "content=1&download=1"
        )

        code, headers, raw = _request(
            base,
            "/hawking/web/attachments?id="
            + ref["artifact_id"]
            + "&content=1&download=1",
        )
        assert code == 200
        assert headers.get_content_type() == "text/markdown"
        assert headers.get("Content-Disposition", "").endswith(
            "hawking-plan.md"
        )
        assert raw == b"## Bounded plan\n\n1. Verify.\n"

        reloaded = SessionStore(str(tmp_path)).load(session.id)
        assert reloaded is not None
        assert reloaded.messages[-1]["attachments"][0]["artifact_id"] == ref[
            "artifact_id"
        ]
        assert reloaded.messages[-1]["attachments"][0]["kind"] == "markdown_export"
        reloaded.workspace = str(tmp_path)
        projected = _web_conversation_messages(reloaded)
        assert "BEGIN HAWKING ATTACHED MARKDOWN" not in projected[-1]["content"]
    finally:
        server.shutdown()
        server.server_close()


def test_goal_attachment_read_path_is_trimmed_to_a_repo_relative_reference() -> None:
    """A Goal attachment reference must carry a clean repo-relative read path."""
    from hawking.goal_surface import _normalize_attachment_refs

    artifact_id = "artifact-" + "a" * 64
    digest = "b" * 64
    padded = "  .hawking/web-artifacts/artifacts/29/" + "a" * 64 + "  "

    refs = _normalize_attachment_refs([
        {
            "artifact_id": artifact_id,
            "digest": digest,
            "read_path": padded,
        }
    ])

    assert refs[0]["artifact_id"] == artifact_id
    assert refs[0]["read_path"] == ".hawking/web-artifacts/artifacts/29/" + "a" * 64
    assert refs[0]["read_path"] == refs[0]["read_path"].strip()
