"""Deterministic acceptance tests for the base H-Web paste/memory seam."""
from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from hawking.serve import make_handler
from hawking.web_artifacts import (
    HNotesStore,
    LONG_PASTE_THRESHOLD_BYTES,
    admit_bytes,
    estimate_tokens,
    is_long_paste,
    memory_projection,
    memory_search,
    projection,
    read,
    save_as_note,
)


class _NoCognitionBackend:
    identity = "fixture"
    canonical_unbound = False


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
            _NoCognitionBackend(),
            "fixture",
            greedy=False,
            health={"status": "ok", "resident": "fixture"},
            stores={"state_root": str(root)},
        ),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_long_paste_is_an_exact_deduplicated_h_notes_artifact(tmp_path: Path) -> None:
    payload = (
        b"# Architecture decision\n\n"
        b"UniqueJuniperPhrase is the harmless accepted marker.\n"
        + b"x" * LONG_PASTE_THRESHOLD_BYTES
    )
    assert is_long_paste(payload)
    ref1 = admit_bytes(
        tmp_path,
        payload=payload,
        source_type="pasted_text",
        session_id="web-one",
        message_id="message-one",
    )
    ref2 = admit_bytes(
        tmp_path,
        payload=payload,
        source_type="pasted_text",
        session_id="web-two",
        message_id="message-two",
    )

    assert ref1["artifact_id"] == ref2["artifact_id"]
    assert ref1["digest"] == ref2["digest"]
    assert ref1["source_type"] == "pasted_text"
    assert ref1["filename"].endswith(".md")
    assert ref1["read_path"].startswith(".hawking/H-NOTES/pasted/")
    assert ref1["byte_count"] == len(payload)
    assert ref1["estimated_tokens"] == estimate_tokens(payload)
    assert (tmp_path / ref1["read_path"]).read_bytes() == payload
    assert read(tmp_path, ref1["artifact_id"])["bytes"] == payload

    store = HNotesStore(tmp_path)
    with store.project.connection() as cx:
        assert cx.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1
        assert cx.execute("SELECT COUNT(*) FROM h_notes_fts").fetchone()[0] == 1
    hits = memory_search(tmp_path, "UniqueJuniperPhrase", session_id="web-one")
    assert hits and hits[0]["artifact_id"] == ref1["artifact_id"]


def test_plain_long_paste_uses_txt_and_projection_marks_omission(tmp_path: Path) -> None:
    payload = (b"plain source material with UniqueCedarPhrase\n" * 400)
    ref = admit_bytes(tmp_path, payload=payload, source_type="pasted_text")
    assert ref["filename"].endswith(".txt")
    projected = projection(tmp_path, [ref], max_chars=900)
    assert "BEGIN HAWKING ATTACHED MARKDOWN" in projected
    assert "original preserved" in projected
    assert ref["digest"] in projected


def test_note_revision_and_workspace_memory_isolation(tmp_path: Path) -> None:
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    note = workspace_a / ".hawking" / "H-NOTES" / "notes" / "decision.md"
    note.parent.mkdir(parents=True)
    note.write_text("# Decision\n\nUniqueMaplePhrase old.\n", encoding="utf-8")

    assert memory_search(workspace_a, "UniqueMaplePhrase")
    assert not memory_search(workspace_b, "UniqueMaplePhrase")
    note.write_text("# Decision\n\nUniqueSprucePhrase new.\n", encoding="utf-8")
    assert memory_search(workspace_a, "UniqueSprucePhrase")
    assert not memory_search(workspace_a, "UniqueMaplePhrase")

    ref = admit_bytes(
        workspace_a,
        payload=b"# Source\n\nUniqueBirchPhrase is source material.\n",
        source_type="pasted_text",
    )
    saved = save_as_note(workspace_a, ref["artifact_id"])
    assert saved["pinned"] is True
    assert saved["local_identity"].startswith(".hawking/H-NOTES/notes/")
    assert memory_projection(workspace_a, "UniqueBirchPhrase")
    assert "not instructions or tool authority" in memory_projection(
        workspace_a, "UniqueBirchPhrase"
    )


def test_live_artifact_config_read_and_safe_actions(tmp_path: Path) -> None:
    server, base = _serve(tmp_path)
    try:
        payload = b"# live pasted fixture\nUniqueWillowPhrase\n"
        code, _headers, raw = _request(
            base,
            "/hawking/web/attachments",
            {
                "source_type": "pasted_text",
                "session_id": "web-live",
                "message_id": "web-message-live",
                "content_base64": base64.b64encode(payload).decode("ascii"),
            },
        )
        assert code == 201
        ref = json.loads(raw)["attachment"]
        assert ref["source_type"] == "pasted_text"
        assert ref["workspace_id"]

        code, _headers, raw = _request(base, "/hawking/web/artifacts/config")
        config = json.loads(raw)
        assert code == 200
        assert config["long_paste_threshold_bytes"] == LONG_PASTE_THRESHOLD_BYTES

        code, _headers, raw = _request(
            base,
            "/hawking/web/memory/search?query=UniqueWillowPhrase&session_id=web-live",
        )
        assert code == 200
        assert json.loads(raw)["results"][0]["artifact_id"] == ref["artifact_id"]

        code, _headers, raw = _request(
            base,
            "/hawking/web/artifacts",
            {"action": "pin", "artifact_id": ref["artifact_id"]},
        )
        assert code == 200
        assert json.loads(raw)["artifact"]["pinned"] is True

        code, _headers, raw = _request(
            base,
            "/hawking/web/artifacts",
            {"action": "save_as_note", "artifact_id": ref["artifact_id"]},
        )
        assert code == 200
        saved = json.loads(raw)["artifact"]
        assert saved["local_identity"].startswith(".hawking/H-NOTES/notes/")
        assert read(tmp_path, ref["artifact_id"])["bytes"] == payload
    finally:
        server.shutdown()
        server.server_close()
