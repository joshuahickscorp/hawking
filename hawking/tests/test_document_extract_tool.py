from hawking.perception.host import Host
from hawking.perception.tools import register
from hawking.perception.document_extract_contract import extraction_digest, red_before_green


def test_document_extract_is_artifact_bound_and_text_only(tmp_path):
    host = register(Host(tmp_path))
    project = tmp_path / "p"
    project.mkdir()
    store = __import__("hawking.perception.store", fromlist=["ProjectStore"]).ProjectStore.create(project, "p")
    artifact = store.ingest_bytes(b"# benign fixture\n", media_type="text/markdown", source_name="fixture.md")
    tool = host._tool_manager.get_tool("document.extract")
    result = tool.fn(project_path="p", digest=artifact["digest"], media_type="text/markdown")
    assert result["status"] == "GREEN"
    assert result["verified"] is True
    refused = tool.fn(project_path="p", digest=artifact["digest"], media_type="image/png")
    assert refused["status"] == "UNSUPPORTED_MEDIA"


def test_document_extract_refuses_bounds_and_unknown_or_malformed_artifacts(tmp_path):
    host = register(Host(tmp_path))
    project = tmp_path / "p"
    project.mkdir()
    from hawking.perception.store import ProjectStore
    store = ProjectStore.create(project, "p")
    artifact = store.ingest_bytes(b"bounded", media_type="text/plain")
    tool = host._tool_manager.get_tool("document.extract")
    bounded = tool.fn(project_path="p", digest=artifact["digest"], media_type="text/plain", max_bytes=1)
    assert bounded["status"] == "REFUSED_BOUNDS"
    try:
        tool.fn(project_path="p", digest="f" * 64, media_type="text/plain")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("unadmitted digest must fail closed")
    try:
        tool.fn(project_path="p", digest="bad", media_type="text/plain")
    except ValueError:
        pass
    else:
        raise AssertionError("malformed digest must fail closed")


def test_document_extract_result_satisfies_red_before_green_contract():
    digest = "a" * 64
    result = red_before_green(
        {"status": "RED", "source_digest": digest},
        {"status": "GREEN", "source_digest": digest,
         "extracted_digest": extraction_digest("fixture"), "verified": True},
    )
    assert result["ok"] is True
