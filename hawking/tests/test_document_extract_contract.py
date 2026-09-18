import pytest

from hawking.perception.document_extract_contract import (
    DocumentExtractRequest, extraction_digest, red_before_green,
)


def test_request_is_artifact_bound_and_bounded():
    request = DocumentExtractRequest("artifact-" + "a" * 64, "b" * 64, "text/markdown", 12, "workspace-a")
    request.validate()
    with pytest.raises(ValueError):
        DocumentExtractRequest("/tmp/file", "b" * 64, "text/plain", 1, "workspace-a").validate()
    with pytest.raises(ValueError):
        DocumentExtractRequest("artifact-a", "b" * 64, "text/plain", 64 * 1024 * 1024 + 1, "workspace-a").validate()


def test_red_before_green_requires_matching_source_and_verification():
    baseline = {"status": "RED", "source_digest": "a" * 64}
    assert red_before_green(baseline, {"status": "GREEN", "source_digest": "b" * 64, "extracted_digest": "c", "verified": True})["ok"] is False
    result = red_before_green(baseline, {"status": "GREEN", "source_digest": "a" * 64, "extracted_digest": extraction_digest("fixture"), "verified": True})
    assert result["ok"] is True
    assert "OCR/capture capability" in result["claim_boundary"]
