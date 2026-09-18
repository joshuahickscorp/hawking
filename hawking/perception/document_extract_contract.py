"""Typed, fail-closed contract for the future ``document.extract`` door.

This is an admission/verification seam only.  It does not read arbitrary
paths, invoke OCR, or claim that document perception is implemented.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping


SCHEMA = "hawking.document_extract.contract.v1"
MAX_SOURCE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class DocumentExtractRequest:
    source_artifact_id: str
    source_digest: str
    media_type: str
    byte_count: int
    workspace_id: str

    def validate(self) -> None:
        if not self.source_artifact_id or not self.source_artifact_id.startswith("artifact-"):
            raise ValueError("document.extract requires a Hawking artifact")
        if len(self.source_digest) != 64 or any(c not in "0123456789abcdef" for c in self.source_digest.lower()):
            raise ValueError("document.extract source_digest must be sha256")
        if not self.media_type or "/" not in self.media_type:
            raise ValueError("document.extract media_type is required")
        if not 0 <= int(self.byte_count) <= MAX_SOURCE_BYTES:
            raise ValueError("document.extract source exceeds bounded size")
        if not self.workspace_id.strip():
            raise ValueError("document.extract workspace is required")


def red_before_green(baseline: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the minimum evidence order without accepting extractor claims."""
    baseline_digest = str(baseline.get("source_digest") or "").lower()
    result_digest = str(result.get("source_digest") or "").lower()
    red = str(baseline.get("status") or "").upper() == "RED" and bool(baseline_digest)
    green = (
        red
        and str(result.get("status") or "").upper() == "GREEN"
        and result_digest == baseline_digest
        and bool(str(result.get("extracted_digest") or ""))
        and result.get("verified") is True
    )
    return {
        "schema": SCHEMA,
        "ok": bool(red and green),
        "red": red,
        "green": green,
        "source_digest": baseline_digest,
        "extracted_digest": str(result.get("extracted_digest") or ""),
        "claim_boundary": "contract evidence only; no OCR/capture capability is implied",
    }


def extraction_digest(text: str) -> str:
    """Canonical digest helper for a future bounded extractor result."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


__all__ = ["DocumentExtractRequest", "SCHEMA", "extraction_digest", "red_before_green"]
