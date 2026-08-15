"""Regression coverage for stable Qwen80 source-audit authority artifacts."""
from __future__ import annotations

from lab.operators.ascension_qwen80_acquisition import (
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    Qwen80Acquisition,
)
from lab.receipts import seal


def test_equivalent_rehashed_source_body_reuses_existing_audit_without_reseal(tmp_path) -> None:
    target = tmp_path / "Qwen3-Coder-Next"
    worker = Qwen80Acquisition(
        metadata_path=tmp_path / "metadata.json",
        target=target,
        root=tmp_path / "acquisition",
        workers=1,
    )
    audited = [
        {"path": "config.json", "kind": "control", "bytes": 7, "sha256": "a" * 64},
        {
            "path": "model-00001-of-00001.safetensors",
            "kind": "weight",
            "bytes": 13,
            "sha256": "b" * 64,
        },
    ]
    totals = {
        "file_count": 2,
        "bytes": 20,
        "weight_file_count": 1,
        "weight_bytes": 13,
    }
    metadata = {"seal_sha256": "c" * 64}
    existing = seal(
        {
            "schema": "hawking.ascension.qwen80_source_body_audit_candidate.v1",
            "status": "CANDIDATE_FULL_PINNED_SOURCE_BODY_VERIFIED",
            "recorded_at": "2026-08-08T00:00:00Z",
            "source_admission_seal_sha256": metadata["seal_sha256"],
            "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
            "target_directory": str(target),
            "files": audited,
            "totals": totals,
            "reserve_at_start": {"free_before_bytes": 1},
        }
    )

    assert worker._existing_audit_matches(
        existing, metadata=metadata, audited=audited, totals=totals
    )

    changed_file = [dict(row) for row in audited]
    changed_file[1]["sha256"] = "d" * 64
    assert not worker._existing_audit_matches(
        existing, metadata=metadata, audited=changed_file, totals=totals
    )
    assert not worker._existing_audit_matches(
        existing,
        metadata={"seal_sha256": "e" * 64},
        audited=audited,
        totals=totals,
    )
