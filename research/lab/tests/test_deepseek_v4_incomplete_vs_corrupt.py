"""An absent chunk body must be resumable; a corrupt one must never be.

Regression guard for the trap state that stranded the full-43-layer DSV4F
stream: the sealed manifest survived, the 148.65 GiB body did not, and every
recovery path reported the missing body as a containment violation
("chunk path escapes artifact") and refused to resume.  Three guards had to
agree that missing != corrupt before the stream could be re-fetched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.operators.deepseek_v4_gravity import (
    RANGE_SCHEMA,
    DeepSeekV4GravityError,
    DeepSeekV4GravityIncompleteError,
    _completed_ranges,
    reverify_full_model,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact_with_range(tmp_path: Path, *, body: bytes | None) -> tuple[Path, dict]:
    """Artifact holding one journaled range; `body=None` means the chunk is absent."""
    artifact = tmp_path / "art"
    (artifact / "chunks" / "00").mkdir(parents=True)
    payload = b"deepseek-v4-chunk"
    relpath = f"chunks/00/{_sha256(payload)}"
    if body is not None:
        (artifact / relpath).write_bytes(body)
    row = {
        "schema": RANGE_SCHEMA,
        "status": "SEALED",
        "range_id": "shard-0:0-17",
        "chunk_relpath": relpath,
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }
    ranges = artifact / "stream-ranges.jsonl"
    ranges.write_text(json.dumps(row, sort_keys=True) + "\n")
    return artifact, {"ranges": ranges}


def test_absent_chunk_is_not_a_completed_range(tmp_path: Path) -> None:
    """Missing body => range simply not completed, so the stream re-fetches it."""
    artifact, paths = _artifact_with_range(tmp_path, body=None)
    assert _completed_ranges(paths, artifact) == {}


def test_present_and_matching_chunk_stays_completed(tmp_path: Path) -> None:
    artifact, paths = _artifact_with_range(tmp_path, body=b"deepseek-v4-chunk")
    completed = _completed_ranges(paths, artifact)
    assert list(completed) == ["shard-0:0-17"]


def test_corrupt_chunk_still_hard_fails(tmp_path: Path) -> None:
    """Present but wrong content is corruption, never a resumable absence."""
    artifact, paths = _artifact_with_range(tmp_path, body=b"deepseek-v4-WRONG")
    with pytest.raises(DeepSeekV4GravityError) as caught:
        _completed_ranges(paths, artifact)
    assert not isinstance(caught.value, DeepSeekV4GravityIncompleteError)


def test_live_stranded_artifact_reports_absence_not_escape() -> None:
    """The real artifact that motivated this: manifest sealed, body reclaimed."""
    artifact = Path(
        "/Users/scammermike/Downloads/hawking/workspace/campaign/records/runs"
        "/deepseek-v4/full-43-layer-stream.gravity"
    )
    if not (artifact / "manifest.json").is_file():
        pytest.skip("stranded DSV4F artifact not present on this machine")
    try:
        reverify_full_model(artifact)
    except DeepSeekV4GravityIncompleteError as exc:
        assert "absent" in str(exc)
        assert "escapes artifact" not in str(exc)
    except DeepSeekV4GravityError as exc:  # pragma: no cover - body fully restored
        assert "escapes artifact" not in str(exc), (
            "a missing chunk must never be reported as a containment violation"
        )
    # A fully restored body verifying cleanly is also a pass.


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
