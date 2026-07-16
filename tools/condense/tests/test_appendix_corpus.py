from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "appendix_corpus.py"
SPEC = importlib.util.spec_from_file_location("appendix_corpus", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixture(root: pathlib.Path) -> None:
    (root / "a").mkdir()
    (root / "a" / "request.json").write_text("{}\n", encoding="utf-8")
    (root / "a" / "result.json").write_text('{"ok":true}\n', encoding="utf-8")
    (root / "failed.partial").write_bytes(b"partial")


def test_preview_is_name_only_and_index_round_trips(tmp_path: pathlib.Path) -> None:
    _fixture(tmp_path)
    preview = MODULE.preview(tmp_path)
    assert preview["opens_or_hashes_files"] is False
    assert preview["file_count"] == 3
    index = MODULE.build_index(
        tmp_path, active_owners=[], source_base_commit="0123456789abcdef"
    )
    assert MODULE.verify_index(index, active_owners=[]) == []
    assert index["semantic_counts"]["partial"] == 1
    assert index["contains_explicit_negative_failure_or_partial_evidence"] is True


def test_semantic_census_truthfully_allows_zero_negative_or_failure(tmp_path: pathlib.Path) -> None:
    (tmp_path / "neutral.json").write_text(
        '{"status":"complete","value":1}\n', encoding="utf-8"
    )
    index = MODULE.build_index(
        tmp_path, active_owners=[], source_base_commit="0123456789abcdef"
    )
    assert index["semantic_counts"] == {
        "negative_outcome": 0, "failure": 0, "partial": 0,
    }
    assert index["contains_explicit_negative_failure_or_partial_evidence"] is False
    assert MODULE.verify_index(index, active_owners=[]) == []


def test_semantic_census_rejects_false_negative_assertion(tmp_path: pathlib.Path) -> None:
    (tmp_path / "neutral.json").write_text('{"status":"complete"}\n', encoding="utf-8")
    index = MODULE.build_index(
        tmp_path, active_owners=[], source_base_commit="0123456789abcdef"
    )
    forged = copy.deepcopy(index)
    forged["entries"][0]["semantics"] = ["negative_outcome"]
    forged["semantic_counts"]["negative_outcome"] = 1
    forged["contains_explicit_negative_failure_or_partial_evidence"] = True
    forged = MODULE._stamp(forged)
    assert any(
        "semantic census mismatch" in error
        for error in MODULE.verify_index(forged, active_owners=[])
    )


def test_active_owner_blocks_hashing_and_tampering_is_detected(tmp_path: pathlib.Path) -> None:
    _fixture(tmp_path)
    try:
        MODULE.build_index(
            tmp_path, active_owners=[{"pid": 7}], source_base_commit="0123456789abcdef"
        )
    except RuntimeError as exc:
        assert "heavy owner" in str(exc)
    else:
        raise AssertionError("active owner did not block hashing")

    index = MODULE.build_index(
        tmp_path, active_owners=[], source_base_commit="0123456789abcdef"
    )
    (tmp_path / "a" / "result.json").write_text("tampered\n", encoding="utf-8")
    errors = MODULE.verify_index(index, active_owners=[])
    assert any("fingerprint mismatch" in error for error in errors)

    restamped = copy.deepcopy(index)
    restamped["entries"][0]["sha256"] = "0" * 64
    restamped = MODULE._stamp(restamped)
    assert MODULE.verify_index(restamped, active_owners=[])

    wrong_kind = copy.deepcopy(index)
    wrong_kind["entries"][0]["kind"] = "other_evidence"
    wrong_kind = MODULE._stamp(wrong_kind)
    assert any(
        "corpus kind mismatch" in error
        for error in MODULE.verify_index(wrong_kind, active_owners=[])
    )


def test_verification_rejects_files_added_after_freeze(tmp_path: pathlib.Path) -> None:
    _fixture(tmp_path)
    index = MODULE.build_index(
        tmp_path, active_owners=[], source_base_commit="0123456789abcdef"
    )
    (tmp_path / "late-evidence.json").write_text("{}\n", encoding="utf-8")
    errors = MODULE.verify_index(index, active_owners=[])
    assert any("unindexed corpus file: late-evidence.json" in error for error in errors)
