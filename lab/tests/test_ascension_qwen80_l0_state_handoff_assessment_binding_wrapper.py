from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from lab.receipts import seal, verify
from lab.operators import ascension_qwen80_l0_state_handoff_assessment_binding_wrapper as wrapper


def _live_paths() -> dict[str, Path]:
    return {label: evidence.path.resolve() for label, evidence in wrapper.CANONICAL_EVIDENCE.items()}


def _build_live() -> dict[str, object]:
    paths = _live_paths()
    return wrapper.build_wrapper(
        assessment_path=paths["assessment"],
        outer_path=paths["outer"],
        inner_path=paths["inner"],
        release_path=paths["release"],
    )


def _expect_fixture_assessment(monkeypatch: pytest.MonkeyPatch, path: Path, document: dict[str, object]) -> None:
    raw = path.read_bytes()
    expected = dict(wrapper.CANONICAL_EVIDENCE)
    actual = wrapper.CANONICAL_EVIDENCE["assessment"]
    expected["assessment"] = wrapper.ExpectedEvidence(
        path=path.resolve(),
        schema=actual.schema,
        status=actual.status,
        seal_sha256=str(document["seal_sha256"]),
        document_sha256=wrapper._sha256(document),
        file_sha256=wrapper._sha256(raw),
        bytes=len(raw),
    )
    monkeypatch.setattr(wrapper, "CANONICAL_EVIDENCE", expected)


def _write_resealed_assessment(
    path: Path, mutate: Callable[[dict[str, object]], None]
) -> dict[str, object]:
    source = json.loads(wrapper.CANONICAL_EVIDENCE["assessment"].path.read_text(encoding="utf-8"))
    mutate(source)
    document = seal(source)
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return document


def test_live_chain_creates_a_sealed_provenance_only_binding(tmp_path: Path) -> None:
    document = _build_live()
    verify(document, label="assessment binding wrapper")

    assert document["schema"] == wrapper.WRAPPER_SCHEMA
    assert document["status"] == wrapper.WRAPPER_STATUS
    assert document["assessment_result_bound"] is True
    assert document["required_assessment"]["earned_status"] == wrapper.ASSESSMENT_STATUS
    assert document["post_capture_assessment"]["document_seal_sha256"] == wrapper.CANONICAL_EVIDENCE["assessment"].seal_sha256
    assert document["l0_outer_terminal"]["document_seal_sha256"] == wrapper.CANONICAL_EVIDENCE["outer"].seal_sha256
    assert document["l0_inner_capture"]["document_seal_sha256"] == wrapper.CANONICAL_EVIDENCE["inner"].seal_sha256
    assert document["lease_release_receipt"]["document_seal_sha256"] == wrapper.CANONICAL_EVIDENCE["release"].seal_sha256
    assert document["retained_l0_state_handoff"]["retained_l0_second_residual"]["bytes"] == 8_192
    assert document["future_joint_capture_requirement"]["fresh_l0_reencode_dispatches"] == 23
    assert document["future_joint_capture_requirement"]["future_l1_slot1_prefix_dispatches"] == 9
    assert document["future_joint_capture_requirement"]["historical_pinned_buffer_or_state_import_allowed"] is False
    assert document["authority_boundary"]["l1_dispatches_authorized"] == 0

    output = tmp_path / "binding.json"
    wrapper.write_new(output.resolve(), document)
    assert json.loads(output.read_text(encoding="utf-8"))["seal_sha256"] == document["seal_sha256"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        wrapper.write_new(output.resolve(), document)


def test_refuses_an_arbitrary_path_substitution(tmp_path: Path) -> None:
    paths = _live_paths()
    copied_assessment = tmp_path / "assessment-copy.json"
    copied_assessment.write_bytes(paths["assessment"].read_bytes())
    with pytest.raises(wrapper.AssessmentBindingError, match="historical substitution refused"):
        wrapper.build_wrapper(
            assessment_path=copied_assessment.resolve(),
            outer_path=paths["outer"],
            inner_path=paths["inner"],
            release_path=paths["release"],
        )


def test_refuses_a_valid_resealed_assessment_with_a_substituted_outer_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "assessment.json"
    document = _write_resealed_assessment(
        fixture,
        lambda source: source["l0_outer_terminal"].update({"document_sha256": "0" * 64}),
    )
    _expect_fixture_assessment(monkeypatch, fixture, document)
    paths = _live_paths()
    with pytest.raises(wrapper.AssessmentBindingError, match="assessment.l0_outer_terminal.document_sha256"):
        wrapper.build_wrapper(
            assessment_path=fixture.resolve(),
            outer_path=paths["outer"],
            inner_path=paths["inner"],
            release_path=paths["release"],
        )


def test_refuses_a_valid_resealed_assessment_that_rewrites_historical_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "assessment.json"
    document = _write_resealed_assessment(
        fixture,
        lambda source: source["l1_continuation_contract"].update(
            {
                "document_sha256": "c" * 64,
                "document_seal_sha256": "d" * 64,
            }
        ),
    )
    _expect_fixture_assessment(monkeypatch, fixture, document)
    paths = _live_paths()
    with pytest.raises(
        wrapper.AssessmentBindingError,
        match="assessment.l1_continuation_contract.document_sha256",
    ):
        wrapper.build_wrapper(
            assessment_path=fixture.resolve(),
            outer_path=paths["outer"],
            inner_path=paths["inner"],
            release_path=paths["release"],
        )
