"""Tests for lab.verification_authority (bible §2 / §22)."""
from __future__ import annotations

import pytest

from lab.receipts import verify
from lab.verification_authority import (
    AuthorityPrincipal,
    CandidateKind,
    ForbiddenAuthoritativeClaim,
    ResultClassification,
    SelfPromotionError,
    VerificationAuthority,
    VerificationAuthorityError,
    classify_check_aggregate,
    default_authority,
    map_promotion_gate_verdict,
    scan_forbidden_claims,
)


@pytest.fixture
def auth() -> VerificationAuthority:
    return default_authority()


# ---------------------------------------------------------------------------
# §22 classification vocabulary
# ---------------------------------------------------------------------------


def test_result_classification_enum_is_complete() -> None:
    expected = {
        "PROMOTED_MECHANISM",
        "REJECTED_MECHANISM",
        "TOOL_DEFECT",
        "PLANNING_DEFECT",
        "VERIFIER_DEFECT",
        "ENVIRONMENT_DEFECT",
        "INSUFFICIENT_EVIDENCE",
    }
    assert {c.value for c in ResultClassification} == expected


def test_promotion_gate_mapping_matches_frankenstein_discipline() -> None:
    assert (
        map_promotion_gate_verdict("ACCEPT")
        is ResultClassification.PROMOTED_MECHANISM
    )
    assert (
        map_promotion_gate_verdict("REJECT")
        is ResultClassification.REJECTED_MECHANISM
    )
    assert (
        map_promotion_gate_verdict("PENDING")
        is ResultClassification.INSUFFICIENT_EVIDENCE
    )
    with pytest.raises(VerificationAuthorityError):
        map_promotion_gate_verdict("PROMOTED")  # not a gate verdict


def test_check_aggregate_never_fabricates_accept_from_partial() -> None:
    assert (
        classify_check_aggregate(["PASS", "PENDING"])
        is ResultClassification.INSUFFICIENT_EVIDENCE
    )
    assert (
        classify_check_aggregate(["PASS", "FAIL"])
        is ResultClassification.REJECTED_MECHANISM
    )
    assert (
        classify_check_aggregate(["PASS", "PASS"])
        is ResultClassification.PROMOTED_MECHANISM
    )
    assert classify_check_aggregate([]) is ResultClassification.INSUFFICIENT_EVIDENCE


# ---------------------------------------------------------------------------
# Candidate emission (models OK)
# ---------------------------------------------------------------------------


def test_sandbox_model_emits_candidate(auth: VerificationAuthority) -> None:
    sealed = auth.emit_candidate(
        principal=AuthorityPrincipal.SANDBOX_MODEL,
        author="grok-build",
        kind=CandidateKind.ROADBLOCK_CANDIDATE,
        body={
            "repeated_mechanism_class": "kv_codec",
            "unchanged_bottleneck": "state",
            "repetitions": 3,
            "note": "report only; not a final verdict",
        },
        campaign_id="camp-1",
    )
    verify(sealed, label="candidate")
    assert sealed["authority_level"] == "candidate"
    assert sealed["kind"] == "ROADBLOCK_CANDIDATE"
    assert sealed["claim_boundary"]["may_declare_promoted"] is False
    assert sealed["claim_boundary"]["authoritative"] is False


def test_sandbox_model_emits_tg_rung_candidate(auth: VerificationAuthority) -> None:
    sealed = auth.emit_candidate(
        principal=AuthorityPrincipal.SANDBOX_MODEL,
        author="executor",
        kind=CandidateKind.TG_RUNG_CANDIDATE,
        body={"rung": 4, "evidence": "synthetic"},
    )
    assert sealed["kind"] == "TG_RUNG_CANDIDATE"
    assert sealed["authority_level"] == "candidate"


@pytest.mark.parametrize(
    "status",
    [
        "PROMOTED",
        "COMPLETE",
        "PHYSICAL LIMIT REACHED",
        "FAMILY_EXHAUSTED",
        "SAFE TO DELETE",
        "FINAL VERDICT",
    ],
)
def test_sandbox_model_cannot_smuggle_forbidden_verdict(
    auth: VerificationAuthority, status: str
) -> None:
    with pytest.raises(SelfPromotionError, match="forbidden authoritative"):
        auth.emit_candidate(
            principal=AuthorityPrincipal.SANDBOX_MODEL,
            author="rogue",
            kind=CandidateKind.CANDIDATE_MECHANISM,
            body={"verdict": status, "detail": "I win"},
        )


def test_incomplete_is_not_complete_false_positive(auth: VerificationAuthority) -> None:
    # "INCOMPLETE" must not match forbidden COMPLETE.
    sealed = auth.emit_candidate(
        principal=AuthorityPrincipal.SANDBOX_MODEL,
        author="executor",
        kind=CandidateKind.KNOWN_LIMITATION,
        body={"status": "INCOMPLETE", "note": "work in progress"},
    )
    assert sealed["body"]["status"] == "INCOMPLETE"


def test_sandbox_cannot_set_classification_field(auth: VerificationAuthority) -> None:
    with pytest.raises(SelfPromotionError, match="classification"):
        auth.emit_candidate(
            principal=AuthorityPrincipal.SANDBOX_MODEL,
            author="rogue",
            kind=CandidateKind.CANDIDATE_MECHANISM,
            body={"classification": "PROMOTED_MECHANISM"},
        )


def test_sandbox_cannot_set_authority_level_authoritative(
    auth: VerificationAuthority,
) -> None:
    with pytest.raises(SelfPromotionError, match="authority_level"):
        auth.emit_candidate(
            principal=AuthorityPrincipal.SANDBOX_MODEL,
            author="rogue",
            kind=CandidateKind.CANDIDATE_MECHANISM,
            body={"authority_level": "authoritative", "note": "nope"},
        )


# ---------------------------------------------------------------------------
# Certification (controller only)
# ---------------------------------------------------------------------------


def test_sandbox_model_cannot_certify(auth: VerificationAuthority) -> None:
    with pytest.raises(SelfPromotionError, match="may not certify"):
        auth.certify(
            principal=AuthorityPrincipal.SANDBOX_MODEL,
            certifier_id="grok-build",
            classification=ResultClassification.PROMOTED_MECHANISM,
            reason="I promote myself",
        )


def test_protected_controller_can_certify_promotion(
    auth: VerificationAuthority,
) -> None:
    candidate = auth.emit_candidate(
        principal=AuthorityPrincipal.SANDBOX_MODEL,
        author="grok-build",
        kind=CandidateKind.CANDIDATE_MECHANISM,
        body={"mechanism": "route_residual", "scores_ref": "pending"},
    )
    verdict = auth.certify(
        principal=AuthorityPrincipal.PROTECTED_CONTROLLER,
        certifier_id="protected_controller",
        classification=ResultClassification.PROMOTED_MECHANISM,
        reason="held-out + independent challenge passed",
        candidate_seals=[candidate["seal_sha256"]],
        campaign_id="camp-1",
    )
    verify(verdict, label="verdict")
    assert verdict["authority_level"] == "authoritative"
    assert verdict["classification"] == "PROMOTED_MECHANISM"
    assert verdict["certified_by"] == "protected_controller"
    assert candidate["seal_sha256"] in verdict["candidate_seals"]


def test_promote_mechanism_requires_candidate_seals(
    auth: VerificationAuthority,
) -> None:
    with pytest.raises(VerificationAuthorityError, match="candidate seal"):
        auth.promote_mechanism(
            principal=AuthorityPrincipal.PROTECTED_CONTROLLER,
            certifier_id="protected_controller",
            reason="empty",
            candidate_seals=[],
        )


def test_human_can_certify_insufficient_evidence(auth: VerificationAuthority) -> None:
    verdict = auth.certify(
        principal=AuthorityPrincipal.HUMAN_OPERATOR,
        certifier_id="owner",
        classification=ResultClassification.INSUFFICIENT_EVIDENCE,
        reason="scores not sealed yet — PENDING honesty",
    )
    assert verdict["classification"] == "INSUFFICIENT_EVIDENCE"


def test_controller_can_classify_tool_defect(auth: VerificationAuthority) -> None:
    verdict = auth.certify(
        principal=AuthorityPrincipal.PROTECTED_CONTROLLER,
        certifier_id="hawking_controller",
        classification=ResultClassification.TOOL_DEFECT,
        reason="harness crashed before scoring; not a mechanism reject",
    )
    assert verdict["classification"] == "TOOL_DEFECT"


# ---------------------------------------------------------------------------
# Sign own results ban
# ---------------------------------------------------------------------------


def test_sandbox_model_cannot_sign_receipt(auth: VerificationAuthority) -> None:
    with pytest.raises(SelfPromotionError, match="may not sign"):
        auth.sign_receipt(
            principal=AuthorityPrincipal.SANDBOX_MODEL,
            signer_id="grok-build",
            document={"schema": "x", "result": "ok"},
        )


def test_controller_can_sign_receipt(auth: VerificationAuthority) -> None:
    envelope = auth.sign_receipt(
        principal=AuthorityPrincipal.PROTECTED_CONTROLLER,
        signer_id="protected_controller",
        document={"schema": "x", "result": "ok"},
    )
    verify(envelope, label="signed")
    assert envelope["authority_level"] == "authoritative"
    assert envelope["signed_by"] == "protected_controller"


# ---------------------------------------------------------------------------
# Ingestion guard
# ---------------------------------------------------------------------------


def test_reject_self_declared_promotion_payload(auth: VerificationAuthority) -> None:
    with pytest.raises(SelfPromotionError):
        auth.reject_self_declared_promotion(
            {
                "classification": "PROMOTED_MECHANISM",
                "certified_by": "sandbox_model",
            }
        )
    with pytest.raises(SelfPromotionError):
        auth.reject_self_declared_promotion(
            {"verdict": "PROMOTED", "authority_level": "candidate"}
        )


def test_scan_forbidden_claims_on_declaration_keys_only() -> None:
    # Discussion in a non-declaration field is not scanned as a claim.
    assert scan_forbidden_claims({"note": "we hope for PROMOTED someday"}) == []
    assert scan_forbidden_claims({"verdict": "PROMOTED"}) == [
        ForbiddenAuthoritativeClaim.PROMOTED.value
    ]
