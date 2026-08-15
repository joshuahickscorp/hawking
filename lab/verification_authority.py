#!/usr/bin/env python3.12
"""Verification authority + report-only model authority (Ascension Bible §2, §22).

Generalizes the pattern already proven in:
  - ``lab/operators/frankenstein_promotion_gate.py`` (PENDING / ACCEPT / REJECT;
    never fabricates ACCEPT; frozen targets; claim_boundary)
  - ``lab/operators/frankenstein_v0_seal.py`` independent ``verify`` (builder ≠
    challenger ≠ verifier; seal integrity)
  - ``lab/rules.py`` author-is-not-admitter
  - ``lab/receipts.GateEvidence`` (builder / challenger / verifier independence)
  - session grok-orchestration (Claude reviews, Grok proposes, nothing
    self-promotes)

Models emit *candidate* reports. Only the protected controller (or human
operator) may certify an *authoritative* classification. Sandbox models must
not declare PROMOTED / COMPLETE / PHYSICAL LIMIT / FINAL VERDICT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from lab.receipts import seal


CANDIDATE_REPORT_SCHEMA = "hawking.lab.candidate_report.v1"
AUTHORITATIVE_VERDICT_SCHEMA = "hawking.lab.authoritative_verdict.v1"
CLASSIFICATION_SCHEMA = "hawking.lab.result_classification.v1"


class ResultClassification(str, Enum):
    """Bible §22 result classes — every result is one of these."""

    PROMOTED_MECHANISM = "PROMOTED_MECHANISM"
    REJECTED_MECHANISM = "REJECTED_MECHANISM"
    TOOL_DEFECT = "TOOL_DEFECT"
    PLANNING_DEFECT = "PLANNING_DEFECT"
    VERIFIER_DEFECT = "VERIFIER_DEFECT"
    ENVIRONMENT_DEFECT = "ENVIRONMENT_DEFECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class CandidateKind(str, Enum):
    """Bible §2 — what sandbox models may emit."""

    CANDIDATE_MECHANISM = "candidate_mechanism"
    IMPLEMENTATION_RECEIPT = "implementation_receipt"
    PARITY_EVIDENCE = "parity_evidence"
    CAPABILITY_EVIDENCE = "capability_evidence"
    BENCHMARK_EVIDENCE = "benchmark_evidence"
    REVIEW_OBJECTION = "review_objection"
    KNOWN_LIMITATION = "known_limitation"
    RECOMMENDED_NEXT_EXPERIMENT = "recommended_next_experiment"
    REPETITION_FINGERPRINT = "repetition_fingerprint"
    ROADBLOCK_CANDIDATE = "ROADBLOCK_CANDIDATE"
    TG_RUNG_CANDIDATE = "TG_RUNG_CANDIDATE"


class ForbiddenAuthoritativeClaim(str, Enum):
    """Bible §2 — models may not authoritatively declare these."""

    PROMOTED = "PROMOTED"
    COMPLETE = "COMPLETE"
    PHYSICAL_LIMIT_REACHED = "PHYSICAL_LIMIT_REACHED"
    FAMILY_EXHAUSTED = "FAMILY_EXHAUSTED"
    SAFE_TO_DELETE = "SAFE_TO_DELETE"
    FINAL_VERDICT = "FINAL_VERDICT"


class AuthorityPrincipal(str, Enum):
    SANDBOX_MODEL = "sandbox_model"
    PROTECTED_CONTROLLER = "protected_controller"
    HUMAN_OPERATOR = "human_operator"


# Promotion-gate style aggregate (frankenstein_promotion_gate.evaluate_promotion)
# maps into §22 classifications without fabricating certainty.
_PROMOTION_GATE_TO_CLASSIFICATION: dict[str, ResultClassification] = {
    "ACCEPT": ResultClassification.PROMOTED_MECHANISM,
    "REJECT": ResultClassification.REJECTED_MECHANISM,
    "PENDING": ResultClassification.INSUFFICIENT_EVIDENCE,
}


class VerificationAuthorityError(PermissionError):
    """Raised when a principal exceeds report-only / certification bounds."""


class SelfPromotionError(VerificationAuthorityError):
    """Sandbox model attempted an authoritative claim or self-certification."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _principal(value: AuthorityPrincipal | str) -> AuthorityPrincipal:
    if isinstance(value, AuthorityPrincipal):
        return value
    return AuthorityPrincipal(str(value))


def _contains_forbidden_claim(text: str) -> ForbiddenAuthoritativeClaim | None:
    """Match forbidden claim tokens on word boundaries (avoid INCOMPLETE→COMPLETE)."""

    upper = text.upper()
    # Longer / multi-word phrases first.
    ordered = (
        ForbiddenAuthoritativeClaim.PHYSICAL_LIMIT_REACHED,
        ForbiddenAuthoritativeClaim.FAMILY_EXHAUSTED,
        ForbiddenAuthoritativeClaim.SAFE_TO_DELETE,
        ForbiddenAuthoritativeClaim.FINAL_VERDICT,
        ForbiddenAuthoritativeClaim.PROMOTED,
        ForbiddenAuthoritativeClaim.COMPLETE,
    )
    for claim in ordered:
        underscored = claim.value
        spaced = claim.value.replace("_", " ")
        for token in (underscored, spaced):
            pattern = r"(?<![A-Z0-9])" + re.escape(token) + r"(?![A-Z0-9])"
            if re.search(pattern, upper):
                return claim
    return None


def scan_forbidden_claims(payload: Mapping[str, Any]) -> list[str]:
    """Return forbidden claim tokens found in string fields of a candidate body.

    Scans top-level string values and common status keys. Does not walk the
    entire tree aggressively (candidate bodies may *discuss* the vocabulary);
    keys named ``verdict``, ``status``, ``declaration``, ``claim``,
    ``authoritative_claim``, ``final_verdict`` are treated as declarations.
    """

    declaration_keys = {
        "verdict",
        "status",
        "declaration",
        "claim",
        "authoritative_claim",
        "final_verdict",
        "terminal_state",
        "promotion",
    }
    found: list[str] = []
    for key, value in payload.items():
        if not isinstance(value, str):
            continue
        key_l = str(key).lower()
        if key_l in declaration_keys or key_l.endswith("_verdict") or key_l.endswith("_status"):
            claim = _contains_forbidden_claim(value)
            if claim is not None:
                # Candidate kinds are allowed as values of kind/type fields.
                if key_l in {"kind", "type", "candidate_kind"}:
                    continue
                # Allow explicit candidate-scoped labels.
                if value.upper().endswith("_CANDIDATE"):
                    continue
                if value.upper() in {c.value for c in CandidateKind}:
                    continue
                # INSUFFICIENT etc. are §22 classes — only forbidden claims.
                if claim.value in value.upper() or claim.value.replace("_", " ") in value.upper():
                    found.append(claim.value)
    return sorted(set(found))


@dataclass(frozen=True)
class CandidateReport:
    """Sandbox-model emission — report only, never a terminal verdict."""

    kind: CandidateKind
    author: str
    body: Mapping[str, Any]
    campaign_id: str = ""
    recorded_at: str = ""
    schema: str = CANDIDATE_REPORT_SCHEMA
    authority_level: str = "candidate"  # fixed; cannot be upgraded by author

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "authority_level": "candidate",
            "kind": self.kind.value,
            "author": self.author,
            "campaign_id": self.campaign_id,
            "recorded_at": self.recorded_at or _utc_now(),
            "body": dict(self.body),
            "claim_boundary": {
                "authoritative": False,
                "may_declare_promoted": False,
                "may_declare_complete": False,
                "may_declare_final_verdict": False,
                "may_declare_safe_to_delete": False,
                "certification_required_from": [
                    AuthorityPrincipal.PROTECTED_CONTROLLER.value,
                    AuthorityPrincipal.HUMAN_OPERATOR.value,
                ],
            },
        }

    def sealed(self) -> dict[str, Any]:
        return seal(self.to_dict())


@dataclass(frozen=True)
class AuthoritativeVerdict:
    """Protected-controller certification — the only path to §22 classes."""

    classification: ResultClassification
    certified_by: AuthorityPrincipal
    certifier_id: str
    reason: str
    candidate_seals: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    campaign_id: str = ""
    recorded_at: str = ""
    schema: str = AUTHORITATIVE_VERDICT_SCHEMA
    authority_level: str = "authoritative"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "authority_level": "authoritative",
            "classification": self.classification.value,
            "certified_by": self.certified_by.value,
            "certifier_id": self.certifier_id,
            "reason": self.reason,
            "candidate_seals": list(self.candidate_seals),
            "evidence_refs": list(self.evidence_refs),
            "campaign_id": self.campaign_id,
            "recorded_at": self.recorded_at or _utc_now(),
            "claim_boundary": {
                "authoritative": True,
                "sandbox_model_may_not_issue": True,
                "result_classification_schema": CLASSIFICATION_SCHEMA,
            },
        }

    def sealed(self) -> dict[str, Any]:
        return seal(self.to_dict())


def map_promotion_gate_verdict(verdict: str) -> ResultClassification:
    """Map frankenstein-style ACCEPT/REJECT/PENDING onto §22 classes.

    This is a pure mapping helper for bridge code. It does **not** promote
    anything — callers still need ``VerificationAuthority.certify``.
    """

    key = str(verdict).upper()
    if key not in _PROMOTION_GATE_TO_CLASSIFICATION:
        raise VerificationAuthorityError(
            f"unknown promotion-gate verdict {verdict!r}; "
            f"expected one of {sorted(_PROMOTION_GATE_TO_CLASSIFICATION)}"
        )
    return _PROMOTION_GATE_TO_CLASSIFICATION[key]


def classify_check_aggregate(
    statuses: Sequence[str],
) -> ResultClassification:
    """Generalize frankenstein_promotion_gate status aggregation.

    - any FAIL → REJECTED_MECHANISM
    - all PASS → PROMOTED_MECHANISM (still requires certify() to be authoritative)
    - otherwise → INSUFFICIENT_EVIDENCE (PENDING honesty; never fabricate ACCEPT)
    """

    normalized = {str(s).upper() for s in statuses}
    if "FAIL" in normalized:
        return ResultClassification.REJECTED_MECHANISM
    if normalized and normalized <= {"PASS"}:
        return ResultClassification.PROMOTED_MECHANISM
    return ResultClassification.INSUFFICIENT_EVIDENCE


@dataclass
class VerificationAuthority:
    """Enforces candidate-vs-authoritative boundary.

    Sandbox models: ``emit_candidate`` only.
    Protected controller / human: ``certify`` / ``sign_receipt`` / promote path.
    """

    controller_ids: frozenset[str] = frozenset({"protected_controller", "hawking_controller"})
    human_ids: frozenset[str] = frozenset({"human_operator", "owner"})

    def emit_candidate(
        self,
        *,
        principal: AuthorityPrincipal | str,
        author: str,
        kind: CandidateKind | str,
        body: Mapping[str, Any],
        campaign_id: str = "",
    ) -> dict[str, Any]:
        """Emit a sealed candidate report.

        Rejects sandbox attempts to smuggle forbidden authoritative claims in
        declaration fields, and rejects non-model principals pretending the
        emission is a certification (certify is a separate path).
        """

        principal_e = _principal(principal)
        kind_e = kind if isinstance(kind, CandidateKind) else CandidateKind(str(kind))

        if principal_e not in {
            AuthorityPrincipal.SANDBOX_MODEL,
            AuthorityPrincipal.PROTECTED_CONTROLLER,
            AuthorityPrincipal.HUMAN_OPERATOR,
        }:
            raise VerificationAuthorityError(f"unknown principal {principal_e!r}")

        # Even controllers use emit_candidate only for candidate-shaped notes;
        # their authoritative path is certify().
        forbidden = scan_forbidden_claims(body)
        if principal_e == AuthorityPrincipal.SANDBOX_MODEL and forbidden:
            raise SelfPromotionError(
                "sandbox model candidate body declares forbidden authoritative "
                f"claim(s): {forbidden}. Emit evidence; do not declare "
                "PROMOTED/COMPLETE/FINAL VERDICT (bible §2)."
            )

        # Reject bodies that set authority_level or classification as if certified.
        if body.get("authority_level") == "authoritative":
            raise SelfPromotionError(
                "candidate body may not set authority_level=authoritative"
            )
        if "classification" in body and principal_e == AuthorityPrincipal.SANDBOX_MODEL:
            raise SelfPromotionError(
                "sandbox model may not set §22 classification; only candidates. "
                "Protected controller certifies."
            )

        report = CandidateReport(
            kind=kind_e,
            author=author,
            body=dict(body),
            campaign_id=campaign_id,
            recorded_at=_utc_now(),
        )
        return report.sealed()

    def certify(
        self,
        *,
        principal: AuthorityPrincipal | str,
        certifier_id: str,
        classification: ResultClassification | str,
        reason: str,
        candidate_seals: Sequence[str] = (),
        evidence_refs: Sequence[str] = (),
        campaign_id: str = "",
    ) -> dict[str, Any]:
        """Issue an authoritative §22 classification.

        Sandbox models are hard-rejected (self-promotion / self-signing ban).
        """

        principal_e = _principal(principal)
        class_e = (
            classification
            if isinstance(classification, ResultClassification)
            else ResultClassification(str(classification))
        )

        if principal_e == AuthorityPrincipal.SANDBOX_MODEL:
            raise SelfPromotionError(
                "sandbox model may not certify authoritative verdicts "
                "(bible §2 / §22). Emit a candidate report instead."
            )

        if principal_e == AuthorityPrincipal.PROTECTED_CONTROLLER:
            allowed_ids = self.controller_ids | {
                AuthorityPrincipal.PROTECTED_CONTROLLER.value
            }
            if certifier_id not in allowed_ids:
                raise VerificationAuthorityError(
                    f"certifier_id {certifier_id!r} is not a registered "
                    f"protected controller id"
                )

        if principal_e == AuthorityPrincipal.HUMAN_OPERATOR:
            allowed_ids = self.human_ids | {AuthorityPrincipal.HUMAN_OPERATOR.value}
            if certifier_id not in allowed_ids:
                raise VerificationAuthorityError(
                    f"certifier_id {certifier_id!r} is not a registered human operator id"
                )

        verdict = AuthoritativeVerdict(
            classification=class_e,
            certified_by=principal_e,
            certifier_id=certifier_id,
            reason=reason,
            candidate_seals=tuple(candidate_seals),
            evidence_refs=tuple(evidence_refs),
            campaign_id=campaign_id,
            recorded_at=_utc_now(),
        )
        return verdict.sealed()

    def sign_receipt(
        self,
        *,
        principal: AuthorityPrincipal | str,
        signer_id: str,
        document: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Receipt signing authority — protected controller / human only.

        Mirrors §21 "sign their own results" ban and §22 "receipt signing".
        The document is re-sealed under an authority envelope; sandbox models
        cannot produce this envelope.
        """

        principal_e = _principal(principal)
        if principal_e == AuthorityPrincipal.SANDBOX_MODEL:
            raise SelfPromotionError(
                "sandbox model may not sign results (bible §21 deny-list / §22 "
                "receipt signing). Controllers sign; models only produce unsigned "
                "candidate evidence."
            )

        envelope = {
            "schema": "hawking.lab.signed_receipt_envelope.v1",
            "authority_level": "authoritative",
            "signed_by": principal_e.value,
            "signer_id": signer_id,
            "signed_at": _utc_now(),
            "document": dict(document),
        }
        return seal(envelope)

    def promote_mechanism(
        self,
        *,
        principal: AuthorityPrincipal | str,
        certifier_id: str,
        reason: str,
        candidate_seals: Sequence[str],
        evidence_refs: Sequence[str] = (),
        campaign_id: str = "",
    ) -> dict[str, Any]:
        """Convenience: certify PROMOTED_MECHANISM (controller-only)."""

        if not candidate_seals:
            raise VerificationAuthorityError(
                "promotion requires at least one candidate seal reference "
                "(no free-floating PROMOTED without evidence chain)"
            )
        return self.certify(
            principal=principal,
            certifier_id=certifier_id,
            classification=ResultClassification.PROMOTED_MECHANISM,
            reason=reason,
            candidate_seals=candidate_seals,
            evidence_refs=evidence_refs,
            campaign_id=campaign_id,
        )

    def reject_self_declared_promotion(
        self, payload: Mapping[str, Any]
    ) -> None:
        """Fail closed if a payload pretends to be an authoritative promotion.

        Used by ingestion paths that accept model outputs.
        """

        level = payload.get("authority_level")
        if level == "authoritative" and payload.get("certified_by") in {
            None,
            AuthorityPrincipal.SANDBOX_MODEL.value,
            "sandbox_model",
        }:
            raise SelfPromotionError(
                "payload claims authoritative level without protected certifier"
            )

        classification = payload.get("classification")
        if classification == ResultClassification.PROMOTED_MECHANISM.value:
            certified_by = payload.get("certified_by")
            if certified_by not in {
                AuthorityPrincipal.PROTECTED_CONTROLLER.value,
                AuthorityPrincipal.HUMAN_OPERATOR.value,
            }:
                raise SelfPromotionError(
                    "PROMOTED_MECHANISM requires protected_controller or "
                    "human_operator certification"
                )

        for key in ("verdict", "status", "declaration", "final_verdict"):
            value = payload.get(key)
            if isinstance(value, str):
                claim = _contains_forbidden_claim(value)
                if claim is not None and payload.get("authority_level") != "authoritative":
                    raise SelfPromotionError(
                        f"non-authoritative payload declares {claim.value}"
                    )


def default_authority() -> VerificationAuthority:
    return VerificationAuthority()
