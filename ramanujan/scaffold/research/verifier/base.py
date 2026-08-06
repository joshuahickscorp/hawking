"""Shared verifier vocabulary.

A backend may only return ACCEPTED when it actually checked something.  Uncertainty
and missing toolchains are first-class outcomes, never silent passes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class Verdict(str, Enum):
    """Outcome of one verification attempt."""

    ACCEPTED = "ACCEPTED"  # independently checked and matched
    REJECTED = "REJECTED"  # checked and failed
    UNCERTAIN = "UNCERTAIN"  # could not decide (malformed input, ambiguous)
    UNAVAILABLE = "UNAVAILABLE"  # backend / toolchain not present; never counts as ACCEPTED


class BackendStatus(str, Enum):
    """Whether this host can actually run the backend."""

    REAL = "REAL"  # executable here with a real check
    GATED = "GATED"  # interface present; requires external resource (Lean image, GLM, …)
    ABSENT = "ABSENT"  # not installed; fails closed


@dataclass(frozen=True)
class BackendAvailability:
    backend_id: str
    status: BackendStatus
    detail: str
    version: str | None = None
    capabilities: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "status": self.status.value,
            "detail": self.detail,
            "version": self.version,
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class VerificationRequest:
    """What the student claims, plus enough structure for an independent check.

    The verifier recomputes from ``problem`` fields.  Callers must not treat a
    pre-stored expected answer as the sole oracle when a recomputation path
    exists — backends that recompute ignore ``claimed_answer`` equality alone.
    """

    problem_id: str
    statement: str
    kind: str  # "exact_numeric" | "sympy" | "lean_capsule" | ...
    claimed_answer: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    # Optional independent oracle for fixtures / exact-check problems where the
    # recomputation expression is in payload["expression"].
    checker_id: str = "host_verifier"

    def as_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "statement": self.statement,
            "kind": self.kind,
            "claimed_answer": self.claimed_answer,
            "payload": dict(self.payload),
            "checker_id": self.checker_id,
        }


@dataclass(frozen=True)
class VerificationResult:
    backend_id: str
    verdict: Verdict
    detail: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    # Trace-compatible outcome row (for TraceRecord.verifier_outcomes).
    outcome_kind: str = "independent_exact_check"

    @property
    def accepted(self) -> bool:
        return self.verdict is Verdict.ACCEPTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "verdict": self.verdict.value,
            "detail": self.detail,
            "evidence": dict(self.evidence),
            "outcome_kind": self.outcome_kind,
            "accepted": self.accepted,
        }

    def as_verifier_outcome(
        self,
        *,
        generator_id: str = "student",
        checker_id: str = "host_verifier",
    ) -> dict[str, Any]:
        """Map to Odyssey TraceVerifier outcome vocabulary."""
        if self.verdict is Verdict.ACCEPTED:
            if self.outcome_kind == "lean_replay":
                return {
                    "kind": "lean_replay",
                    "container_hash": self.evidence.get("container_hash"),
                    "backend_id": self.backend_id,
                    "detail": self.detail,
                }
            return {
                "kind": "independent_exact_check",
                "checker_id": checker_id,
                "generator_id": generator_id,
                "backend_id": self.backend_id,
                "detail": self.detail,
                "evidence": dict(self.evidence),
            }
        if self.verdict is Verdict.REJECTED:
            return {
                "kind": "negative",
                "backend_id": self.backend_id,
                "detail": self.detail,
                "evidence": dict(self.evidence),
            }
        return {
            "kind": "unavailable" if self.verdict is Verdict.UNAVAILABLE else "uncertain",
            "backend_id": self.backend_id,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


@runtime_checkable
class VerifierBackend(Protocol):
    backend_id: str

    def availability(self) -> BackendAvailability:
        """Honest probe: REAL / GATED / ABSENT on this host."""
        ...

    def supports(self, request: VerificationRequest) -> bool:
        ...

    def verify(self, request: VerificationRequest) -> VerificationResult:
        """Run a real check or return UNAVAILABLE. Never invent ACCEPTED."""
        ...


def content_hash(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def normalize_answer(text: str) -> str:
    """Normalize a claimed math answer for comparison without changing meaning."""
    s = (text or "").strip()
    # strip common wrappers
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # collapse whitespace
    s = " ".join(s.split())
    # strip trailing period often left by LLMs
    if s.endswith(".") and s.count(".") == 1:
        s = s[:-1].strip()
    return s
