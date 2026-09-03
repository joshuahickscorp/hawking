"""Backend registry: discover REAL vs GATED vs ABSENT verifiers on this host."""
from __future__ import annotations

from typing import Any, Sequence

from ramanujan.verifier.base import (
    BackendAvailability,
    BackendStatus,
    VerificationRequest,
    VerificationResult,
    Verdict,
    VerifierBackend,
)
from ramanujan.verifier.exact_numeric import ExactNumericBackend
from ramanujan.verifier.lean_backend import LeanBackend
from ramanujan.verifier.sympy_backend import SympyBackend


class VerifierRegistry:
    """Routes a request to the first supporting backend; never fakes ACCEPTED."""

    def __init__(self, backends: Sequence[VerifierBackend] | None = None) -> None:
        self.backends: list[VerifierBackend] = list(
            backends
            if backends is not None
            else (ExactNumericBackend(), SympyBackend(), LeanBackend())
        )

    def probe(self) -> list[BackendAvailability]:
        return [b.availability() for b in self.backends]

    def probe_report(self) -> dict[str, Any]:
        rows = [a.as_dict() for a in self.probe()]
        real = [r for r in rows if r["status"] == BackendStatus.REAL.value]
        gated = [r for r in rows if r["status"] == BackendStatus.GATED.value]
        absent = [r for r in rows if r["status"] == BackendStatus.ABSENT.value]
        return {
            "schema": "hawking.ramanujan.verifier_backend_probe.v1",
            "backends": rows,
            "real": [r["backend_id"] for r in real],
            "gated": [r["backend_id"] for r in gated],
            "absent": [r["backend_id"] for r in absent],
            "has_real_backend": bool(real),
        }

    def select(self, request: VerificationRequest) -> VerifierBackend | None:
        for backend in self.backends:
            if backend.supports(request):
                return backend
        return None

    def verify(self, request: VerificationRequest) -> VerificationResult:
        backend = self.select(request)
        if backend is None:
            return VerificationResult(
                backend_id="registry",
                verdict=Verdict.UNAVAILABLE,
                detail=f"no registered backend supports kind={request.kind!r}",
                evidence={"kind": request.kind},
            )
        result = backend.verify(request)
        # Hard invariant: UNAVAILABLE / UNCERTAIN never promoted to ACCEPTED.
        if result.verdict is not Verdict.ACCEPTED and result.accepted:
            return VerificationResult(
                backend_id=result.backend_id,
                verdict=Verdict.REJECTED,
                detail="backend claimed accepted without ACCEPTED verdict; fail closed",
            )
        return result


def default_registry() -> VerifierRegistry:
    return VerifierRegistry()


def probe_backends() -> dict[str, Any]:
    return default_registry().probe_report()
