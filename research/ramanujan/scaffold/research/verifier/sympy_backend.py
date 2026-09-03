"""Sympy symbolic verifier — REAL when sympy imports, else ABSENT/GATED.

Uses the current interpreter first; probes known project interpreters
(same pattern as toolchain_selftest) without inventing results.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

from ramanujan.verifier.base import (
    BackendAvailability,
    BackendStatus,
    VerificationRequest,
    VerificationResult,
    Verdict,
    normalize_answer,
)

BACKEND_ID = "sympy"

_EXTRA_PYTHONS = (
    Path.home() / ".grok-vision" / "bin" / "python",
    Path(__file__).resolve().parents[4] / ".venv" / "glm52" / "bin" / "python",
)


def _try_import_sympy() -> tuple[Any | None, str | None, str]:
    try:
        sp = importlib.import_module("sympy")
        return sp, getattr(sp, "__version__", "unknown"), "current_interpreter"
    except ImportError:
        pass
    for py in _EXTRA_PYTHONS:
        if not py.is_file():
            continue
        proc = subprocess.run(
            [str(py), "-c", "import sympy; print(sympy.__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0:
            ver = (proc.stdout or "").strip().splitlines()[0] if proc.stdout else "unknown"
            return None, ver, f"available_via:{py}"
    return None, None, "not_found"


class SympyBackend:
    """Symbolic simplify / equality check for algebraic answers."""

    backend_id = BACKEND_ID

    def __init__(self) -> None:
        self._sp, self._version, self._where = _try_import_sympy()

    def availability(self) -> BackendAvailability:
        if self._sp is not None:
            return BackendAvailability(
                backend_id=self.backend_id,
                status=BackendStatus.REAL,
                detail=f"sympy importable in {self._where}",
                version=self._version,
                capabilities=("simplify", "symbolic_equal"),
            )
        if self._version is not None:
            return BackendAvailability(
                backend_id=self.backend_id,
                status=BackendStatus.GATED,
                detail=(
                    f"sympy {self._version} found under {self._where} but not in "
                    f"current interpreter ({sys.executable}); re-run with that python "
                    "for REAL sympy checks"
                ),
                version=self._version,
                capabilities=("simplify",),
            )
        return BackendAvailability(
            backend_id=self.backend_id,
            status=BackendStatus.ABSENT,
            detail="sympy not importable on this host",
            version=None,
        )

    def supports(self, request: VerificationRequest) -> bool:
        return request.kind in {"sympy", "symbolic", "algebra"} and bool(
            request.payload.get("expression") or request.payload.get("expected_expr")
        )

    def verify(self, request: VerificationRequest) -> VerificationResult:
        avail = self.availability()
        if avail.status is not BackendStatus.REAL or self._sp is None:
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.UNAVAILABLE,
                detail=avail.detail,
                evidence={"status": avail.status.value},
            )
        sp = self._sp
        claimed = normalize_answer(request.claimed_answer)
        expected_src = str(
            request.payload.get("expected_expr")
            or request.payload.get("expression")
            or ""
        )
        if not expected_src or not claimed:
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.UNCERTAIN,
                detail="missing expression or claimed answer",
            )
        try:
            expected = sp.simplify(sp.sympify(expected_src))
            claimed_expr = sp.simplify(sp.sympify(claimed))
            equal = bool(
                expected.equals(claimed_expr)
                or sp.simplify(sp.expand(expected - claimed_expr)) == 0
            )
        except Exception as exc:  # sympy can raise many parse errors
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.UNCERTAIN,
                detail=f"sympy could not decide: {exc}",
                evidence={"expression": expected_src, "claimed": claimed},
            )
        if equal:
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.ACCEPTED,
                detail="sympy reports exact symbolic equality after simplify",
                evidence={
                    "expression": expected_src,
                    "recomputed": str(expected),
                    "claimed": str(claimed_expr),
                    "arithmetic": "exact",
                    "checker_id": request.checker_id,
                    "generator_id": "student",
                },
            )
        return VerificationResult(
            backend_id=self.backend_id,
            verdict=Verdict.REJECTED,
            detail=f"sympy: claimed {claimed_expr} != expected {expected}",
            evidence={
                "expression": expected_src,
                "recomputed": str(expected),
                "claimed": str(claimed_expr),
            },
        )


def sympy_available() -> bool:
    return SympyBackend().availability().status is BackendStatus.REAL
