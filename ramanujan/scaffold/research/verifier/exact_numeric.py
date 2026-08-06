"""REAL exact-numeric verifier: pure Python AST + fractions.

No sympy required.  Safely evaluates a closed arithmetic expression from the
problem payload and compares the claimed answer to the recomputed value.

This is the host backend that the expert-iteration fixture exercises end-to-end
when Lean is absent and GLM is gated.
"""
from __future__ import annotations

import ast
import operator
from fractions import Fraction
from typing import Any

from ramanujan.verifier.base import (
    BackendAvailability,
    BackendStatus,
    VerificationRequest,
    VerificationResult,
    Verdict,
    normalize_answer,
)

BACKEND_ID = "exact_numeric"

# Closed, side-effect-free operators only.
_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant {node.value!r}")
        return Fraction(node.value).limit_denominator()
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op = _BINOPS[type(node.op)]
        if type(node.op) is ast.Pow:
            # Restrict exponents to small integers to avoid huge work.
            if right.denominator != 1 or abs(int(right)) > 64:
                raise ValueError("power exponents must be integers with |n| <= 64")
            return Fraction(left) ** int(right)
        return Fraction(op(left, right))
    # Reject names, calls, attributes, etc.
    raise ValueError(f"disallowed expression node {type(node).__name__}")


def evaluate_exact(expression: str) -> Fraction:
    """Evaluate a pure arithmetic expression to an exact Fraction."""
    expr = (expression or "").strip()
    if not expr:
        raise ValueError("empty expression")
    if len(expr) > 500:
        raise ValueError("expression too long")
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree)


def parse_claimed_number(text: str) -> Fraction:
    """Parse a claimed answer as an exact rational when possible."""
    s = normalize_answer(text)
    if not s:
        raise ValueError("empty answer")
    # allow "a/b"
    if "/" in s and s.count("/") == 1 and not any(c in s for c in "+-*()"):
        num, den = s.split("/")
        return Fraction(int(num.strip()), int(den.strip()))
    # integer or decimal
    if any(c.isalpha() for c in s):
        raise ValueError(f"non-numeric answer {s!r}")
    return Fraction(s)


class ExactNumericBackend:
    """Recompute payload['expression'] and compare to the claimed answer."""

    backend_id = BACKEND_ID

    def availability(self) -> BackendAvailability:
        # Always real: stdlib only.
        return BackendAvailability(
            backend_id=self.backend_id,
            status=BackendStatus.REAL,
            detail="pure-Python AST + fractions.Fraction; always available on host",
            version="stdlib",
            capabilities=("exact_arithmetic", "rational_compare"),
        )

    def supports(self, request: VerificationRequest) -> bool:
        if request.kind not in {"exact_numeric", "numeric", "arithmetic"}:
            return False
        return bool(request.payload.get("expression"))

    def verify(self, request: VerificationRequest) -> VerificationResult:
        if not self.supports(request):
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.UNAVAILABLE,
                detail="request kind/payload not supported by exact_numeric",
                evidence={"kind": request.kind},
            )
        expression = str(request.payload["expression"])
        try:
            expected = evaluate_exact(expression)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.UNCERTAIN,
                detail=f"could not recompute expression: {exc}",
                evidence={"expression": expression},
            )
        try:
            claimed = parse_claimed_number(request.claimed_answer)
        except (ValueError, ZeroDivisionError) as exc:
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.REJECTED,
                detail=f"claimed answer is not an exact number: {exc}",
                evidence={
                    "expression": expression,
                    "recomputed": _frac_str(expected),
                    "claimed_raw": request.claimed_answer,
                },
            )
        if claimed == expected:
            return VerificationResult(
                backend_id=self.backend_id,
                verdict=Verdict.ACCEPTED,
                detail=f"claimed answer matches exact recomputation of {expression!r}",
                evidence={
                    "expression": expression,
                    "recomputed": _frac_str(expected),
                    "claimed": _frac_str(claimed),
                    "arithmetic": "exact",
                    "checker_id": request.checker_id,
                    "generator_id": "student",
                },
                outcome_kind="independent_exact_check",
            )
        return VerificationResult(
            backend_id=self.backend_id,
            verdict=Verdict.REJECTED,
            detail=(
                f"claimed {_frac_str(claimed)} != recomputed {_frac_str(expected)} "
                f"for expression {expression!r}"
            ),
            evidence={
                "expression": expression,
                "recomputed": _frac_str(expected),
                "claimed": _frac_str(claimed),
            },
        )


def _frac_str(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
