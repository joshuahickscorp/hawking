"""Canonical verification utilities.

This package owns checks about what a probe establishes. Historical receipt
schemas remain readable, but new live callers import the verifier from this
functional namespace rather than from a historical sidecar directory.
"""
from __future__ import annotations

from .status_causality import (
    CLAIM_CHECK_VERDICTS,
    FIVE_RECORDED_FIELDS,
    OVERREACHING,
    SUPPORTED,
    UNDERDETERMINED,
    UNTESTED,
    challenge,
    check_claim,
    emit,
    stamp,
)

__all__ = [
    "CLAIM_CHECK_VERDICTS",
    "FIVE_RECORDED_FIELDS",
    "OVERREACHING",
    "SUPPORTED",
    "UNDERDETERMINED",
    "UNTESTED",
    "challenge",
    "check_claim",
    "emit",
    "stamp",
]
