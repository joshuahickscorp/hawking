"""Focused regression for RUNTIME_RESIDENCY_LIFECYCLE_V5 role evidence.

Exercises the production helper ``_coerce_evidence`` in
``hawking.canonical_runtime``: a role binding is only evidence-backed when it
carries at least one non-empty evidence token, and a bare string must be read
as a single token rather than exploded into characters.

Boundary: this is a unit-level check of resolver behaviour only.  It makes no
claim about release readiness, multi-worker co-residency, or hardware
qualification.
"""
from __future__ import annotations

from hawking.canonical_runtime import _coerce_evidence


def test_missing_evidence_is_not_evidence_backed() -> None:
    assert _coerce_evidence(None) == ()
    assert _coerce_evidence("") == ()
    assert _coerce_evidence("   ") == ()
    assert _coerce_evidence([]) == ()
    assert _coerce_evidence(["", "  "]) == ()


def test_bare_string_is_one_token_not_characters() -> None:
    assert _coerce_evidence("RESIDENT_QUALIFIED") == ("RESIDENT_QUALIFIED",)


def test_sequence_is_trimmed_and_deduplicated_in_order() -> None:
    assert _coerce_evidence(
        [" RESIDENT_QUALIFIED ", "CAPABILITY_QUALIFIED", "RESIDENT_QUALIFIED", ""]
    ) == ("RESIDENT_QUALIFIED", "CAPABILITY_QUALIFIED")


def test_non_iterable_scalar_is_withheld() -> None:
    assert _coerce_evidence(7) == ()
    assert _coerce_evidence({"evidence": "RESIDENT_QUALIFIED"}) == ()