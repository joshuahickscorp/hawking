"""Pure tests for the strict historical-HCLI route recovery boundary."""
from __future__ import annotations

from lab.operators.ascension_qwen30_quality_repack_hcli_route_recovery import (
    STATUS,
    _sha256,
)


def test_route_recovery_status_remains_a_refusal_not_a_membership_claim() -> None:
    assert "BLOCKED" in STATUS
    assert "ROUTE_MEMBERSHIP" in STATUS


def test_prompt_hash_matches_the_hcli_negative_evidence_convention() -> None:
    assert _sha256("Reply with exactly the single word HAWKING.") == (
        "b7cd626f844d937768c916bc3a7bdea625e7d84fcbbc74be89ba3e1551427e30"
    )
