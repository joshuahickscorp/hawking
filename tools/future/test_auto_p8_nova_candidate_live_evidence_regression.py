"""Focused regression for candidate live-evidence status classification.

Exercises the production behavior added in
``tools/future/kimi_operator_live_evidence.py``: a candidate with no bound
live payload must be reported as ``unqualified`` with no evidence digest,
while a candidate carrying a non-empty live observation is reported as
``measured`` with a stable sha256 digest.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.future.kimi_operator_live_evidence import (  # noqa: E402
    _candidate_live_status,
    _candidate_live_summary,
    _evidence_digest,
)


def test_unmeasured_candidate_is_unqualified_without_digest() -> None:
    summary = _candidate_live_summary(None)
    assert summary == {"status": "unqualified", "evidence_digest": None}
    assert _candidate_live_status({}) == "unqualified"
    assert _candidate_live_status({"live": []}) == "unqualified"


def test_measured_candidate_binds_stable_digest() -> None:
    payload = {"live": [{"step": 1, "delta": 0.25}]}
    summary = _candidate_live_summary(payload)
    assert summary["status"] == "measured"
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    assert summary["evidence_digest"] == expected
    assert _evidence_digest(payload) == expected


def test_digest_is_order_independent() -> None:
    a = {"live": [{"step": 1}], "candidate": "OPE"}
    b = {"candidate": "OPE", "live": [{"step": 1}]}
    assert _evidence_digest(a) == _evidence_digest(b)


if __name__ == "__main__":
    test_unmeasured_candidate_is_unqualified_without_digest()
    test_measured_candidate_binds_stable_digest()
    test_digest_is_order_independent()
    print("P8_NOVA_CANDIDATE_LIVE_EVIDENCE regression: OK")