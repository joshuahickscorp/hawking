"""Round 18 did everything right and the engine threw the science away.

It selected a specimen, guarded, ran odyssey.dense_anatomy, and then WROTE A REAL RECEIPT --
the thing every previous round failed to do:

    {"kind":"measurement","specimen":"Qwen--Qwen3-Embedding-0.6B@97b0c614be4d",
     "axis":"nr_candidate","value":41.09,"source":"odyssey.dense_anatomy",
     "organ":"k_proj","deficit_pct":41.09,"stderr_pct":0.0225,"rank":0, ...}

Specimen, axis, value, source tool, organ, deficit, stderr. Legitimate evidence.

Then: `rolled_back: True`, `status: failed`. Writing a file made the engine classify the
round as a CODE MUTATION, so it demanded red-before-green with an admissible pytest. HCLI
offered `odyssey.record_measurement` as the test, that form is NOT_ADMITTED, validation
failed, and the receipt was rolled back off disk.

The harness told it to write a receipt, then treated the receipt as a code change, then
demanded a proving test for a JSON file, then deleted the measurement.

A receipt is EVIDENCE, not code. Red-before-green is the right contract for a source edit
and a meaningless one for "I wrote down what I measured". What validates a receipt is
whether it is well-formed and carries what a receipt must carry.

The gate is NOT weakened: the evidence path applies only when EVERY operation targets
receipts/ AND every file parses as JSON. One source file anywhere in the operation set and
the whole unit stays on the strict mutation contract, because that is exactly how a bypass
would be built.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcli.engine import Engine

REPO = Path(__file__).resolve().parents[2]


def test_a_pure_receipt_write_is_classified_as_evidence():
    assert Engine._is_evidence_only([Path("receipts/future/X.json")]) is True
    assert Engine._is_evidence_only([Path(REPO / "receipts" / "future" / "Y.json")]) is True


def test_one_source_file_anywhere_keeps_the_strict_contract():
    """The bypass this would otherwise open, closed by construction."""
    assert Engine._is_evidence_only([
        Path("receipts/future/X.json"), Path("hcli/engine.py")]) is False
    assert Engine._is_evidence_only([Path("tools/future/thing.py")]) is False
    assert Engine._is_evidence_only([Path("receipts/future/sneaky.py")]) is False
    assert Engine._is_evidence_only([]) is False


def test_a_receipt_outside_receipts_is_not_evidence():
    """Only the evidence partition. A json anywhere else is still an unknown write."""
    assert Engine._is_evidence_only([Path("workspace/thing.json")]) is False
    assert Engine._is_evidence_only([Path("hcli/config.json")]) is False


def test_evidence_validation_checks_the_RECEIPT_not_a_pytest(tmp_path):
    good = REPO / "workspace" / "_t_receipt_ok.json"
    bad = REPO / "workspace" / "_t_receipt_bad.json"
    good.parent.mkdir(exist_ok=True)
    good.write_text(json.dumps({"kind": "measurement", "specimen": "x", "value": 1.0}))
    bad.write_text("{not json")
    try:
        ok = Engine._validate_evidence([good])
        assert ok["ok"] is True, ok
        assert any(c["kind"] == "receipt_wellformed" for c in ok["checks"]), ok
        # and it must NOT demand a test
        assert not any(c.get("reason") == "NOT_ADMITTED" for c in ok["checks"]), ok

        broken = Engine._validate_evidence([bad])
        assert broken["ok"] is False, "malformed JSON was accepted as evidence"
        assert any("json" in str(c).lower() for c in broken["checks"]), broken
    finally:
        good.unlink(missing_ok=True)
        bad.unlink(missing_ok=True)


def test_an_empty_receipt_is_not_evidence(tmp_path):
    """A round that writes {} has recorded nothing."""
    p = REPO / "workspace" / "_t_receipt_empty.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text("{}")
    try:
        res = Engine._validate_evidence([p])
        assert res["ok"] is False, "an empty object was accepted as a measurement"
    finally:
        p.unlink(missing_ok=True)


def test__validate_ACTUALLY_ROUTES_to_the_evidence_path():
    """A classifier the validator does not consult changes nothing.

    Every test above exercises the two helpers directly, so unwiring the routing in
    _validate left them all green. Third time this session a fix has been correct and
    unreachable; the call site is what the round actually goes through.
    """
    src = (REPO / "hcli" / "engine.py").read_text()
    body = src.split("    def _validate(")[1].split("\n    def ")[0]
    assert "self._is_evidence_only(path_list)" in body, (
        "_validate never asks whether the unit is evidence, so a receipt still takes the "
        "code-mutation contract")
    assert "self._validate_evidence(path_list)" in body, body[:400]
    # and the strict path must still be reachable for everything else
    assert "test_list = list(tests)" in body


def test_a_real_receipt_write_validates_end_to_end():
    """The round-18 payload, verbatim in shape, through the real _validate."""
    e = Engine.__new__(Engine)
    e.root = REPO
    p = REPO / "receipts" / "future" / "_t_round18_shape.json"
    p.write_text(json.dumps({
        "kind": "measurement", "specimen": "Qwen--Qwen3-Embedding-0.6B@97b0c614be4d",
        "axis": "nr_candidate", "value": 41.09, "source": "odyssey.dense_anatomy",
        "organ": "k_proj", "deficit_pct": 41.09, "stderr_pct": 0.0225, "rank": 0}))
    try:
        res = e._validate([p], tests=["odyssey.record_measurement"])
        assert res["ok"] is True, res
        assert res.get("kind") == "evidence", res
        # the NOT_ADMITTED that rolled the real round back must not appear
        assert "NOT_ADMITTED" not in json.dumps(res), res
    finally:
        p.unlink(missing_ok=True)
