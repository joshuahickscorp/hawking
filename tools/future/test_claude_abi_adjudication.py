"""Contract test for Codex's ABI adjudication module.

`claude_abi_adjudication.py` is CODEX's work, written into the sidecar partition
as its response to the six static ABI findings. The sidecar does not own it and
must not change its verdicts. But `abi_verdicts.ingest_codex_verdicts()` now
depends on the shape of what it emits, so that dependency gets a contract test:
if Codex changes the receipt's shape, the sidecar should fail loudly here rather
than silently ingest nothing.
"""
import json

import pytest

from tools.future import abi_verdicts as av
from tools.future import claude_abi_adjudication as caa
from tools.future._common import REPO


def _doc():
    return json.loads((REPO / "receipts/future/CLAUDE_SIDECAR_ABI_ADJUDICATION.json").read_text())


def test_module_builds_a_sealed_receipt():
    out = caa.build() if hasattr(caa, "build") else None
    doc = _doc()
    assert doc["schema"] == caa.SCHEMA
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    if out is not None:
        assert out.name == caa.RECEIPT


def test_every_outcome_carries_the_fields_the_sidecar_ingests():
    """These are exactly the keys ingest_codex_verdicts() reads."""
    outcomes = _doc()["outcomes"]
    assert outcomes, "no adjudicated outcomes"
    for o in outcomes:
        assert o["finding_id"], o
        assert o["classification"], o
        assert o["evidence"], "a verdict with no evidence would be REFUSED by record_verdict"
        assert isinstance(o["evidence"], list)
        assert o["status"], o
        pre = o["preflight"]
        assert pre["check"] and pre["severity"] == "ERROR"
        # kernel_existence uses host_sites (list); type_width uses host_site (scalar).
        assert pre.get("host_sites") or pre.get("host_site"), pre


def test_every_codex_classification_maps_to_a_harness_verdict_class():
    """An unmapped classification would be silently dropped by the ingest."""
    seen = {o["classification"] for o in _doc()["outcomes"]}
    unmapped = seen - set(av.CODEX_CLASS_MAP)
    assert not unmapped, f"unmapped Codex classifications: {sorted(unmapped)}"
    for codex_class in seen:
        assert av.CODEX_CLASS_MAP[codex_class] in av.VERDICT_CLASS_NAMES


def test_summary_totals_agree_with_the_outcome_rows():
    doc = _doc()
    s, outcomes = doc["summary"], doc["outcomes"]
    assert s["reported_findings"] == len(outcomes)
    real = sum(1 for o in outcomes if o["classification"] == "REAL_DEFECT")
    assert s["real_defects_fixed"] == real
    assert s["outstanding_confirmed_defects"] == sum(
        1 for o in outcomes if o["classification"] == "REAL_DEFECT" and o["status"] != "FIXED"
    )


def test_ingest_accounts_for_every_outcome_through_the_evidence_checked_path():
    """Every Codex outcome must end up accounted for, not necessarily fresh.

    `store()` is shared, so by the time this runs another test may already have
    ingested the same verdicts and `record_verdict` correctly refuses duplicates.
    Asserting "recorded == 6" would then be asserting test ORDER. The invariant
    that actually matters is that no outcome is silently dropped: each one is
    either recorded now or already present, and none is unmapped.
    """
    st = av.store()
    res = av.ingest_codex_verdicts(st)
    assert res["available"] is True
    assert res["unmapped"] == []

    outcome_ids = {o["finding_id"] for o in _doc()["outcomes"]}
    recorded_ids = {r["finding_id"] for r in res["recorded"]}
    refused_ids = {r["finding_id"] for r in res["refused"]}
    assert outcome_ids == recorded_ids | refused_ids, "an outcome was silently dropped"

    # Anything refused here must be refused ONLY as a duplicate; any other
    # refusal reason is a real defect in the ingest path.
    for r in res["refused"]:
        assert "already has a verdict" in r["error"], r
        assert r["finding_id"] in st.verdicts, r


def test_negative_control_an_outcome_without_evidence_is_refused():
    """record_verdict must refuse a verdict with no evidence, whoever submits it.

    Codex is the adjudicating authority, but authority does not exempt it from
    the evidence requirement -- that is the whole point of routing its rulings
    through the same path.
    """
    st = av.store()
    fid = _doc()["outcomes"][0]["finding_id"]
    with pytest.raises(av.VerdictRefused):
        av.record_verdict(fid, "CONFIRMED_BUG", None, recorded_by="codex:test", st=st)
    with pytest.raises(av.VerdictRefused):
        av.record_verdict(fid, "CONFIRMED_BUG", [], recorded_by="codex:test", st=st)


def test_sidecar_never_appears_to_have_ruled():
    """Recording Codex's verdict must not be counted as a sidecar ruling."""
    doc = json.loads((REPO / "receipts/future/ABI_VERDICT_HARNESS.json").read_text())
    assert doc["rulings_issued_by_this_sidecar"] == 0
    assert doc["rulings_recorded_from_external_authority"] == len(_doc()["outcomes"])
    assert doc["codex_files_untouched"] is True
