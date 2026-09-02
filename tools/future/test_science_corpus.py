"""Science corpus: real historical receipts, round-trip, schema evolution."""
from __future__ import annotations

import json

import pytest

from tools.future import science_corpus as sc


def test_load_historical_corpus_is_nonempty_and_names_receipts():
    corpus = sc.load_historical_corpus()
    assert corpus["n_records"] == len(corpus["records"])
    assert corpus["n_records"] > 0
    loaded = corpus["named_receipts_loaded"]
    assert loaded, "no named receipts loaded"
    for rel in (
        "receipts/future/CUDA_LOWBIT_HYPOTHESES.json",
        "receipts/future/CAMPAIGN_SCARS.json",
        "receipts/future/ODYSSEY2_LAW_STORE.json",
        "receipts/future/HCLI_FUTURE_WORKUNITS.json",
        "receipts/future/BA_DELTA_AB.json",
        "receipts/future/ORGAN_BANDWIDTH.json",
    ):
        assert rel in loaded, rel
    by_kind = corpus["by_kind"]
    for kind in sc.KINDS:
        assert by_kind[kind] > 0, f"corpus missing kind {kind}"


def test_roundtrip_preserves_key_fields_for_each_kind():
    corpus = sc.load_historical_corpus()
    seen = set()
    for rec in corpus["records"]:
        kind = rec["kind"]
        if kind in seen:
            continue
        restored = sc.round_trip(rec)
        assert sc.key_fields_preserved(rec, restored)
        assert restored["kind"] == rec["kind"]
        assert restored["record_id"] == rec["record_id"]
        assert restored["source_receipt"] == rec["source_receipt"]
        seen.add(kind)
        if len(seen) == len(sc.KINDS):
            break
    assert seen == set(sc.KINDS)


def test_cuda_hypothesis_roundtrip_preserves_id_and_statement():
    corpus = sc.load_historical_corpus()
    hyps = [
        r
        for r in corpus["records"]
        if r["kind"] == "hypothesis"
        and r["source_receipt"].endswith("CUDA_LOWBIT_HYPOTHESES.json")
    ]
    assert hyps, "CUDA_LOWBIT_HYPOTHESES.json produced no hypotheses"
    rec = hyps[0]
    assert rec["key_fields"]["id"]
    assert rec["key_fields"]["statement"]
    restored = sc.round_trip(rec)
    assert restored["key_fields"]["id"] == rec["key_fields"]["id"]
    assert restored["key_fields"]["statement"] == rec["key_fields"]["statement"]
    assert restored["key_fields"]["hypothesis_family"] == rec["key_fields"][
        "hypothesis_family"
    ]


def test_law_roundtrip_preserves_law_id_statement_scope():
    corpus = sc.load_historical_corpus()
    laws = [r for r in corpus["records"] if r["kind"] == "law"]
    assert laws
    rec = laws[0]
    for field in ("law_id", "statement", "scope"):
        assert rec["key_fields"].get(field), field
    restored = sc.round_trip(rec)
    assert restored["key_fields"]["law_id"] == rec["key_fields"]["law_id"]
    assert restored["key_fields"]["statement"] == rec["key_fields"]["statement"]
    assert restored["key_fields"]["scope"] == rec["key_fields"]["scope"]


def test_scar_roundtrip_preserves_id_and_claim_refuted():
    corpus = sc.load_historical_corpus()
    scars = [
        r
        for r in corpus["records"]
        if r["kind"] == "scar"
        and r["source_receipt"].endswith("CAMPAIGN_SCARS.json")
    ]
    assert scars
    rec = scars[0]
    restored = sc.round_trip(rec)
    assert restored["key_fields"]["id"] == rec["key_fields"]["id"]
    assert restored["key_fields"]["claim_refuted"] == rec["key_fields"]["claim_refuted"]
    assert restored["key_fields"]["verdict"] == rec["key_fields"]["verdict"]


def test_adapter_reads_older_scar_schema_without_migrating():
    """v0 keys (scar_id, what_was_wrong, reopen_if) still project."""
    old = {
        "schema": "hawking.future.campaign_scars.v0",
        "evidence_class": "STATIC_ONLY",
        "scars": [
            {
                "scar_id": "OLD-SCAR-1",
                "what_was_wrong": "divided by the wrong denominator",
                "verdict": "FALSIFIED",
                "reopen_if": "numerator matches denominator events",
                "family": "DENOMINATOR",
            }
        ],
    }
    recs = sc.adapt_document(old, source_receipt="memory:v0-scars")
    assert len(recs) == 1
    keys = recs[0]["key_fields"]
    assert keys["id"] == "OLD-SCAR-1"
    assert keys["claim_refuted"] == "divided by the wrong denominator"
    assert keys["reopen_condition"] == "numerator matches denominator events"
    assert keys["hypothesis_family"] == "DENOMINATOR"
    assert recs[0]["schema_family"] == "hawking.future.campaign_scars"
    restored = sc.round_trip(recs[0])
    assert restored["key_fields"]["id"] == "OLD-SCAR-1"


def test_evidence_tiers_are_not_merged_or_promoted_to_hardware():
    corpus = sc.load_historical_corpus()
    tiers = {r["evidence_tier"] for r in corpus["records"]}
    assert tiers <= set(sc.EVIDENCE_TIERS)
    assert "HARDWARE_MEASURED" not in tiers
    for rec in corpus["records"]:
        if rec["source_receipt"].endswith("ECONOMICS_CALIBRATION.json") and rec["kind"] == "measurement":
            assert rec["evidence_tier"] == "COST_MODEL"


def test_mission_kernel_outcomes_call_deterministic_belief_update():
    corpus = sc.load_historical_corpus()
    outcomes = [
        r
        for r in corpus["records"]
        if r["kind"] == "outcome"
        and r["source_receipt"].endswith("HCLI_MISSION_KERNEL.json")
    ]
    assert outcomes
    belief = [
        r for r in outcomes if r["key_fields"].get("id") == "mission_kernel_belief_update"
    ]
    assert belief
    assert belief[0]["key_fields"]["learned"] is False
    assert belief[0]["key_fields"]["n_hypotheses"] >= 1


def test_empty_corpus_is_refused():
    with pytest.raises(sc.CorpusRefused, match="empty"):
        sc.load_historical_corpus(sources=("receipts/future/NO_SUCH.json",))
