"""Scar reevaluator: fidelity bars vs structure, and a tolerance change cannot reopen structure."""
from __future__ import annotations

import json

import pytest

from tools.future import negative_index as ni
from tools.future import scar_reevaluator as sr
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def _scar(**overrides):
    base = {
        "scar_id": "fixture#scar",
        "original_id": "scar",
        "source_path": "fixture.json",
        "parse_status": ni.PARSED,
        "hypothesis_family": "fixture_family",
        "organ": "mlp",
        "model": "fixture-parent",
        "representation": "unrecorded",
        "verdict": "NEGATIVE",
        "failure_mechanism": "fixture",
        "claim_refuted": "fixture",
        "reopen_condition": "unrecorded",
        "refuse_eligible": True,
        "level": "MODEL_SPECIFIC",
        "status": "NEGATIVE",
    }
    base.update(overrides)
    return base


def test_missing_probe_receipt_refuses(monkeypatch):
    monkeypatch.setattr(sr, "PROBE_REL", "receipts/future/NO_SUCH_PROBE.json")
    with pytest.raises(sr.ReevaluatorRefused, match="NO_SUCH_PROBE"):
        sr.probe_tolerance()


def test_empty_corpus_refuses_rather_than_defaulting():
    with pytest.raises(sr.ReevaluatorRefused, match="zero scars"):
        sr.classify_corpus([])


def test_classify_refuses_a_missing_scar():
    with pytest.raises(sr.ReevaluatorRefused, match="missing"):
        sr.classify(None)


def test_structurally_refuted_scar_cannot_be_reopened_by_a_tolerance_change():
    """LOAD-BEARING: a structure that does not exist is not a fidelity-bar miss.

    Cross-expert sharing died because pairwise expert cosine is ~0.004
    (mutually orthogonal). Dropping a reconstruction cosine bar from 0.99
    to 0.0 does not create a shared template, so the class stays
    STRUCTURALLY_REFUTED and tolerance_change_reopens stays false.
    """
    record = _scar(
        scar_id="fixture#cross_expert_structure",
        hypothesis_family="cross_expert_structure",
        organ="gate",
        failure_mechanism="trivial global expert sharing / shared expert template",
        claim_refuted=(
            "experts do not share a global template (near-orthogonal). "
            "pairwise_cosine_mean=0.00414"
        ),
        reopen_condition="row-normalized mean off-diagonal cosine >= 0.10",
        verdict="NEGATIVE",
    )
    a = sr.classify(record)
    assert a["class"] == sr.STRUCTURALLY_REFUTED
    assert a["tolerance_change_reopens"] is False
    for bar in (0.99, 0.90, 0.50, 0.10, 0.0):
        b = sr.classify(record, fidelity_cosine_bar=bar)
        assert b["class"] == sr.STRUCTURALLY_REFUTED, bar
        assert b["tolerance_change_reopens"] is False, bar


def test_organ_gate_fidelity_scar_is_possibly_reopenable():
    record = _scar(
        scar_id="fixture#binary_sign_scale128",
        hypothesis_family="binary_quantization",
        organ="gate",
        failure_mechanism="binary_sign_scale128",
        claim_refuted="component_sensitive_organ_gate_failed",
        measured_outcome={
            "functional": {
                "claim_boundary": (
                    "teacher tensor matvec only; not full-model parity, "
                    "mini-generation, HCLI, or capability"
                ),
                "cosine": 0.618982195854187,
                "relative_l2": 0.8372106552124023,
            }
        },
    )
    rows = sr.classify_corpus([record])
    assert len(rows) == 1
    a = rows[0]
    assert a["class"] == sr.POSSIBLY_REOPENABLE
    assert a["tolerance_change_reopens"] is True
    assert a["died_at"]["kind"] == "organ_output_cosine"
    assert a["died_at"]["measured"] == pytest.approx(0.618982195854187)
    assert "0.618" in a["died_at"]["summary"]
    assert "relative_l2" in a["died_at"]["summary"]


def test_pairwise_expert_cosine_is_structural_even_though_cosine_appears():
    record = _scar(
        hypothesis_family="cross_expert_structure",
        claim_refuted="mean pairwise cosine between experts = 1e-4; experts are mutually orthogonal",
        failure_mechanism="delta coding / shared low-rank bases / cluster-mean subtraction across experts",
        reopen_condition="a future parent measures mean pairwise expert cosine >= 0.10 on its own weights",
    )
    a = sr.classify(record)
    assert a["class"] == sr.STRUCTURALLY_REFUTED
    assert a["tolerance_change_reopens"] is False


def test_generation_incoherent_is_structurally_refuted():
    record = _scar(
        hypothesis_family="binary_quantization",
        failure_mechanism="the 1.25-bpw binary body is physically fast but generation-injured",
        claim_refuted="generation incoherent",
        reopen_condition="a healing scheme that restores coherent generation",
    )
    a = sr.classify(record)
    assert a["class"] == sr.STRUCTURALLY_REFUTED
    assert a["tolerance_change_reopens"] is False


def test_campaign_process_scar_is_structurally_refuted():
    record = _scar(
        hypothesis_family="prefill_over_generated_token_denominator",
        level="GENERAL_PHYSICAL",
        verdict="FALSIFIED",
        failure_mechanism="NUMERATOR_AND_DENOMINATOR_COUNT_DIFFERENT_EVENTS",
        claim_refuted="that dividing prefill+decode totals by generated tokens yields a per-token production cost",
        reopen_condition="a field whose numerator is decode-only",
    )
    a = sr.classify(record)
    assert a["class"] == sr.STRUCTURALLY_REFUTED
    assert a["tolerance_change_reopens"] is False


def test_held_out_relative_l2_kill_is_possibly_reopenable():
    record = _scar(
        scar_id="fixture#DISTILLED",
        hypothesis_family="distilled",
        organ="mlp",
        failure_mechanism=(
            "Distilled operator held-out relative L2 0.442177 vs mean 0.858742. "
            "Kill is 0.25."
        ),
        claim_refuted="F(x)=down(silu(gate(x))*up(x)) on the teacher corpus",
        verdict="MEASURED_NEGATIVE",
    )
    a = sr.classify(
        record,
        evidence={"held_out_relative_l2": 0.442177, "held_out_kill_rel": 0.25},
    )
    assert a["class"] == sr.POSSIBLY_REOPENABLE
    assert a["tolerance_change_reopens"] is True
    assert a["died_at"]["bar"] == pytest.approx(0.25)
    assert a["died_at"]["measured"] == pytest.approx(0.442177)


def test_unparsed_scar_is_method_unrecorded():
    record = _scar(
        parse_status=ni.UNPARSED,
        verdict="unrecorded",
        failure_mechanism="unrecorded",
        claim_refuted="unrecorded",
        hypothesis_family="unrecorded",
        reopen_condition="unrecorded",
    )
    a = sr.classify(record)
    assert a["class"] == sr.METHOD_UNRECORDED
    assert a["tolerance_change_reopens"] is False


def test_live_reopen_is_not_a_refutation():
    record = _scar(
        verdict="LIVE_REOPEN_HOLDS",
        failure_mechanism="shared_basis_across_experts",
        claim_refuted="That routed experts share a basis",
        reopen_condition="Never on Q80",
        refuse_eligible=False,
    )
    a = sr.classify(record)
    assert a["class"] == sr.NOT_A_REFUTATION


def test_expert_merge_reconstruction_is_structural_not_a_codec_bar():
    record = _scar(
        hypothesis_family="expert_merge",
        failure_mechanism="reconstruct an omitted MoE expert from a learned combination of surviving experts",
        claim_refuted=(
            "median held-out relative error of the BEST single surviving expert "
            "is 0.885/0.993/0.995; ~1.0 means the reconstruction is no better "
            "than predicting zero"
        ),
        reopen_condition="a parent measures best-single-survivor reconstruction error <= 0.5",
    )
    a = sr.classify(record)
    assert a["class"] == sr.STRUCTURALLY_REFUTED
    assert a["tolerance_change_reopens"] is False


def test_rank_orders_by_ebpw_then_token_then_cost():
    rows = [
        sr.classify(
            _scar(
                scar_id="a#uniform_q4",
                hypothesis_family="uniform_q4",
                organ="embed",
                claim_refuted="component_sensitive_organ_gate_failed",
            )
        ),
        sr.classify(
            _scar(
                scar_id="b#binary",
                hypothesis_family="binary_quantization",
                organ="mlp",
                claim_refuted="component_sensitive_organ_gate_failed",
            )
        ),
        sr.classify(
            _scar(
                scar_id="c#ternary",
                hypothesis_family="ternary",
                organ="mlp",
                claim_refuted="component_sensitive_organ_gate_failed",
            )
        ),
    ]
    ranked = sr.rank_reopenable(rows)
    assert [r["scar_id"] for r in ranked] == ["b#binary", "c#ternary", "a#uniform_q4"]
    assert ranked[0]["theoretical_ebpw_reduction_rank"] >= ranked[1]["theoretical_ebpw_reduction_rank"]
    assert ranked[0]["token_ns_opportunity"] == "HIGH"
    assert ranked[0]["implementation_cost"] in {"LOW", "MEDIUM", "HIGH"}
    assert all(r["ranking_is_not_a_relaunch"] is True for r in ranked)


def test_real_corpus_counts_both_classes():
    scars = ni.ingest()
    assert len(scars) >= 685
    rows = sr.classify_corpus(scars)
    cov = sr.counts(rows)
    assert cov["n_scars"] == len(scars)
    assert cov["n_structurally_refuted"] >= 1
    assert cov["n_possibly_reopenable"] >= 1
    assert cov["n_scars"] == (
        cov["n_structurally_refuted"]
        + cov["n_possibly_reopenable"]
        + cov["n_method_unrecorded"]
        + cov["n_not_a_refutation"]
    )


def test_real_cross_expert_scar_stays_structural_across_tolerance_bars():
    scars = ni.ingest()
    hits = [
        s
        for s in scars
        if s.parse_status == ni.PARSED
        and s.hypothesis_family == "cross_expert_structure"
        and s.refuse_eligible
    ]
    assert hits, "cross_expert_structure must be in the real corpus"
    scar = hits[0]
    a = sr.classify(scar)
    assert a["class"] == sr.STRUCTURALLY_REFUTED
    for bar in (0.99, 0.0):
        b = sr.classify(scar, fidelity_cosine_bar=bar)
        assert b["class"] == sr.STRUCTURALLY_REFUTED
        assert b["tolerance_change_reopens"] is False


def test_build_writes_sealed_receipt_and_nothing_was_relaunched():
    out = sr.build()
    assert out.parent == RECEIPTS
    assert out.name == sr.RECEIPT
    doc = json.loads(out.read_text())
    assert doc["schema"] == sr.SCHEMA
    assert doc["seal_sha256"]
    assert doc["nothing_relaunched"] is True
    assert "did not relaunch" in doc["nothing_relaunched_statement"]
    assert doc["resident_decides"] is True
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["counts"]["n_scars"] >= 685
    assert doc["counts"]["n_possibly_reopenable"] >= 1
    assert doc["counts"]["n_structurally_refuted"] >= 1
    assert doc["probe_tolerance"]["worst_damage"] is not None
    assert doc["probe_tolerance"]["at_fraction_zeroed"] == 0.4
    assert doc["probe_tolerance"]["source"] == sr.PROBE_REL
    top = doc["top_reopenable_families"]
    assert top, "expected named reopenable families"
    for fam in top:
        assert fam["died_at_threshold"]
        assert fam["ranking_is_not_a_relaunch"] is True
        assert "hypothesis_family" in fam
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        assert key not in doc or doc[key] in (None, "UNKNOWN")


def test_top_reopenable_names_the_threshold_each_died_at():
    scars = ni.ingest()
    rows = sr.classify_corpus(scars)
    ranked = sr.rank_reopenable(rows)
    assert ranked, "expected at least one POSSIBLY_REOPENABLE scar"
    named = [r for r in ranked if r["died_at_threshold"] and r["died_at_threshold"] != "UNRECORDED"]
    assert named, "top reopenable scars must carry the threshold they died at"
    organ_gate = [
        r
        for r in named
        if (r.get("died_at") or {}).get("kind") == "organ_output_cosine"
        and (r.get("died_at") or {}).get("measured") is not None
    ]
    assert organ_gate, "JSONL organ-gate scars must surface measured cosine/relative_l2"
    assert organ_gate[0]["died_at"]["measured"] < 0.99
