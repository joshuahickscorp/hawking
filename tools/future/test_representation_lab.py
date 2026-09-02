"""Beyond-Dense laboratory: corpus, eval hook, verifier, second family."""
from __future__ import annotations

import json

import pytest

from tools.future import complete_ebpw as ce
from tools.future import representation_lab as lab
from tools.future._common import RECEIPTS, _assert_no_hardware_claims
from tools.odyssey import noetic_compiler as nc


NAMED = (
    "receipts/future/COMPLETE_EBPW.json",
    "receipts/future/REPRESENTATION_FLOOR.json",
    "receipts/future/RIVAL_CODEC_SCREEN.json",
    "receipts/future/MLP_CODE_INFORMATION.json",
    "receipts/future/MLP_AUXILIARY_INFORMATION.json",
    "receipts/future/MLP_SPARSE_RESIDUAL.json",
    "receipts/future/DELTANET_REPRESENTATION.json",
    "receipts/future/FLASH_BPW_LADDER.json",
    "receipts/future/AUX_CAPABILITY_SCREEN.json",
    "receipts/future/REPRESENTATION_DECODE_FUSION.json",
    "receipts/future/MLP_BYTE_CENSUS.json",
    "receipts/future/ECONOMICS_CALIBRATION.json",
)


def test_corpus_loads_named_real_receipts():
    corpus = lab.load_prior_results()
    assert corpus["n_records"] == len(corpus["records"])
    assert corpus["n_records"] > 0
    loaded = corpus["named_receipts_loaded"]
    for rel in NAMED:
        assert rel in loaded, rel
    assert corpus["evidence_tier"] == "STATIC"
    for rec in corpus["records"]:
        for key in lab.RECORD_KEYS:
            assert key in rec, (rec.get("record_id"), key)
        assert rec["source_receipt"] in NAMED
        assert rec["evidence_tier"] in {"STATIC", "COST_MODEL"}
        assert rec["evidence_tier"] != "HARDWARE_MEASURED"


def test_corpus_names_sealed_incumbent_and_shannon_floor():
    corpus = lab.load_prior_results()
    sealed = lab.lookup_prior(corpus, "incumbent_sealed_3_14")
    assert sealed, "COMPLETE_EBPW incumbent was not loaded"
    axes = sealed[0]["axes"]
    assert axes["complete_ebpw"] == pytest.approx(3.1393, abs=0.001)
    assert axes["stored_bytes"] == 10554328856
    entropy = lab.lookup_prior(corpus, "entropy_code_mlp_codes")
    assert entropy, "REPRESENTATION_FLOOR entropy_code_mlp_codes was not loaded"
    assert entropy[0]["source_receipt"] == "receipts/future/REPRESENTATION_FLOOR.json"
    assert entropy[0]["status"] == "MEASURED"
    assert entropy[0]["axes"]["bytes_saved"] == 277697891


def test_lookup_prior_empty_means_unmeasured_not_zero():
    corpus = lab.load_prior_results()
    hits = lab.lookup_prior(corpus, "this_move_does_not_exist_zz")
    assert hits == []


def test_second_family_round_trips_and_is_not_named_in_core():
    nc.ensure_families()
    spec = nc.get_family("toy_mean_residual")
    assert spec.source_path == "tools/odyssey/families/toy_mean_residual.py"
    identity = lab.plugin_not_in_core("toy_mean_residual")
    assert identity["plugin"] is True
    assert identity["named_in_core_source"] is False
    assert identity["source_path_in_core_rels"] is False
    result = nc.round_trip("toy_mean_residual")
    assert result["verified"] is True
    assert result["reconciled"] is True
    assert result["execute"]["match_atol_1e5"] is True


def test_verify_family_calls_unbilled_gate_and_eval_hook():
    """CALL SITES: refuse_unbilled_components, score_representation_family."""
    report = lab.verify_family("toy_mean_residual")
    assert report["passed"] is True
    assert report["checks"]["ebpw_reconciled"] is True
    assert report["checks"]["scored_on_incumbent_axes"] is True
    assert report["score"]["evaluator_id"] == "representation.family_axes"
    assert report["score"]["same_axes_as_incumbent"] is True
    assert report["score"]["subject_kind"] == "representation"
    names = {p["name"] for p in report["accounting"]["parts"]}
    assert "toy_mean_residual_codes" in names
    assert "mean_residual_decoder_stub" in names
    assert report["accounting"]["is_sub2_executable"] is False


def test_claim_refuses_sub2_and_hardware_from_the_toy():
    allowed = lab.claim(
        "toy_mean_residual",
        "FUNCTIONAL_SIM micro-site; not a research candidate",
        asserted={"is_sub2_executable": False},
    )
    assert allowed["allowed"] is True
    with pytest.raises(lab.FamilyClaimRefused, match="sub-2"):
        lab.claim(
            "toy_mean_residual",
            "this is a sub-2 executable",
            asserted={"is_sub2_executable": True},
        )
    with pytest.raises(lab.FamilyClaimRefused, match="HARDWARE_MEASURED"):
        lab.claim(
            "toy_mean_residual",
            "hardware measured",
            asserted={"evidence_tier": "HARDWARE_MEASURED"},
        )
    with pytest.raises(lab.FamilyClaimRefused, match="sealed-3.14"):
        lab.claim(
            "toy_mean_residual",
            "beats the incumbent",
            asserted={"beats_sealed_incumbent": True},
        )


def test_family_sidecar_is_refused_by_the_unbilled_gate():
    """The gate's own symbol is invoked, not merely imported."""
    from tools.odyssey.families import toy_mean_residual as toy

    payload = toy.demo_payload()
    parts = toy.bill_parts(payload)
    parts["sidecar_codebook"] = [
        {
            "name": "hidden_free_codebook",
            "bytes": 64,
            "stream_class": ce.STREAM_WEIGHT_CODES,
        }
    ]
    probe = {
        "id": "toy_mean_residual:sidecar",
        "parent_params": 32,
        "stated_total_bytes": 0,
        **{c: parts.get(c, []) for c in ce.PART_CATEGORIES},
        "sidecar_codebook": parts["sidecar_codebook"],
        "reconstructs_dense_parent": False,
        "consumes_representation_directly": True,
    }
    with pytest.raises(ce.CompleteEbpwRefused, match="unbilled component|hidden free"):
        ce.refuse_unbilled_components(probe)
    with pytest.raises(ce.CompleteEbpwRefused, match="unbilled component|hidden free"):
        ce.candidate_from_parts(
            family_id="toy_mean_residual:sidecar",
            parent_params=32,
            parts=parts,
        )


def test_unbilled_guard_mutation_lets_family_sidecar_pass(monkeypatch):
    """Mutation check: removing refuse_unbilled_components hides the sidecar.

    The refusal test FAILS (cost succeeds, sidecar unbilled) when the guard
    is gone. Monkeypatch is the mutation; source is not written.
    """
    cand = ce.incumbent_candidate()
    cand["sidecar_codebook"] = [
        {
            "name": "hidden_free_codebook",
            "bytes": 1_000,
            "stream_class": ce.STREAM_WEIGHT_CODES,
        }
    ]
    with pytest.raises(ce.CompleteEbpwRefused, match="unbilled component|hidden free"):
        ce.refuse_unbilled_components(cand)

    monkeypatch.setattr(ce, "refuse_unbilled_components", lambda _c: None)
    row = ce.cost(cand)
    assert row["reconciled"] is True
    assert "hidden_free_codebook" not in {p["name"] for p in row["parts"]}


def test_score_family_uses_the_same_axes_as_local_incumbent():
    scored = lab.score_family("toy_mean_residual")
    assert scored["score"]["same_axes_as_incumbent"] is True
    assert scored["compared"]["same_axes"] == list(ce.COMPARE_AXES)
    cand = scored["compared"]["candidate_axes"]
    inc = scored["compared"]["incumbent_axes"]
    assert set(cand) == set(inc) == set(ce.COMPARE_AXES)
    # Local dense f32 is 32 bpw; the packed family must be smaller on this site.
    assert cand["complete_ebpw"] < inc["complete_ebpw"]
    assert cand["stored_bytes"] < inc["stored_bytes"]
    assert "complete_ebpw.refuse_unbilled_components" in ",".join(scored["call_sites"])
    assert "capability_eval.score_representation_family" in ",".join(scored["call_sites"])


def test_run_second_family_proof_and_build_receipt():
    proof = lab.run_second_family_proof()
    assert proof["family_id"] == "toy_mean_residual"
    assert proof["plugin"] is True
    assert proof["named_in_core_source"] is False
    assert proof["verified"] is True
    assert proof["sub2_claim_refused"] is True
    assert proof["hardware_claim_refused"] is True
    assert proof["same_axes_as_incumbent"] is True

    rc = lab.main(["--build"])
    assert rc == 0
    written = RECEIPTS / lab.RECEIPT
    doc = json.loads(written.read_text())
    assert doc["schema"] == lab.SCHEMA
    assert doc["not_a_sub2_search"] is True
    assert set(NAMED) <= set(doc["named_receipts_loaded"])
    assert doc["second_toy"]["family_id"] == "toy_mean_residual"
    assert doc["second_toy"]["named_in_core_source"] is False
    assert doc["second_toy"]["verified"] is True
    _assert_no_hardware_claims(doc)
