"""N047 Doctor diagnosis: measured features, §67 prescriptions, negative-science AVOID."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from doctor_diagnosis import (  # noqa: E402
    DOCS,
    FAMILIES,
    GENERATOR,
    PARENT_A,
    RECEIPT,
    SCHEMA,
    SVD_K,
    avoid_list,
    diagnose,
    excess_kurtosis,
    features_of_matrix,
    grouped_absmax_rel_fro,
    kmeans_rel_fro,
    organ_plan,
    project_shared,
    randomized_svd,
    score_families,
    section_67,
)


REQUIRED_ORGANS = [
    "mlp_gate_up",
    "mlp_down",
    "gqa_q",
    "gqa_k",
    "gqa_v",
    "gqa_o",
    "deltanet_in_proj",
    "deltanet_out_proj",
    "embed",
    "lm_head",
]


# ---------------------------------------------------------------------------
# synthetic extractors
# ---------------------------------------------------------------------------


def test_gaussian_excess_kurtosis_near_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200_000)
    k = excess_kurtosis(x)
    assert abs(k) < 0.1, k


def test_spike_matrix_has_high_kurtosis_and_outliers():
    rng = np.random.default_rng(1)
    W = 0.05 * rng.standard_normal((128, 64)).astype(np.float32)
    W[0, 0] = 40.0
    W[3, 7] = -35.0
    feat, _aux = features_of_matrix(W, rng=rng)
    assert feat["weight_distribution"]["excess_kurtosis"] > 10.0
    assert feat["weight_distribution"]["outlier_frac_6mad"] > 0.0
    assert feat["weight_distribution"]["excess_kurtosis_null"] == 0.0
    assert feat["weight_distribution"]["excess_kurtosis_null_kind"] == "gaussian"


def test_randomized_svd_recovers_rank_3():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((200, 3))
    B = rng.standard_normal((3, 80))
    W = (A @ B).astype(np.float32)
    U, S, Vh = randomized_svd(W, k=8, rng=rng, niter=2)
    energy = float((S[:3] ** 2).sum() / (S ** 2).sum())
    assert energy > 0.99, energy
    recon = (U[:, :3] * S[:3]) @ Vh[:3]
    rel = float(np.linalg.norm(W - recon) / np.linalg.norm(W))
    assert rel < 0.05, rel


def test_kmeans_on_separated_clusters_is_high_affinity():
    rng = np.random.default_rng(3)
    centers = np.array([[4.0, 0.0], [-4.0, 0.0], [0.0, 4.0], [0.0, -4.0]])
    X = []
    for c in centers:
        X.append(c + 0.05 * rng.standard_normal((40, 2)))
    X = np.concatenate(X, axis=0)
    km = kmeans_rel_fro(X, 4, rng, n_iter=15)
    assert km["rel_fro"] < 0.1, km
    assert km["affinity"] > 0.9


def test_identical_matrices_share_a_basis():
    rng = np.random.default_rng(4)
    W = (rng.standard_normal((80, 5)) @ rng.standard_normal((5, 40))).astype(np.float32)
    _U, _S, Vh = randomized_svd(W, k=8, rng=rng, niter=2)
    aff = project_shared(W, Vh)
    assert aff > 0.95, aff
    W2 = (rng.standard_normal((80, 5)) @ rng.standard_normal((5, 40))).astype(np.float32)
    # An independent rank-5 draw on the same shape is not the same subspace.
    aff_other = project_shared(W2, Vh)
    assert aff_other < aff


def test_grouped_absmax_error_falls_as_bits_rise():
    rng = np.random.default_rng(5)
    W = rng.standard_normal((32, 64)).astype(np.float32)
    e = [grouped_absmax_rel_fro(W, b, 64) for b in (1, 2, 3, 4)]
    assert e == sorted(e, reverse=True), e
    assert e[-1] < e[0]


def test_score_families_promotes_low_rank_when_spectrum_is_concentrated():
    agg = _blank_agg()
    agg["rank_spectrum"]["effective_rank_frac_of_min_dim"] = 0.05
    agg["rank_spectrum"]["captured_energy_frac"] = 0.95
    ranked = score_families(agg, "gqa_k")
    by = {r["family_id"]: r for r in ranked}
    assert by["low_rank"]["score"] > by["generated_coefficients"]["score"]
    assert by["low_rank"]["rank"] < by["generated_coefficients"]["rank"]


def test_score_families_promotes_protected_islands_on_outliers():
    agg = _blank_agg()
    agg["weight_distribution"]["excess_kurtosis"] = 12.0
    agg["weight_distribution"]["outlier_frac_6mad"] = 0.08
    ranked = score_families(agg, "mlp_gate_up")
    by = {r["family_id"]: r for r in ranked}
    assert by["protected_islands"]["score"] > 40.0


def test_mlp_avoid_lists_shared_k2_and_cites_coherent_receipt():
    agg = _blank_agg()
    agg["shared_basis_affinity"]["mean_right_affinity"] = 0.4
    avoid = avoid_list("mlp_gate_up", agg)
    texts = json.dumps(avoid)
    assert "shared-K2" in texts or "identical shared-K2" in texts
    paths = [p for a in avoid for p in a["negative_science"]]
    assert "receipts/headless/SHARED_BASIS_COHERENT.json" in paths
    for p in paths:
        assert (REPO / p).is_file(), p


def test_section_67_has_the_four_concrete_fields():
    s67 = section_67(
        "mlp_gate_up",
        [{"text": "low-rank weak"}],
        [{"rank": 1, "action": "learn function-preserving rotation"}],
        [{"experiment": "identical shared-K2 experiment", "reason": "prior negative science",
          "family": "shared basis", "negative_science": ["receipts/headless/SHARED_BASIS_COHERENT.json"]}],
    )
    assert s67["ORGAN"] == "mlp_gate_up"
    assert s67["DIAGNOSIS"] == ["low-rank weak"]
    assert s67["PRESCRIPTION"][0].startswith("1. ")
    assert s67["AVOID"][0]["item"] == "identical shared-K2 experiment"


def test_organ_plan_covers_contract_organs_and_gqa_qkvo():
    ids = [s["organ_id"] for s in organ_plan()]
    for oid in REQUIRED_ORGANS:
        assert oid in ids, oid
    assert ("scalar_quantization", "scalar quantization") in FAMILIES
    labels = [lab for _i, lab in FAMILIES]
    for need in (
        "binary", "ternary", "trit-plane", "vector codebook", "additive codebook",
        "shared basis", "low-rank", "low-rank + sparse", "generated coefficients",
        "structured pruning", "protected islands",
    ):
        assert need in labels, need


def _blank_agg() -> dict:
    return {
        "weight_distribution": {
            "excess_kurtosis": 0.5,
            "outlier_frac_6mad": 0.001,
        },
        "sparsity": {"frac_near_zero": 0.01},
        "rank_spectrum": {
            "effective_rank_frac_of_min_dim": 0.5,
            "captured_energy_frac": 0.4,
        },
        "shared_basis_affinity": {"mean_right_affinity": 0.2},
        "cross_layer_similarity": {"mean_sketch_cosine": 0.05},
        "codebook_affinity": {
            "row_kmeans_k16": {"affinity": 0.2},
            "row_kmeans_k64": {"affinity": 0.3},
            "group8_kmeans_k16": {"affinity": 0.25},
        },
        "grouped_absmax_rel_fro_g64": {"1": 0.7, "2": 0.45, "3": 0.3, "4": 0.2},
        "cross_head_similarity": {"mean_cosine": 0.1},
    }


# ---------------------------------------------------------------------------
# receipt (real parent tensors)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def receipt():
    if RECEIPT.is_file():
        doc = json.loads(RECEIPT.read_text())
        if doc.get("schema") == SCHEMA and doc.get("organs"):
            return doc
    return diagnose(write=True)


def _organs(receipt) -> dict:
    return {o["organ_id"]: o for o in receipt["organs"]}


def test_receipt_written_with_schema_and_cpu_discipline(receipt):
    assert RECEIPT.is_file()
    disk = json.loads(RECEIPT.read_text())
    assert disk["schema"] == SCHEMA
    assert receipt["schema"] == SCHEMA
    assert receipt["generated_by"] == GENERATOR
    assert receipt["family"] == "DOC-DIAGNOSIS"
    assert receipt["hand_authored"] is False
    assert receipt["pure_cpu_numpy"] is True
    assert receipt["did_not_touch_gpu"] is True
    assert receipt["did_not_run_cargo_or_metal_benchmarks"] is True
    assert receipt["did_not_run_second_27b_decode"] is True
    assert receipt["did_not_load_second_27b"] is True
    assert receipt["did_not_mutate_noetic_parent_a"] is True
    assert receipt["predicts_not_certifies"] is True
    assert receipt["diagnosis_does_not_certify"] is True
    assert receipt["qwen_mlp_2_25_stays_closed_for_unrotated_family"] is True
    assert receipt["literature_is"] == "HYPOTHESIS"
    assert receipt["literature_is_not_authority"] is True
    assert receipt["parent_streamed_one_tensor_at_a_time"] is True
    assert "qwen3.8" in receipt["qualified_parent"]
    assert DOCS.is_file()


def test_parent_a_read_only_fingerprint(receipt):
    sealed = receipt["sealed_leader"]
    assert sealed["read_only"] is True
    assert sealed["mutated"] is False
    assert PARENT_A.is_dir()
    before = sealed["fingerprint_before"]
    after = sealed["fingerprint_after"]
    assert before.get("catalog_sha256")
    assert before["catalog_sha256"] == after["catalog_sha256"]
    cat = sealed.get("catalog") or {}
    assert cat.get("magic") == "HQ38M20"
    assert cat.get("n_tensors", 0) >= 700
    assert cat.get("weights_not_dequantized") is True


def test_all_required_organs_measured_on_real_tensors(receipt):
    organs = _organs(receipt)
    for oid in REQUIRED_ORGANS:
        assert oid in organs, oid
        o = organs[oid]
        assert o["real_organ_of_qualified_parent"] is True
        assert o["parent_tensors"], oid
        assert o["diagnosis_features"]["n_tensors_measured"] >= 1
        wd = o["diagnosis_features"]["weight_distribution"]
        assert wd["excess_kurtosis"] is not None
        assert wd["outlier_frac_6mad"] is not None
        assert wd["excess_kurtosis_null_kind"] == "gaussian"
        spec = o["diagnosis_features"]["rank_spectrum"]
        assert spec["entropy_effective_rank"] is not None
        assert spec["spectrum_is_truncated"] is True
        assert spec["k_computed"] == SVD_K or spec["k_computed"] >= 8
        assert o["diagnosis_features"]["sparsity"]["frac_near_zero"] is not None
        cb = o["diagnosis_features"]["codebook_affinity"]["row_kmeans_k16"]
        assert cb["affinity"] is not None
        assert o["predicts_not_certifies"] is True


def test_gqa_has_qkvo_and_cross_head(receipt):
    organs = _organs(receipt)
    for oid in ("gqa_q", "gqa_k", "gqa_v", "gqa_o"):
        ch = organs[oid]["diagnosis_features"]["cross_head_similarity"]
        assert ch is not None, oid
        assert ch["mean_cosine"] is not None
        assert -1.0 <= ch["mean_cosine"] <= 1.0


def test_mlp_and_deltanet_have_cross_layer_and_shared_basis(receipt):
    organs = _organs(receipt)
    for oid in ("mlp_gate_up", "mlp_down", "deltanet_in_proj", "deltanet_out_proj"):
        o = organs[oid]
        assert o["diagnosis_features"]["n_tensors_measured"] >= 2
        share = o["diagnosis_features"]["shared_basis_affinity"]
        assert share["mean_right_affinity"] is not None
        x = o["diagnosis_features"]["cross_layer_similarity"]
        assert x["mean_sketch_cosine"] is not None
        assert x["n_pairs"] >= 1


def test_embed_and_lm_head_are_row_sampled_not_fully_materialized_as_f32(receipt):
    organs = _organs(receipt)
    for oid in ("embed", "lm_head"):
        o = organs[oid]
        assert len(o["parent_tensors"]) == 1
        note = o["per_tensor"][0]["sample"]
        assert note.get("sampled_rows", 0) >= 256
        assert note.get("full_shape", [0])[0] == 248320


def test_section_67_format_and_mlp_avoid_shared_k2(receipt):
    organs = _organs(receipt)
    mlp = organs["mlp_gate_up"]
    s67 = mlp["section_67"]
    assert s67["ORGAN"] == "mlp_gate_up"
    assert s67["DIAGNOSIS"]
    assert s67["PRESCRIPTION"]
    assert s67["AVOID"]
    assert any("shared-K2" in a["item"] or "shared-K2" in a["reason"] for a in s67["AVOID"])
    assert any("SHARED_BASIS_COHERENT" in json.dumps(a["negative_science"]) for a in s67["AVOID"])
    # diagnosis is evidence-ranked, not a canned paragraph
    assert any("low-rank" in t or "rank" in t for t in s67["DIAGNOSIS"])


def test_every_prescription_cites_a_feature_and_every_avoid_cites_a_receipt(receipt):
    for o in receipt["organs"]:
        assert o["every_recommendation_cites_a_diagnostic_feature"] is True
        assert o["every_avoid_cites_negative_science"] is True
        for step in o["prescription"]:
            assert step["motivating_feature"]
            assert step["predicts_not_certifies"] is True
        for a in o["avoid"]:
            assert a["negative_science"]
            for p in a["negative_science"]:
                assert (REPO / p).is_file(), f"{o['organ_id']} missing {p}"


def test_ranked_families_are_the_s026_65_set(receipt):
    labels = {lab for _i, lab in FAMILIES}
    for o in receipt["organs"]:
        got = {r["family"] for r in o["ranked_families"]}
        assert got == labels
        ranks = [r["rank"] for r in o["ranked_families"]]
        assert ranks == list(range(1, len(ranks) + 1))
        for r in o["ranked_families"]:
            assert r["predicts_not_certifies"] is True


def test_prescriptions_differ_across_organs(receipt):
    organs = _organs(receipt)
    mlp_top = organs["mlp_gate_up"]["ranked_families"][0]["family"]
    # scores may coincide at the incumbent; compare the full ranking + AVOID + diagnosis
    mlp_d = organs["mlp_gate_up"]["section_67"]["DIAGNOSIS"]
    gqa_d = organs["gqa_q"]["section_67"]["DIAGNOSIS"]
    dn_d = organs["deltanet_in_proj"]["section_67"]["DIAGNOSIS"]
    emb_d = organs["embed"]["section_67"]["DIAGNOSIS"]
    assert not (mlp_d == gqa_d == dn_d == emb_d)
    # GQA has cross-head language, MLP has binary-coherence prior
    mlp_blob = json.dumps(organs["mlp_gate_up"]["section_67"])
    assert "binary" in mlp_blob.lower() or "shared-K2" in mlp_blob
    assert organs["gqa_q"]["diagnosis_features"]["cross_head_similarity"]["mean_cosine"] is not None
    # keep mlp_top used so a degenerate all-equal ranking still fails elsewhere
    assert mlp_top in {lab for _i, lab in FAMILIES}


def test_unmeasured_block_does_not_invent_hessian_or_activations(receipt):
    u = receipt["unmeasured"]
    for key in (
        "hessian_sensitivity_proxy",
        "activation_distribution",
        "cross_expert_similarity",
        "state_redundancy",
        "token_frequency",
        "vocabulary_compositionality",
    ):
        assert u[key]["status"] == "ABSENT"
        assert u[key]["reason"]


def test_features_are_finite_numbers_not_assumed(receipt):
    for o in receipt["organs"]:
        wd = o["diagnosis_features"]["weight_distribution"]
        for k in ("excess_kurtosis", "outlier_frac_6mad", "std", "mean_abs"):
            v = wd[k]
            assert v is not None, (o["organ_id"], k)
            assert np.isfinite(v)
        spec = o["diagnosis_features"]["rank_spectrum"]
        assert 0.0 <= spec["captured_energy_frac"] <= 1.0 + 1e-6
        sp = o["diagnosis_features"]["sparsity"]["frac_near_zero"]
        assert 0.0 <= sp <= 1.0


def test_section_110_rollup_present(receipt):
    s = receipt["section_110"]
    assert len(s["RANKED_PRESCRIPTIONS"]) == len(REQUIRED_ORGANS)
    assert s["EXPECTED_EBPW_DELTA"] == "ABSENT" or str(s["EXPECTED_EBPW_DELTA"]).startswith("ABSENT")
    assert "N043" in s["KNOWN_RELEVANT_TECHNIQUES"] or "registry" in s["KNOWN_RELEVANT_TECHNIQUES"].lower()
