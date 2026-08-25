"""N051 SENSITIVITY_ALLOCATION: proxy ranks, does not certify; N041 consumer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from sensitivity_allocation import (  # noqa: E402
    BIT_LEVELS,
    CLASSES,
    GENERATOR,
    N041_COMPLETE_EBPW,
    N041_FLOOR,
    N041_RECEIPT,
    NEGATIVE_SCIENCE,
    RECEIPT,
    SCHEMA,
    SOURCE_PARAM_COUNT,
    citation_exists,
    class_cuts,
    classify,
    fisher_diag,
    greedy_fill,
    mse_drop,
    q_mult,
    remaining_mse,
    score_linear,
    spearman,
)


def test_second_order_proxy_sees_001W_and_cosine_would_not():
    rng = np.random.RandomState(0)
    W = rng.randn(16, 32).astype(np.float32)
    X = rng.randn(24, 32).astype(np.float32)
    d = fisher_diag(X)
    a = score_linear(W, d)
    b = score_linear(0.01 * W, d)
    z = score_linear(W * 0.0, d)
    assert b["local_voi"] / a["local_voi"] == pytest.approx(1e-4, rel=1e-4)
    assert z["local_voi"] == pytest.approx(0.0, abs=1e-12)
    ident = score_linear(W, d)
    assert ident["local_voi"] == pytest.approx(a["local_voi"])


def test_activation_aware_is_not_weight_only():
    W = np.ones((4, 8), dtype=np.float32)
    d = np.zeros(8, dtype=np.float64)
    d[0] = 10.0
    d[1:] = 0.1
    sc = score_linear(W, d)
    assert sc["voi_in"][0] > 20.0 * sc["voi_in"][1]
    # isotropic d would rank every input channel the same (W is ones)
    d_iso = np.full(8, float(d.mean()))
    iso = score_linear(W, d_iso)
    assert iso["voi_in"].max() / max(iso["voi_in"].min(), 1e-30) == pytest.approx(1.0, rel=1e-6)


def test_fisher_diag_is_mean_square_not_gaussian():
    X = np.zeros((10, 4), dtype=np.float32)
    X[:, 2] = 3.0
    d = fisher_diag(X)
    assert d[2] == pytest.approx(9.0)
    assert d[0] == pytest.approx(0.0)
    assert d.shape == (4,)


def test_classes_are_the_five_and_rank_based():
    values = np.logspace(-6, 0, 1000)
    cuts = class_cuts(values)
    labels = [classify(v, cuts) for v in values]
    assert set(CLASSES) == {"disposable", "cheap", "ordinary", "sensitive", "critical"}
    assert set(labels) == set(CLASSES)
    assert classify(values[0], cuts) == "disposable"
    assert classify(values[-1], cuts) == "critical"
    n = {c: labels.count(c) for c in CLASSES}
    assert n["disposable"] == pytest.approx(50, abs=5)
    assert n["critical"] == pytest.approx(50, abs=5)
    assert n["ordinary"] == pytest.approx(500, abs=20)


def test_remaining_mse_falls_a_factor_of_four_per_bit():
    assert remaining_mse(16.0, 0.0) == pytest.approx(16.0)
    assert remaining_mse(16.0, 1.0) == pytest.approx(4.0)
    assert remaining_mse(16.0, 2.0) == pytest.approx(1.0)
    assert mse_drop(16.0, 1.0, 2.0) == pytest.approx(3.0)


def test_greedy_starves_cheap_and_feeds_sensitive_at_equal_bits():
    items = [
        {
            "id": "cheap",
            "organ": "mlp",
            "layer": 0,
            "tensor": "up_proj",
            "class": "disposable",
            "n_params": 1000,
            "voi": 1.0,
            "uniform_bpw": 2.25,
            "diagnostic": "test",
            "negative_science": [],
        },
        {
            "id": "hot",
            "organ": "mlp",
            "layer": 63,
            "tensor": "down_proj",
            "class": "critical",
            "n_params": 1000,
            "voi": 400.0,
            "uniform_bpw": 2.25,
            "diagnostic": "test",
            "negative_science": [],
        },
    ]
    budget = 2.25 * 2000
    rec = greedy_fill(items, budget)
    by = {a["id"]: a for a in rec["assignment"]}
    assert rec["hit_budget"] is True
    assert rec["bits"] == pytest.approx(budget, rel=1e-4)
    assert by["hot"]["recommended_bpw"] > by["cheap"]["recommended_bpw"]
    assert by["cheap"]["recommended_bpw"] < 2.25
    greedy_mse = sum(a["remaining_mse_proxy"] for a in rec["assignment"])
    uniform_mse = remaining_mse(1.0, 2.25) + remaining_mse(400.0, 2.25)
    assert greedy_mse < uniform_mse


def test_q_mult_spans_depth_and_is_cited_not_one():
    assert q_mult(0) == pytest.approx(1.0)
    assert q_mult(63) == pytest.approx(2.577e-03 / 1.597e-04, rel=1e-6)
    assert q_mult(63) > 10.0


def test_spearman_of_identical_ranks_is_one():
    x = np.arange(10.0)
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_bit_levels_include_campaign_densities():
    assert 1.25 in BIT_LEVELS
    assert 1.85 in BIT_LEVELS
    assert 2.25 in BIT_LEVELS
    assert 3.125 in BIT_LEVELS
    assert 2.25 == N041_FLOOR["mlp"]


def test_n041_complete_ebpw_is_cited_not_reopened():
    assert N041_COMPLETE_EBPW == pytest.approx(2.596888)
    assert N041_RECEIPT.endswith("WHOLE_MODEL_RECOMPOSE.json")
    assert citation_exists(N041_RECEIPT)
    for p in NEGATIVE_SCIENCE:
        assert citation_exists(p), p


def _receipt() -> dict:
    assert RECEIPT.is_file(), f"missing {RECEIPT} — run python3 tools/headless/sensitivity_allocation.py"
    return json.loads(RECEIPT.read_text())


def test_receipt_schema_proxy_discipline_cpu_no_new_floor():
    doc = _receipt()
    assert doc["schema"] == SCHEMA
    assert doc["generated_by"] == GENERATOR
    assert doc["hand_authored"] is False
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_touch_gpu"] is True
    assert doc["did_not_run_cargo_or_metal_benchmarks"] is True
    assert doc["did_not_mutate_noetic_parent_a"] is True
    assert doc["sensitivity_is_a_proxy"] is True
    assert doc["ranks_does_not_certify"] is True
    assert doc["does_not_claim_new_whole_model_floor"] is True
    assert doc["cited_n041_complete_ebpw"] == pytest.approx(N041_COMPLETE_EBPW)
    assert doc["capture"]["not_gaussian"] is True
    assert "post_attn_norm" in doc["capture"]["site"]
    assert doc["n_layers_scored"] == 64
    trap = doc["scale_trap"]
    assert trap["rejects_scaled_artifact"] is True
    assert trap["ratio_0p01_over_identity"] == pytest.approx(1e-4, rel=1e-3)
    assert trap["zero_voi"] == pytest.approx(0.0, abs=1e-12)


def test_receipt_classifies_five_ways_and_names_least_sensitive():
    doc = _receipt()
    assert doc["class_cuts"]["classes"] == list(CLASSES)
    organs = doc["organs_ranked_least_sensitive_first"]
    names = {o["organ"] for o in organs}
    assert {"mlp", "deltanet", "gqa", "embedding", "output"} <= names
    least = doc["least_sensitive_regions"]
    assert len(least) >= 5
    for r in least:
        assert r["n_params"] >= 1_000_000
        assert r["class"] in CLASSES
        assert "PROXY" in r["why"] or "proxy" in r["why"].lower()
        assert r["negative_science"]
        for p in r["negative_science"]:
            assert citation_exists(p)
    # ranked least-sensitive first
    vp = [o["voi_per_param"] for o in organs]
    assert vp == sorted(vp)
    layers = doc["layers_ranked_least_sensitive_first"]
    assert len(layers) == 64
    assert [L["voi_per_param"] for L in layers] == sorted(L["voi_per_param"] for L in layers)


def test_receipt_n041_consumer_equal_bits_curve_and_citations():
    doc = _receipt()
    c = doc["n041_consumer"]
    assert c["does_not_claim_new_whole_model_floor"] is True
    assert c["sensitivity_is_a_proxy"] is True
    assert c["ranks_does_not_certify"] is True
    assert c["baseline_complete_ebpw_cited"] == pytest.approx(N041_COMPLETE_EBPW)
    assert c["baseline_source"] == N041_RECEIPT
    assert set(c["uniform_organ_floors_cited"]) >= {"mlp", "deltanet", "gqa", "embedding", "output"}
    items = c["items"]
    assert len(items) >= 20
    for it in items[:8]:
        assert it["class"] in CLASSES
        assert it["n_params"] > 0
        assert "diagnostic" in it and "PROXY" in it["diagnostic"]
        assert it["negative_science"]
    assert len(c["curve"]) >= 3
    ebpws = [pt["complete_ebpw"] for pt in c["curve"]]
    assert ebpws == sorted(ebpws)
    # same-bit reallocation vs uniform
    a = doc["assignment"]
    assert a["greedy_beats_uniform_mse_proxy"] is True
    assert a["hit_budget"] is True
    rel = abs(a["greedy_bits"] - a["budget_bits"]) / max(a["budget_bits"], 1.0)
    assert rel < 1e-3
    assert doc["citations_missing"] == []
    for row in doc["negative_science"]:
        assert row["present"] is True
    # activation source is named
    src = doc["capture"]["source_note"]
    assert "capture_diverse2" in src or "post_attn_norm" in src
    assert "Gaussian" in src or "gaussian" in src.lower() or doc["capture"]["not_gaussian"] is True
    assert doc.get("s026_109_unseen_embed_locked") is True
    unseen = [t for t in doc["tensors"] if t["kind"] == "embed_tokens.unseen"]
    assert unseen, "unseen embed rows must be present and locked (S026 §109)"
    assert unseen[0]["locked"] is True
    assert unseen[0]["class"] != "disposable"
    # parent parameter count is the campaign's
    assert SOURCE_PARAM_COUNT == 26_895_998_464
    # illustrative complete EBPW must not be advertised as a new floor
    assert "not a new floor" in doc["headline"] or "not a new floor" in c["note"].lower()
    assert c["illustrative_complete_ebpw_at_same_gemv_bits"] == pytest.approx(
        c["uniform_complete_ebpw_from_measured_n"], rel=5e-3
    )


def test_receipt_every_tensor_has_a_site_and_unlocked_have_channels():
    doc = _receipt()
    n_unlocked = 0
    organs_seen = set()
    for t in doc["tensors"]:
        organs_seen.add(t["organ"])
        assert t["activation_site"]
        assert t["activation_site_status"] in {
            "MEASURED",
            "PROXY_SITE",
            "PROXY_SUBSAMPLE",
            "WEIGHT_ONLY",
            "UNIFORM_PRIOR",
            "UNMEASURED_IN_CAPTURE",
        }
        assert t["class"] in CLASSES
        assert t["n_params"] > 0
        if not t["locked"]:
            n_unlocked += 1
            assert sum(t["channel_class_counts"].values()) == t["n_out"]
    assert n_unlocked >= 100
    assert {"mlp", "deltanet", "gqa", "embedding", "output"} <= organs_seen
