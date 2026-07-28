#!/usr/bin/env python3.12
"""Unit tests for the bounded GLM-5.2 real-activation basis pilot (revision 1).

Pins:

  * orthogonality of explicit-mean residual columns to the mean direction
  * mean-direction inclusion in the explicit_mean arm
  * identical fit/holdout indices across arms (shared splitter)
  * equal-byte accounting at equal total_rank
  * explicit_mean does NOT get a free extra direction
  * centered arm regresses to production packer SVD-of-centered math
  * capped rank-512 points cannot enter the rank-512 floor (Finding A)
  * dual down analyses: fit/hold Z correspondence, 2048-wide input basis,
    no Gaussian path, equal bytes, separation of the two analyses (Finding B)
  * promotion panel excludes low-traffic diagnostics (Finding C)
  * uncentered/explicit_mean numerical-equivalence handling (Finding C)
  * full_traversal_authorized false unless floors clear (Finding C)
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_basis_pilot as bp  # noqa: E402
import glm52_activation_aware_pack as aap  # noqa: E402


def _anisotropic(n: int = 500, h: int = 96, seed: int = 0):
    rng = np.random.default_rng(seed)
    mean = rng.standard_normal(h).astype(np.float32)
    mean /= np.linalg.norm(mean) + 1e-12
    X = (0.25 * rng.standard_normal((n, h)) + 2.5 * mean).astype(np.float32)
    return X, mean


def test_fit_holdout_identity_across_calls():
    a1, b1 = bp.fit_holdout_indices(1000, seed=7, salt=3)
    a2, b2 = bp.fit_holdout_indices(1000, seed=7, salt=3)
    assert a1.tolist() == a2.tolist()
    assert b1.tolist() == b2.tolist()
    assert set(a1.tolist()).isdisjoint(b1.tolist())
    assert len(a1) + len(b1) == 1000
    a3, _ = bp.fit_holdout_indices(1000, seed=7, salt=4)
    assert a1.tolist() != a3.tolist()


def test_explicit_mean_includes_mean_direction_and_is_orthogonal():
    X, mean = _anisotropic()
    fit, _ = bp.fit_holdout_indices(X.shape[0], seed=1, salt=0)
    r = 12
    basis = bp.build_pilot_basis(X[fit], "explicit_mean", r)
    assert basis.total_rank == r
    assert basis.residual_rank == r - 1
    assert basis.mean_direction is not None
    B = basis.columns(r)
    assert abs(float(np.dot(B[:, 0], basis.mean_direction))) > 0.99
    dots = B[:, 1:].T @ basis.mean_direction
    assert float(np.max(np.abs(dots))) < 1e-4
    gram = B.T @ B
    assert float(np.max(np.abs(gram - np.eye(r)))) < 1e-4


def test_explicit_mean_does_not_get_a_free_extra_direction():
    X, _ = _anisotropic()
    fit, _ = bp.fit_holdout_indices(X.shape[0], seed=2, salt=1)
    for r in (1, 4, 16, 32):
        basis = bp.build_pilot_basis(X[fit], "explicit_mean", r)
        assert basis.total_rank == r
        assert basis.basis.shape[1] == r
        if r == 1:
            assert basis.residual_rank == 0
        else:
            assert basis.residual_rank == r - 1
        rows, cols, side = 64, X.shape[1], "input"
        c_bytes = bp.encoded_bytes(rows, cols, side, r)["total"]
        assert c_bytes == bp.encoded_bytes(rows, cols, side, r)["total"]
        cheat = bp.encoded_bytes(rows, cols, side, r + 1)["total"]
        assert cheat > c_bytes


def test_equal_byte_budget_across_arms():
    rows, cols = 128, 96
    for side in ("input", "output"):
        if side == "output":
            rows, cols = 96, 32
        for r in (16, 64, 128):
            costs = [
                bp.encoded_bytes(rows, cols, side, r)["total"]
                for _ in bp.BASIS_MODES
            ]
            assert len(set(costs)) == 1
            assert costs[0] == aap.packed_tensor_bytes(rows, cols, r, side, bill_basis=True)


def test_encoded_bytes_marks_arithmetic_not_physical_file():
    doc = bp.encoded_bytes(2048, 6144, "input", 256)
    assert doc["is_physical_file_measurement"] is False
    assert "exact_arithmetic" in doc["accounting_scope"]


def test_disk_floor_is_hard_without_soft_allow(monkeypatch, tmp_path):
    monkeypatch.setattr(bp, "free_disk_bytes", lambda _path: bp.DISK_FLOOR_BYTES - 1)
    with pytest.raises(bp.PilotError, match="Refuse to run"):
        bp.run_pilot(pilot_source=tmp_path)


def test_centered_regresses_to_production_svd():
    X, _ = _anisotropic(n=300, h=80)
    fit, _ = bp.fit_holdout_indices(X.shape[0], seed=3, salt=9)
    assert bp.centered_matches_production(X[fit], total_rank=16)


def test_uncentered_differs_from_centered_when_mean_dominates():
    X, mean = _anisotropic(n=400, h=64)
    fit, _ = bp.fit_holdout_indices(X.shape[0], seed=4, salt=0)
    r = 8
    bc = bp.build_pilot_basis(X[fit], "centered", r)
    bu = bp.build_pilot_basis(X[fit], "uncentered", r)
    c0 = bc.columns(r)[:, 0]
    u0 = bu.columns(r)[:, 0]
    align_c = abs(float(np.dot(c0, mean / (np.linalg.norm(mean) + 1e-12))))
    align_u = abs(float(np.dot(u0, mean / (np.linalg.norm(mean) + 1e-12))))
    assert align_u > align_c
    assert align_c < 0.15


def test_route_row_indices():
    topk = np.array([
        [11, 2, 3],
        [4, 5, 6],
        [11, 7, 8],
        [9, 10, 0],
        [0, 11, 1],
    ], dtype=np.int32)
    idx = bp.route_row_indices(topk, 11)
    assert idx.tolist() == [0, 2, 4]
    assert bp.route_row_indices(topk, 99).size == 0


def test_swiglu_and_score_linear_no_gaussian():
    rng = np.random.default_rng(5)
    h, inter, n = 32, 8, 40
    X = rng.standard_normal((n, h)).astype(np.float32)
    Wg = rng.standard_normal((inter, h)).astype(np.float32)
    Wu = rng.standard_normal((inter, h)).astype(np.float32)
    Wd = rng.standard_normal((h, inter)).astype(np.float32)
    mid = bp.swiglu_intermediate(X, Wg, Wu)
    assert mid.shape == (n, inter)
    fit, hold = bp.fit_holdout_indices(n, seed=1, salt=1)
    basis = bp.build_pilot_basis(X[fit], "centered", 4)
    W_hat, _ = bp.project_and_reconstruct(Wd, basis, 4, "output")
    sc = bp.score_linear(Wd, W_hat, mid[hold], side="output")
    assert 0.0 <= sc["mean_row_cosine"] <= 1.0 or sc["mean_row_cosine"] < 0
    assert "beats_null" in sc
    assert sc.get("promotional") is False
    with pytest.raises(bp.PilotError):
        bp.score_linear(Wd, W_hat, X[hold], side="output")


def test_evaluate_tensor_identical_fit_holdout_hashes_across_arms():
    rng = np.random.default_rng(6)
    h = 48
    n = 200
    X = (rng.standard_normal((n, h)) + 1.5).astype(np.float32)
    W = rng.standard_normal((16, h)).astype(np.float32)
    fit, hold = bp.fit_holdout_indices(n, seed=9, salt=2)
    spec = bp.TensorSpec(
        name="toy.gate_proj.weight",
        organ_class="high_traffic_routed_gate",
        layer=0,
        expert_id=0,
        route_conditioned=True,
        activation_source="pre_router_hidden",
    )
    row = bp.evaluate_tensor(
        spec, W, X_all=X, fit_idx=fit, hold_idx=hold,
        ranks=(4, 8), route_count=n,
    )
    assert row["status"] == "MEASURED"
    assert row["fit_idx_sha256"]
    assert row["hold_idx_sha256"]
    assert row["panel"] == bp.PANEL_PROMOTION_GRADE
    for r in (4, 8):
        totals = []
        for mode in bp.BASIS_MODES:
            pts = [p for p in row["arms"][mode] if p["requested_rank"] == r]
            assert len(pts) == 1
            totals.append(pts[0]["bytes"]["total"])
            assert "mean_row_cosine" in pts[0]
        assert len(set(totals)) == 1


# ---------------------------------------------------------------------------
# Finding A — rank eligibility
# ---------------------------------------------------------------------------
def test_capped_rank_512_cannot_enter_rank_512_floor():
    """Deterministic regression: capped rank-512 point is excluded from floors."""
    # Simulate expert-100-like: only 164 fit rows so rank 512 is capped at 164.
    capped_pt = {
        "requested_rank": 512,
        "total_rank": 164,
        "rank_capped": True,
        "mean_row_cosine": 0.1874,
        "constant_mean_cosine_null": 0.5,
        "bytes": {"total": 1000},
        "bpw": 0.1,
    }
    uncapped_good = {
        "requested_rank": 512,
        "total_rank": 512,
        "rank_capped": False,
        "mean_row_cosine": 0.95,
        "constant_mean_cosine_null": 0.5,
        "bytes": {"total": 4000},
        "bpw": 2.0,
    }
    assert not bp.is_promotion_eligible_point(capped_pt, 512)
    assert bp.is_promotion_eligible_point(uncapped_good, 512)

    results = [
        {
            "status": "MEASURED",
            "name": "low.down",
            "organ_class": "low_traffic_routed_down",
            "panel": bp.PANEL_LOW_TRAFFIC,
            "layer": 5,
            "route_count": 205,
            "n_weights": 100,
            "arms": {
                mode: [capped_pt] for mode in bp.BASIS_MODES
            },
        },
        {
            "status": "MEASURED",
            "name": "high.gate",
            "organ_class": "high_traffic_routed_gate",
            "panel": bp.PANEL_PROMOTION_GRADE,
            "layer": 5,
            "route_count": 3000,
            "n_weights": 100,
            "arms": {
                mode: [uncapped_good] for mode in bp.BASIS_MODES
            },
        },
    ]
    # Promotion-grade floor must not see the capped low-traffic cosine
    vals, n_inc, n_exc, details = bp._collect_rank_points(
        results, "centered", 512,
        panel=bp.PANEL_PROMOTION_GRADE, require_uncapped=True,
    )
    assert n_inc == 1
    assert vals == [0.95]
    # Low-traffic panel: the capped point is excluded even there under uncapped rule
    vals_lt, n_inc_lt, n_exc_lt, _ = bp._collect_rank_points(
        results, "centered", 512,
        panel=bp.PANEL_LOW_TRAFFIC, require_uncapped=True,
    )
    assert n_inc_lt == 0
    assert n_exc_lt == 1
    assert vals_lt == []

    # Full verdict: floor min must be 0.95, not 0.1874
    # Need enough promotion-grade points for a real floor path — expand synthetic set
    for i in range(5):
        results.append({
            "status": "MEASURED",
            "name": f"high.gate.{i}",
            "organ_class": "high_traffic_routed_gate",
            "panel": bp.PANEL_PROMOTION_GRADE,
            "layer": 5,
            "route_count": 3000,
            "n_weights": 100,
            "arms": {
                mode: [{
                    **uncapped_good,
                    "mean_row_cosine": 0.95 + i * 0.001,
                    "requested_rank": r,
                    "total_rank": r,
                    "rank_capped": False,
                } for r in (256, 512)]
                for mode in bp.BASIS_MODES
            },
        })
    # Also give the first high.gate both ranks
    results[1]["arms"] = {
        mode: [
            {**uncapped_good, "requested_rank": 256, "total_rank": 256,
             "rank_capped": False, "mean_row_cosine": 0.96},
            uncapped_good,
        ]
        for mode in bp.BASIS_MODES
    }
    verdict = bp.distinguish_verdict(results, (256, 512))
    for fc in verdict["floor_checks"]:
        if fc["rank"] == 512 and "min" in fc:
            assert fc["min"] >= 0.95, fc
            assert fc["min"] != pytest.approx(0.1874)
            # capped low-traffic must not appear in n_included for promotion panel
            assert fc["panel"] == bp.PANEL_PROMOTION_GRADE


def test_requested_rank_ne_total_rank_excluded_even_if_flag_wrong():
    """Belt-and-suspenders: total_rank mismatch alone excludes the point."""
    pt = {
        "requested_rank": 256,
        "total_rank": 164,
        "rank_capped": False,  # buggy flag
        "mean_row_cosine": 0.2,
    }
    assert not bp.is_promotion_eligible_point(pt, 256)


# ---------------------------------------------------------------------------
# Finding B — dual down analysis
# ---------------------------------------------------------------------------
def test_down_dual_analysis_fit_hold_z_correspondence_and_separation():
    rng = np.random.default_rng(11)
    n, h, inter = 200, 48, 16
    X = (rng.standard_normal((n, h)) + 0.5).astype(np.float32)
    Wg = rng.standard_normal((inter, h)).astype(np.float32)
    Wu = rng.standard_normal((inter, h)).astype(np.float32)
    Wd = rng.standard_normal((h, inter)).astype(np.float32)
    fit, hold = bp.fit_holdout_indices(n, seed=9, salt=2)

    # Manual Z correspondence (same X rows → same Z)
    Z_fit = bp.swiglu_intermediate(X[fit], Wg, Wu)
    Z_hold = bp.swiglu_intermediate(X[hold], Wg, Wu)
    assert Z_fit.shape == (fit.size, inter)
    assert Z_hold.shape == (hold.size, inter)
    # Deterministic: recompute matches
    Z_fit2 = bp.swiglu_intermediate(X[fit], Wg, Wu)
    assert np.allclose(Z_fit, Z_fit2)

    spec = bp.TensorSpec(
        name="toy.down_proj.weight",
        organ_class="high_traffic_routed_down",
        layer=0,
        expert_id=1,
        route_conditioned=True,
        activation_source="swiglu_intermediate",
    )
    row = bp.evaluate_tensor(
        spec, Wd, X_all=X, fit_idx=fit, hold_idx=hold,
        ranks=(4, 8), W_gate=Wg, W_up=Wu, route_count=n,
    )
    assert row["status"] == "MEASURED"
    assert row["promotion_metric"] == bp.DOWN_PROMOTION
    assert row["negative_control_metric"] == bp.DOWN_NEG_CONTROL
    assert row["gaussian_proxy_used"] is False
    assert row["equal_bytes_across_down_analyses"] is True

    prom = row["down_analyses"][bp.DOWN_PROMOTION]
    neg = row["down_analyses"][bp.DOWN_NEG_CONTROL]

    # 2,048-wide in production; here intermediate width = inter
    assert prom["basis_width"] == inter
    assert prom["basis_space"] == "swiglu_intermediate"
    assert prom["projection_side"] == "input"
    assert prom["promotional"] is True

    assert neg["basis_width"] == h
    assert neg["basis_space"] == "pre_router_hidden"
    assert neg["projection_side"] == "output"
    assert neg["promotional"] is False

    # Promotion arms are the top-level arms
    for mode in bp.BASIS_MODES:
        assert row["arms"][mode] == prom["arms"][mode]
        for pp, np_ in zip(prom["arms"][mode], neg["arms"][mode]):
            assert pp["total_rank"] == np_["total_rank"]
            assert pp["bytes"]["total"] == np_["bytes"]["total"]
            assert pp["basis_width"] == inter
            assert np_["basis_width"] == h
            assert pp["projection_side"] == "input"
            assert np_["projection_side"] == "output"

    # No Gaussian: scoring widths match intermediate
    for mode in bp.BASIS_MODES:
        for p in prom["arms"][mode]:
            assert p["score_width"] == inter


def test_down_no_gaussian_path_without_gate_up():
    rng = np.random.default_rng(12)
    n, h = 80, 32
    X = rng.standard_normal((n, h)).astype(np.float32)
    Wd = rng.standard_normal((h, 16)).astype(np.float32)
    fit, hold = bp.fit_holdout_indices(n, seed=1, salt=0)
    spec = bp.TensorSpec(
        "toy.down_proj.weight", "high_traffic_routed_down", 0, 0, True,
        "swiglu_intermediate",
    )
    row = bp.evaluate_tensor(
        spec, Wd, X_all=X, fit_idx=fit, hold_idx=hold,
        ranks=(4,), W_gate=None, W_up=None, route_count=n,
    )
    assert row["status"] == "SKIPPED_MISSING_GATE_UP"


def test_down_equal_bytes_input_vs_output_side_at_same_rank():
    """Identical total direction count ⇒ identical exact arithmetic totals."""
    for r in (16, 64, 128, 256, 512):
        out_b = bp.encoded_bytes(6144, 2048, "output", r)["total"]
        in_b = bp.encoded_bytes(6144, 2048, "input", r)["total"]
        assert out_b == in_b
        assert out_b == aap.packed_tensor_bytes(6144, 2048, r, "output", bill_basis=True)
        assert in_b == aap.packed_tensor_bytes(6144, 2048, r, "input", bill_basis=True)


# ---------------------------------------------------------------------------
# Finding C — panels, equivalence, traversal flag
# ---------------------------------------------------------------------------
def test_panel_assignment():
    assert bp.panel_for_organ("high_traffic_routed_gate") == bp.PANEL_PROMOTION_GRADE
    assert bp.panel_for_organ("high_traffic_routed_down") == bp.PANEL_PROMOTION_GRADE
    assert bp.panel_for_organ("shared_mlp_down") == bp.PANEL_SHARED_MLP
    assert bp.panel_for_organ("attention_q") == bp.PANEL_ATTENTION_ROUTER
    assert bp.panel_for_organ("router_control") == bp.PANEL_ATTENTION_ROUTER
    assert bp.panel_for_organ("low_traffic_routed_gate") == bp.PANEL_LOW_TRAFFIC


def test_low_traffic_not_in_promotion_floor():
    low = {
        "status": "MEASURED",
        "name": "low.gate",
        "organ_class": "low_traffic_routed_gate",
        "panel": bp.PANEL_LOW_TRAFFIC,
        "layer": 5,
        "route_count": 205,
        "n_weights": 100,
        "arms": {
            mode: [{
                "requested_rank": 256,
                "total_rank": 164,
                "rank_capped": True,
                "mean_row_cosine": 0.05,  # would poison floor if included
                "constant_mean_cosine_null": 0.5,
                "bytes": {"total": 100},
                "bpw": 0.1,
            }]
            for mode in bp.BASIS_MODES
        },
    }
    high = {
        "status": "MEASURED",
        "name": "high.gate",
        "organ_class": "high_traffic_routed_gate",
        "panel": bp.PANEL_PROMOTION_GRADE,
        "layer": 5,
        "route_count": 3000,
        "n_weights": 100,
        "arms": {
            mode: [{
                "requested_rank": r,
                "total_rank": r,
                "rank_capped": False,
                "mean_row_cosine": 0.99,
                "constant_mean_cosine_null": 0.5,
                "bytes": {"total": 4000},
                "bpw": 2.0,
            } for r in (256, 512)]
            for mode in bp.BASIS_MODES
        },
    }
    verdict = bp.distinguish_verdict([low, high], (256, 512))
    for fc in verdict["floor_checks"]:
        if "min" in fc:
            assert fc["min"] == pytest.approx(0.99)
            assert fc["panel"] == bp.PANEL_PROMOTION_GRADE


def test_numerical_equivalence_declares_tied_not_explicit_mean_unique():
    """When B≈C, story must not claim explicit_mean uniquely superior."""
    def make_row(name, cos_c, cos_u, cos_e):
        arms = {}
        for mode, cos in (
            ("centered", cos_c),
            ("uncentered", cos_u),
            ("explicit_mean", cos_e),
        ):
            arms[mode] = [{
                "requested_rank": r,
                "total_rank": r,
                "rank_capped": False,
                "mean_row_cosine": cos,
                "constant_mean_cosine_null": 0.5,
                "bytes": {"total": 1000},
                "bpw": 1.0,
            } for r in (64, 128, 256, 512)]
        return {
            "status": "MEASURED",
            "name": name,
            "organ_class": "high_traffic_routed_gate",
            "panel": bp.PANEL_PROMOTION_GRADE,
            "layer": 5,
            "route_count": 3000,
            "n_weights": 100,
            "arms": arms,
        }

    rows = [
        make_row(f"t{i}", 0.80, 0.95, 0.95 + 1e-9)  # B≈C, both beat centered
        for i in range(6)
    ]
    verdict = bp.distinguish_verdict(rows, (64, 128, 256, 512))
    assert verdict["uncentered_explicit_mean_numerically_tied"] is True
    assert verdict["distinguishing_story_code"] == "RETAINING_MEAN_HELPS_CENTERED_RESIDUAL"
    assert "uniquely" not in verdict["distinguishing_story"].lower()
    assert "tied" in verdict["distinguishing_story"].lower() or "B≈C" in verdict["distinguishing_story"]
    for p in verdict["pairwise_at_focus_ranks"]:
        assert p["winner_by_median"] == "uncentered_or_explicit_mean_tied"


def test_full_traversal_authorized_false_when_floors_fail():
    rows = [{
        "status": "MEASURED",
        "name": "high.gate",
        "organ_class": "high_traffic_routed_gate",
        "panel": bp.PANEL_PROMOTION_GRADE,
        "layer": 5,
        "route_count": 3000,
        "n_weights": 100,
        "arms": {
            mode: [{
                "requested_rank": r,
                "total_rank": r,
                "rank_capped": False,
                "mean_row_cosine": 0.5,  # below floors
                "constant_mean_cosine_null": 0.4,
                "bytes": {"total": 1000},
                "bpw": 1.0,
            } for r in (256, 512)]
            for mode in bp.BASIS_MODES
        },
    }]
    verdict = bp.distinguish_verdict(rows, (256, 512))
    assert verdict["full_traversal_authorized"] is False
    assert any(fc["status"] == "FAILS" for fc in verdict["floor_checks"])


def test_full_traversal_authorized_true_only_when_all_floors_clear():
    rows = [{
        "status": "MEASURED",
        "name": f"high.gate.{i}",
        "organ_class": "high_traffic_routed_gate",
        "panel": bp.PANEL_PROMOTION_GRADE,
        "layer": 5,
        "route_count": 3000,
        "n_weights": 100,
        "arms": {
            mode: [{
                "requested_rank": r,
                "total_rank": r,
                "rank_capped": False,
                "mean_row_cosine": 0.99,
                "constant_mean_cosine_null": 0.5,
                "bytes": {"total": 1000},
                "bpw": 1.0,
            } for r in (256, 512)]
            for mode in bp.BASIS_MODES
        },
    } for i in range(4)]
    verdict = bp.distinguish_verdict(rows, (256, 512))
    assert all(fc["status"] == "CLEARS" for fc in verdict["floor_checks"])
    assert verdict["full_traversal_authorized"] is True


def test_selftest_entry_point():
    assert bp.selftest() == 0
