"""Tests for the MLP error-budget probe.

Load-bearing negatives a guard nobody has watched fail is not a guard:

  * reporting the target relative error as if it were achieved is REFUSED
  * argmax agreement alone cannot be presented as parity
  * both error geometries and at least two scopes must be present
  * the headline tolerated-error number states its criterion
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tools.future import mlp_error_budget as meb
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def test_reporting_target_as_achieved_is_refused():
    """NEGATIVE CONTROL: a copied target is not a calibration."""
    with pytest.raises(meb.TargetReportedAsAchieved, match="not measured|achieved"):
        meb.validate_calibration(
            {
                "target_relative_l2": 0.01,
                "achieved_relative_l2": 0.01,
                "achieved_measured": False,
            }
        )
    with pytest.raises(meb.TargetReportedAsAchieved, match="missing achieved"):
        meb.validate_calibration({"target_relative_l2": 0.01})
    with pytest.raises(meb.TargetReportedAsAchieved):
        meb.validate_calibration(
            {
                "target_relative_l2": 0.01,
                "achieved_relative_l2": 0.01,
                "achieved_measured": True,
                "achieved_is_target": True,
                "n_injections": 4,
                "achieved_max_abs_drift_from_target": 0.0,
                "calibration_ok": True,
            }
        )
    with pytest.raises(meb.TargetReportedAsAchieved, match="achieved_values is None"):
        meb.emit_sweep_point(
            geometry=meb.ISOTROPIC,
            scope=meb.SCOPE_ONE,
            scope_layers=(0,),
            n_layers_total=2,
            target_relative_l2=0.01,
            achieved_values=None,  # type: ignore[arg-type]
            last_token_kl=0.0,
            mean_top5_agreement=1.0,
            n_tokens=1,
        )
    with pytest.raises(meb.TargetReportedAsAchieved, match="empty"):
        meb.emit_sweep_point(
            geometry=meb.ISOTROPIC,
            scope=meb.SCOPE_ONE,
            scope_layers=(0,),
            n_layers_total=2,
            target_relative_l2=0.01,
            achieved_values=(),
            last_token_kl=0.0,
            mean_top5_agreement=1.0,
            n_tokens=1,
        )


def test_drift_is_not_rewritten_as_the_target():
    """A sweep whose achieved error is not the target must say so."""
    point = meb.emit_sweep_point(
        geometry=meb.ISOTROPIC,
        scope=meb.SCOPE_ONE,
        scope_layers=(1,),
        n_layers_total=4,
        target_relative_l2=0.1,
        achieved_values=(0.5, 0.5, 0.5),
        last_token_kl=2.0,
        mean_top5_agreement=0.1,
        argmax_agreement=1.0,
        n_tokens=3,
    )
    assert point["target_relative_l2"] == pytest.approx(0.1)
    assert point["achieved_relative_l2"] == pytest.approx(0.5)
    assert point["achieved_relative_l2"] != point["target_relative_l2"]
    assert point["calibration_ok"] is False
    assert point["achieved_measured"] is True
    assert point["achieved_is_target"] is False
    assert point["band"] == meb.BAND_BREAKS
    # Argmax survived; that is not a pass.
    assert point["argmax_agreement"] == pytest.approx(1.0)
    assert point["argmax_is_not_parity"] is True
    assert point["usable"] is False


def test_honest_calibration_reports_achieved_beside_target():
    rng = np.random.default_rng(1)
    f = rng.standard_normal(32).astype(np.float32)
    q = meb.output_basis_from_down(
        rng.standard_normal((32, 40)).astype(np.float32), rank=6, rng=rng
    )
    achieved = []
    for _ in range(5):
        _hat, ach = meb.inject_relative_error(
            f, 0.03, meb.STRUCTURED, rng=rng, basis=q
        )
        achieved.append(ach)
    point = meb.emit_sweep_point(
        geometry=meb.STRUCTURED,
        scope=meb.SCOPE_FEW,
        scope_layers=(1, 2, 3),
        n_layers_total=8,
        target_relative_l2=0.03,
        achieved_values=achieved,
        last_token_kl=0.01,
        mean_top5_agreement=0.96,
        argmax_agreement=1.0,
        n_tokens=5,
    )
    assert point["target_relative_l2"] == pytest.approx(0.03)
    assert point["achieved_relative_l2"] == pytest.approx(0.03, abs=1e-6)
    assert point["calibration_ok"] is True
    assert point["n_injections"] == 5
    assert "achieved_max_abs_drift_from_target" in point
    meb.validate_calibration(point)


def test_module_refuses_to_report_argmax_agreement_alone_as_parity():
    """NEGATIVE CONTROL: argmax agreement is not parity."""
    with pytest.raises(meb.ArgmaxPresentedAsParity, match="not parity"):
        meb.present_as_parity({"argmax_agreement": 1.0})
    with pytest.raises(meb.ArgmaxPresentedAsParity):
        meb.present_as_parity({"argmax_identical": True, "top1_agreement": 1.0})
    with pytest.raises(meb.ArgmaxPresentedAsParity):
        meb.present_as_parity({"argmax": 1.0, "argmax_agreement": 0.99})
    with pytest.raises(meb.ArgmaxPresentedAsParity, match="argmax"):
        meb.usability_verdict(
            last_token_kl=None,
            mean_top5_agreement=None,
            argmax_agreement=1.0,
        )
    with pytest.raises(meb.ArgmaxPresentedAsParity):
        meb.refuse_argmax_as_parity(
            {
                "parity": True,
                "parity_metrics": ["argmax_agreement"],
                "argmax_agreement": 1.0,
            }
        )
    # KL + top-k is a usability verdict, never a parity claim.
    v = meb.usability_verdict(
        last_token_kl=0.01,
        mean_top5_agreement=0.95,
        argmax_agreement=1.0,
    )
    assert v["band"] == meb.BAND_USABLE
    assert v["parity"] is False
    assert v["argmax_is_not_parity"] is True
    assert "argmax" not in v["metrics_used"]
    meb.refuse_argmax_as_parity(v)
    # Even a fully scored record is refused as "parity".
    with pytest.raises(meb.ArgmaxPresentedAsParity):
        meb.present_as_parity(
            {
                "last_token_kl": 0.01,
                "mean_top5_agreement": 0.95,
                "argmax_agreement": 1.0,
            }
        )


def test_geometries_are_not_interchangeable():
    rng = np.random.default_rng(2)
    f = rng.standard_normal(64).astype(np.float32)
    q = meb.output_basis_from_down(
        rng.standard_normal((64, 80)).astype(np.float32), rank=8, rng=rng
    )
    hat_i, ach_i = meb.inject_relative_error(f, 0.1, meb.ISOTROPIC, rng=rng, basis=q)
    hat_s, ach_s = meb.inject_relative_error(f, 0.1, meb.STRUCTURED, rng=rng, basis=q)
    assert ach_i == pytest.approx(0.1, abs=1e-6)
    assert ach_s == pytest.approx(0.1, abs=1e-6)
    d_i = hat_i.astype(np.float64) - f.astype(np.float64)
    d_s = hat_s.astype(np.float64) - f.astype(np.float64)
    cos = float(
        (d_i @ d_s)
        / (np.linalg.norm(d_i) * np.linalg.norm(d_s))
    )
    assert abs(cos) < 0.95
    with pytest.raises(meb.ErrorBudgetRefuse, match="basis"):
        meb.inject_relative_error(f, 0.1, meb.STRUCTURED, rng=rng, basis=None)


def test_structured_does_not_silently_become_isotropic():
    rng = np.random.default_rng(3)
    f = rng.standard_normal(8).astype(np.float32)
    with pytest.raises(meb.ErrorBudgetRefuse, match="basis|structured"):
        meb.inject_relative_error(f, 0.05, meb.STRUCTURED, rng=rng, basis=None)
    with pytest.raises(meb.ErrorBudgetRefuse, match="unknown geometry"):
        meb.inject_relative_error(f, 0.05, "gaussian_proxy", rng=rng)


def test_selftest_fires_the_guards():
    out = meb.selftest()
    assert out["target_reported_as_achieved_refused"] is True
    assert out["none_achieved_values_refused"] is True
    assert out["argmax_presented_as_parity_refused"] is True
    assert out["argmax_verdict_without_kl_refused"] is True
    assert out["isotropic_calibrates"] is True
    assert out["structured_calibrates"] is True
    assert out["geometries_not_interchangeable"] is True


def _tiny_sweep(**kwargs):
    model = meb.TinyResidualLM(seed=meb.RNG_SEED, generate_new=2, **kwargs)
    return meb.run_sweep(model, with_generate=True)


def test_fixture_sweep_has_both_geometries_and_at_least_two_scopes():
    sweep = _tiny_sweep()
    geos = {p["geometry"] for p in sweep["points"]}
    scopes = {p["scope"] for p in sweep["points"]}
    assert geos == {meb.ISOTROPIC, meb.STRUCTURED}
    assert meb.SCOPE_ONE in scopes
    assert meb.SCOPE_ALL in scopes
    assert len(scopes) >= 2
    targets = {p["target_relative_l2"] for p in sweep["points"]}
    for t in meb.DEFAULT_TARGETS:
        assert any(abs(x - t) < 1e-12 for x in targets)
    for p in sweep["points"]:
        meb.validate_calibration(p)
        meb.refuse_argmax_as_parity(p)
        assert p["achieved_measured"] is True
        assert p["achieved_is_target"] is False
        assert "target_relative_l2" in p
        assert "achieved_relative_l2" in p
        assert p["physical_primitive"] in ATLAS_PRIMITIVES
        assert p["argmax_is_not_parity"] is True
        assert p["verdict"]["parity"] is False
        if p["calibration_ok"]:
            assert abs(p["achieved_relative_l2"] - p["target_relative_l2"]) <= p[
                "calibration_tol"
            ]


def test_headline_states_its_criterion():
    sweep = _tiny_sweep()
    head = sweep["headline"]
    assert "tolerated_per_layer_relative_l2" in head
    assert "degrades_at" in head
    assert "breaks_at" in head
    assert head["criterion"]["name"] == "last_token_kl_and_mean_top5"
    assert head["criterion"]["last_token_kl_usable"] == meb.USABLE_KL
    assert head["criterion"]["mean_top5_usable"] == meb.USABLE_TOP5
    assert head["criterion"]["argmax_is_not_parity"] is True
    assert "argmax_agreement" in head["criterion"]["not_authority"]
    assert "0.10" in head["criterion_defense"] or "0.1" in head["criterion_defense"]
    assert head["argmax_is_not_parity"] is True
    assert head["parity"] is False
    assert head["scope_of_headline"] == meb.SCOPE_ALL
    meb.refuse_argmax_as_parity(head)
    ans = sweep["answers"]
    assert "what_error_does_the_model_actually_tolerate" in ans
    assert ans["are_families_killed_at_0_92_still_dead"].startswith("YES")


def test_zero_error_is_a_null_control():
    model = meb.TinyResidualLM(seed=0, generate_new=0)
    base = model.capture_baseline()
    rng = np.random.default_rng(0)
    run = model.replay(
        base,
        perturb_layers=(model.n_layers // 2,),
        target=0.0,
        geometry=meb.ISOTROPIC,
        rng=rng,
    )
    assert run["achieved_values"]
    assert max(run["achieved_values"]) == pytest.approx(0.0, abs=1e-12)
    scored = meb.score_logits(base["logits"], run["logits"])
    assert scored["last_token_kl"] == pytest.approx(0.0, abs=1e-9)
    assert scored["mean_top5_agreement"] == pytest.approx(1.0)
    assert scored["argmax_is_not_parity"] is True


def test_usability_bands_are_kl_and_top5_not_argmax():
    usable = meb.usability_verdict(
        last_token_kl=0.05, mean_top5_agreement=0.90, argmax_agreement=0.0
    )
    assert usable["band"] == meb.BAND_USABLE
    assert usable["argmax_agreement"] == pytest.approx(0.0)
    dead = meb.usability_verdict(
        last_token_kl=0.0, mean_top5_agreement=0.90, argmax_agreement=1.0
    )
    assert dead["band"] == meb.BAND_USABLE
    breaks = meb.usability_verdict(
        last_token_kl=1.5, mean_top5_agreement=0.9, argmax_agreement=1.0
    )
    assert breaks["band"] == meb.BAND_BREAKS
    degrades = meb.usability_verdict(
        last_token_kl=0.4, mean_top5_agreement=0.6, argmax_agreement=1.0
    )
    assert degrades["band"] == meb.BAND_DEGRADES


def test_assemble_receipt_is_sealed_static_only():
    model = meb.TinyResidualLM(seed=1, generate_new=1, n_tokens=4)
    sweep = meb.run_sweep(model, with_generate=True)
    doc = meb.assemble_receipt(
        sweep, specimen_name="fixture_residual_stack", model_authority=False
    )
    assert doc["schema"] == meb.SCHEMA
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["headline"]["criterion"]["argmax_is_not_parity"] is True
    geos = {p["geometry"] for p in doc["sweep"]}
    scopes = {p["scope"] for p in doc["sweep"]}
    assert geos == {meb.ISOTROPIC, meb.STRUCTURED}
    assert len(scopes) >= 2
    for p in doc["sweep"]:
        assert "target_relative_l2" in p
        assert "achieved_relative_l2" in p
        assert p["achieved_measured"] is True
    assert doc["selftest"]["argmax_presented_as_parity_refused"] is True
    assert doc["selftest"]["target_reported_as_achieved_refused"] is True
    assert "tps" not in doc
    _assert_no_hardware_claims(doc)
    for field in HARDWARE_FIELDS:
        assert field not in doc


def test_on_disk_receipt_schema_if_present():
    path = RECEIPTS / meb.RECEIPT
    if not path.is_file():
        pytest.skip("receipt not written yet")
    doc = json.loads(path.read_text())
    assert doc["schema"] == meb.SCHEMA
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    _assert_no_hardware_claims(doc)
    assert "seal_sha256" in doc
    assert doc["headline"]["criterion"]["name"] == "last_token_kl_and_mean_top5"
    assert doc["headline"]["argmax_is_not_parity"] is True
    geos = {p["geometry"] for p in doc["sweep"]}
    scopes = {p["scope"] for p in doc["sweep"]}
    assert geos == {meb.ISOTROPIC, meb.STRUCTURED}
    assert len(scopes) >= 2
    for p in doc["sweep"]:
        meb.validate_calibration(p)
        assert p["target_relative_l2"] is not None
        assert p["achieved_relative_l2"] is not None
        assert p["achieved_measured"] is True
        assert p["achieved_is_target"] is False
    meb.refuse_argmax_as_parity(doc["headline"])
    if doc["specimen"]["model_authority"] is True:
        assert doc["specimen"]["source"]["real_forward_pass"] is True
        assert doc["specimen"]["source"].get("synthetic") is False


def test_score_logits_does_not_treat_argmax_as_authority():
    rng = np.random.default_rng(4)
    base = rng.standard_normal((3, 20)).astype(np.float32)
    pert = base.copy()
    # Flip the runner-up onto the argmax slot on the last token, keep mass nearby.
    last = base[-1].copy()
    order = np.argsort(last)
    last[order[-1]], last[order[-2]] = last[order[-2]], last[order[-1]]
    pert[-1] = last
    scored = meb.score_logits(base, pert)
    assert scored["argmax_is_not_parity"] is True
    assert "last_token_kl" in scored
    assert "mean_top5_agreement" in scored
    # Last-token argmax flipped; top-5 may still mostly agree.
    assert scored["last_token_argmax_agreement"] == pytest.approx(0.0)
    assert scored["last_token_kl"] > 0.0
