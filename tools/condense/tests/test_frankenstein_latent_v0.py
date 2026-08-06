#!/usr/bin/env python3.12
"""Tests for full latent V0 architecture + 11-loss schedule + reject gate."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_ablation as ablation  # noqa: E402
from lab.operators import frankenstein_bridges as bridges  # noqa: E402
from lab.operators import frankenstein_latent_v0 as lv  # noqa: E402
from lab.receipts import verify  # noqa: E402


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


def test_v0_module_names_match_steer() -> None:
    assert "GLM_EARLY_CONTEXT_BRIDGE" in lv.V0_BRIDGE_SITES
    assert "GLM_METHOD_CONDITIONED_ROUTE_RESIDUAL" in lv.V0_MODULE_NAMES
    assert "GLM_VALUE_HEAD" in lv.V0_MODULE_NAMES
    assert len(lv.V0_MODULE_NAMES) == 10
    assert set(lv.V0_BRIDGE_SITES).issubset(set(bridges.V0_BRIDGE_SITES))
    assert set(lv.LOSS_NAMES) == set(bridges.V0_LOSS_PORTFOLIO)


def test_teacher_student_latent_align_shapes() -> None:
    tp = lv.TeacherProjector(d_teacher=96, d_latent=32)
    so = lv.StudentObserver(d_student=64, d_latent=32)
    t = torch.randn(2, 3, 96)
    s = torch.randn(2, 3, 64)
    z_t = tp(t)
    z_s = so(s)
    assert z_t.shape == (2, 3, 32)
    assert z_s.shape == (2, 3, 32)
    assert tp.gravity_accounting()["training_only"] is True
    assert so.gravity_accounting()["runtime_resident"] is True


def test_student_intervention_apply_revert_bypass() -> None:
    mod = lv.StudentIntervention(
        name="GLM_METHOD_BRIDGE", d_model=32, rank=4, d_hidden=16, scale=0.1
    )
    x = torch.randn(3, 5, 32)
    y, r = mod.apply_with_residual(x)
    assert y.shape == x.shape
    x_back, info = mod.revert(y, residual=r)
    assert info["exact"] is True
    assert torch.allclose(x_back, x, atol=1e-5)
    mod.set_enabled(False)
    y2 = mod(x)
    assert torch.allclose(y2, x)
    grav = mod.gravity_accounting()
    assert grav["ablatable"] and grav["reversible"] and grav["kimi_bridge_compatible"]


def test_full_geometry_stack_builds() -> None:
    stack = lv.LatentV0Stack(
        d_teacher=lv.GLM_HIDDEN,
        d_student=lv.DSV4F_HIDDEN,
        d_latent=32,
        rank=4,
        d_hidden=32,
        n_experts=8,
    )
    assert stack.d_teacher == 6144
    assert stack.d_student == 4096
    assert set(stack.interventions.keys()) == set(lv.V0_BRIDGE_SITES)
    g = stack.gravity_accounting()
    assert g["total_parameter_bytes"] > 0
    assert g["runtime_parameter_bytes"] < g["total_parameter_bytes"]  # teacher excluded from runtime?
    # teacher is in total; runtime should exclude teacher projector bytes
    assert g["training_only_parameter_bytes"] > 0
    runtime = stack.runtime_state_dict()
    assert not any(k.startswith("teacher_projector.") for k in runtime)


def test_build_stack_for_each_latent_arm() -> None:
    for arm, _ in lv.LATENT_AG_ARMS:
        stack = lv.build_stack_for_arm(
            arm, d_teacher=48, d_student=32, d_latent=16, rank=2, d_hidden=16, n_experts=4
        )
        x = torch.randn(2, 3, 32)
        y, meta = stack.forward_hidden(x)
        assert y.shape == x.shape
        assert isinstance(meta["applied"], list)


def test_loss_rejects_latent_alone_outside_phase_a() -> None:
    w = lv.LossWeights(
        L_latent=1.0,
        L_function=0.0,
        L_span=0.0,
        L_method=0.0,
        L_decomposition=0.0,
        L_formal=0.0,
        L_repair=0.0,
        L_value=0.0,
        L_route=0.0,
        L_retention=0.0,
        L_runtime=0.0,
    )
    with pytest.raises(lv.LatentV0Error, match="FORBIDDEN"):
        w.validate_not_cosine_alone()
    w.allow_latent_only(True)
    w.validate_not_cosine_alone()  # phase A ok


def test_all_eleven_losses_named() -> None:
    assert len(lv.LOSS_NAMES) == 11
    for name in lv.LOSS_NAMES:
        assert name in lv.LOSS_DEFINITIONS
        assert name in lv.DEFAULT_LOSS_WEIGHTS


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


def test_requires_paired_capture_fail_closed() -> None:
    with pytest.raises(lv.RequiresPairedCapture):
        lv.require_paired_capture(None, allow_fixture=False, fixture_flag=False)
    with pytest.raises(lv.RequiresPairedCapture):
        lv.require_paired_capture(
            "/nonexistent/capture.pt", allow_fixture=False, fixture_flag=False
        )
    closed = lv.fail_closed_paired_capture(stage="test", operation="unit")
    verify(closed, label="fail closed")
    assert closed["status"] == "FAIL_CLOSED"
    assert closed["gate"] == lv.REQUIRES_PAIRED_CAPTURE
    assert closed["capability_claim"] is False
    assert closed["fabricated"] is False


def test_fit_real_cli_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pt"
    rc = lv.main(["fit-real", "--paired", str(missing)])
    assert rc == 3


# ---------------------------------------------------------------------------
# Fixture e2e: learns + reverses + reject fires
# ---------------------------------------------------------------------------


def test_fixture_e2e_learns_reverses_reject() -> None:
    doc = lv.run_fixture_end_to_end(
        fixture_cfg=lv.FixtureConfig(
            n_train=32,
            n_eval=8,
            seq_len=2,
            d_teacher=96,
            d_student=64,
            d_latent=32,
            rank=4,
            d_hidden=32,
            n_experts=8,
            batch_size=8,
            seed=11,
        ),
        train_cfg=lv.TrainConfig(
            epochs_per_phase=5,
            lr=5e-3,
            device="cpu",
            phases=("A", "B", "C", "E"),
        ),
        prove_reject=True,
    )
    verify(doc, label="latent fixture e2e")
    assert doc["capability_claim"] is False
    assert doc["fabricated_capability_number"] is False
    assert doc["real_glm_dsv4f_capture"] is False
    assert doc["learned"] is True, doc["result"]
    assert doc["reverse_ok"] is True, doc["result"]
    assert doc["bytes_accounted"] is True
    assert doc["tps_accounted"] is True
    assert doc["reject_proof"]["proved_reject_fires"] is True
    assert doc["fail_closed_real_train"]["gate"] == lv.REQUIRES_PAIRED_CAPTURE
    assert doc["promotion_gate"]["verdict"] == "HOLD_REQUIRES_PAIRED_CAPTURE"
    # Checkpoints written
    assert "CURRENT" in doc["checkpoints"] or doc["result"]["checkpoints"]
    # All 8 bridge sites ablatable
    for site in lv.V0_BRIDGE_SITES:
        assert site in doc["site_ablations"]


def test_train_ag_and_reject_proof() -> None:
    doc = lv.train_ag_variants(
        fixture_cfg=lv.FixtureConfig(
            n_train=24,
            n_eval=8,
            seq_len=2,
            d_teacher=64,
            d_student=48,
            d_latent=24,
            rank=4,
            d_hidden=24,
            n_experts=8,
            batch_size=8,
            seed=3,
        ),
        train_cfg=lv.TrainConfig(
            epochs_per_phase=3,
            lr=5e-3,
            device="cpu",
            phases=("A", "B", "E"),
        ),
        arms=[lv.ARM_C, lv.ARM_G],
    )
    verify(doc, label="latent ag")
    assert doc["capability_claim"] is False
    assert doc["reject_proof"]["proved"] is True
    assert doc["arm_results"][lv.ARM_G]["learned"] is True
    assert doc["arm_results"][lv.ARM_G]["reverse_ok"] is True
    assert doc["promotion_pending_real_capture"]["status"] == (
        "FRAMEWORK_PENDING_REAL_CAPTURE"
    )


def test_retention_gate_fires_on_secondary_regression() -> None:
    base = ablation.default_score_template(0.70)
    proto_sec = dict(base["secondary"])
    proto_sec["coding_and_repository_work"] = 0.50  # big drop
    gate = lv.retention_gate(
        base_secondary=base["secondary"], proto_secondary=proto_sec
    )
    verify(gate, label="retention")
    assert gate["reject_rule_fired"] is True
    assert gate["verdict"] == "REJECT"


def test_ablation_catalog_exports_latent_arms() -> None:
    cat = ablation.latent_v0_arm_catalog()
    ids = {a["id"] for a in cat["arms"]}
    assert ablation.ARM_LATENT_G in ids
    assert ablation.ARM_LATENT_C in ids
    assert len(cat["arms"]) == 7


def test_registry_seal() -> None:
    reg = lv.v0_bridge_registry()
    verify(reg, label="registry")
    assert len(reg["modules"]) == 10
    assert reg["properties"]["kimi_bridge_compatible"] is True
    assert reg["capability_claim"] is False
