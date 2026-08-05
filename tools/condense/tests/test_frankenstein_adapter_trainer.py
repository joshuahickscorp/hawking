#!/usr/bin/env python3.12
"""Tests for reversible adapter trainer (synthetic loop + fail-closed real data)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_adapter_modules as modules  # noqa: E402
from lab.operators import frankenstein_adapter_trainer as trainer  # noqa: E402
from lab.operators import frankenstein_ablation as ablation  # noqa: E402
from lab.receipts import verify  # noqa: E402


# ---------------------------------------------------------------------------
# Modules: reverse + byte accounting
# ---------------------------------------------------------------------------


def test_bridge_apply_revert_and_hash() -> None:
    bridge = modules.ReversibleBridge(d_model=32, d_hidden=16, rank=4, scale=0.1)
    x = torch.randn(4, 8, 32)
    y = bridge(x)
    assert y.shape == x.shape
    assert float((y - x).detach().abs().max()) > 1e-4
    y2, r = bridge.apply_with_residual(x)
    x_exact, info_exact = bridge.revert(y2, residual=r)
    assert info_exact["exact"] is True
    assert torch.allclose(x_exact, x, atol=1e-5)
    x_back, info = bridge.revert(y.detach(), n_iters=20, atol=1e-5)
    assert info["recon_error"] < 1e-2
    assert torch.allclose(x_back, x, atol=1e-2)

    grav = bridge.gravity_accounting()
    assert grav["parameter_bytes"] > 0
    assert grav["hash_bound"] == bridge.content_hash()
    assert grav["ablatable"] is True


def test_adapter_bank_independent_ablation() -> None:
    bank = modules.AdapterBank(d_model=32, rank=4, n_experts=8)
    x = torch.randn(3, 4, 32)
    y, applied = bank.apply_residual(x)
    assert applied
    x_back = bank.revert_residual(y, applied)
    assert torch.allclose(x_back, x, atol=1e-3)
    y2, applied2 = bank.apply_residual(x, skip=["GLM_METHOD_ADAPTER"])
    assert "GLM_METHOD_ADAPTER" not in applied2
    assert bank.gravity_accounting()["glm_router_weights_copied"] is False


def test_route_bias_no_glm_router_copy() -> None:
    rb = modules.RouteBiasResidual(n_experts=8, n_methods=5, rank=2)
    logits = torch.randn(2, 8)
    method = torch.tensor([0, 3])
    out = rb(logits, method_ids=method)
    assert out.shape == logits.shape
    back = rb.revert(out, method_ids=method)
    assert torch.allclose(back, logits, atol=1e-5)
    assert rb.meta["copy_glm_router"] is False


def test_loss_rejects_cosine_alone() -> None:
    w = trainer.LossWeights(
        functional_output=0.0,
        token_action_kl=0.0,
        method_classification=0.0,
        route_behavior=0.0,
        verifier_outcome=0.0,
        latent_cosine=1.0,
    )
    with pytest.raises(trainer.TrainerError, match="FORBIDDEN"):
        w.validate_not_cosine_alone()


# ---------------------------------------------------------------------------
# Fail closed on real paired data
# ---------------------------------------------------------------------------


def test_requires_paired_data_fail_closed() -> None:
    with pytest.raises(trainer.RequiresPairedData):
        trainer.require_paired_data(None, allow_synthetic=False, synthetic_flag=False)
    with pytest.raises(trainer.RequiresPairedData):
        trainer.require_paired_data(
            "/nonexistent/paired.pt", allow_synthetic=False, synthetic_flag=False
        )
    closed = trainer.load_real_paired_or_fail(None)
    # path required — load_real with None
    closed = trainer.fail_closed_paired_data(
        stage="test", operation="unit"
    )
    verify(closed, label="fail closed")
    assert closed["status"] == "FAIL_CLOSED"
    assert closed["gate"] == trainer.REQUIRES_PAIRED_DATA
    assert closed["fabricated"] is False
    assert closed["capability_claim"] is False


def test_fit_real_cli_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pt"
    rc = trainer.main(["fit-real", "--paired", str(missing)])
    assert rc == 3


# ---------------------------------------------------------------------------
# Synthetic train: learns + reverses
# ---------------------------------------------------------------------------


def test_synthetic_e2e_learns_and_reverses() -> None:
    doc = trainer.run_synthetic_end_to_end(
        synth_cfg=trainer.SyntheticConfig(
            n_train=64,
            n_eval=16,
            seq_len=4,
            d_model=32,
            n_experts=8,
            n_actions=6,
            batch_size=8,
            seed=7,
        ),
        train_cfg=trainer.TrainConfig(epochs=35, lr=5e-3, device="cpu"),
    )
    verify(doc, label="synthetic e2e")
    assert doc["data_kind"] == "SYNTHETIC_PAIRED_ACTIVATION_FIXTURE"
    assert doc["capability_claim"] is False
    assert doc["fabricated_capability_number"] is False
    assert doc["learned"] is True, doc["result"]
    assert doc["reverse_ok"] is True, doc["result"]
    assert doc["result"]["bytes_account"]["total_parameter_bytes"] > 0
    # Independent ablation keys present for each residual adapter
    for name in modules.RESIDUAL_ADAPTER_NAMES:
        assert name in doc["adapter_ablations"]


def test_train_ag_variants_wires_reject_rule() -> None:
    doc = trainer.train_ag_variants(
        synth_cfg=trainer.SyntheticConfig(
            n_train=48,
            n_eval=12,
            seq_len=4,
            d_model=32,
            n_experts=8,
            n_actions=6,
            batch_size=8,
            seed=3,
        ),
        train_cfg=trainer.TrainConfig(epochs=20, lr=5e-3, device="cpu"),
        arms=[trainer.ARM_FT_D, trainer.ARM_FT_E, trainer.ARM_FT_G],
    )
    verify(doc, label="ag train")
    assert doc["capability_claim"] is False
    assert doc["real_glm_dsv4f_capture"] is False
    assert trainer.ARM_FT_D in doc["arm_results"]
    assert trainer.ARM_FT_G in doc["arm_results"]
    # At least the full stack should learn on synthetic teacher delta
    assert doc["arm_results"][trainer.ARM_FT_G]["learned"] is True
    assert doc["arm_results"][trainer.ARM_FT_G]["reverse_ok"] is True
    ab = doc["ablation"]
    assert "verdict" in ab
    assert ab.get("fabricated_scores") is False or ab.get("synthetic_proxy_scores") is True
    # Gate policy is the additive-not-subtractive one
    policy = ab.get("gate_policy") or ablation.sealed_gate_policy()
    assert policy["name"] == "additive_not_subtractive_stage1"


def test_build_stack_for_each_arm() -> None:
    for arm, _ in trainer.FUNCTIONAL_TRANSFER_ARMS:
        stack = modules.build_stack_for_arm(arm, d_model=16, d_hidden=8, rank=2, n_experts=4)
        x = torch.randn(2, 3, 16)
        y, meta = stack.forward_hidden(x)
        assert y.shape == x.shape
        assert isinstance(meta["applied"], list)


def test_default_loss_weights_not_cosine_alone() -> None:
    w = trainer.LossWeights()
    w.validate_not_cosine_alone()
    d = w.as_dict()
    assert d["functional_output"] > 0
    assert sum(v for k, v in d.items() if k != "latent_cosine") > 0
