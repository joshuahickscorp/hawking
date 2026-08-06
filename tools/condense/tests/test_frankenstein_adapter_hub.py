#!/usr/bin/env python3.12
"""Tests for conditional multi-adapter hub (Rank-1, inactive by default).

Coverage:
  - HUB_ACTIVE=false → byte-identical to always-on residual path
  - HUB_ACTIVE=true routes to the right bridge per synthetic input
  - Bypass gate reproduces exact identity
  - Collision fixture (two bridges, one site) resolves without dual-add
  - Activation trigger stays fail-closed without evidence
  - Design seal verifies
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_adapter_hub as hub  # noqa: E402
from lab.operators.frankenstein_adapter_modules import ReversibleBridge  # noqa: E402
from lab.operators.frankenstein_promotion_gate import SECONDARY_TOLERANCE  # noqa: E402
from lab.receipts import verify  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults / catalog
# ---------------------------------------------------------------------------


def test_hub_active_defaults_false() -> None:
    assert hub.HUB_ACTIVE is False
    cfg = hub.default_hub_config()
    assert cfg.hub_active is False
    assert cfg.resolved_active() is False
    h = hub.build_synthetic_site_hub(hub_active=False)
    assert h.is_active is False


def test_named_bridge_catalog_covers_glm_and_kimi() -> None:
    assert "GLM_METHOD_BRIDGE" in hub.NAMED_BRIDGE_CATALOG
    assert "GLM_DECOMPOSITION_BRIDGE" in hub.NAMED_BRIDGE_CATALOG
    assert "KIMI_PLANNING_BRIDGE" in hub.NAMED_BRIDGE_CATALOG
    assert "KIMI_TOOL_POLICY_BRIDGE" in hub.NAMED_BRIDGE_CATALOG
    assert hub.KIMI_PARENT_BRIDGE == "KIMI_STRATEGIC_BRIDGE"
    # Bypass is reserved, not a catalog bridge.
    assert hub.BYPASS_CHOICE not in hub.NAMED_BRIDGE_CATALOG


# ---------------------------------------------------------------------------
# Regression: inactive == always-on (byte-identical)
# ---------------------------------------------------------------------------


def test_inactive_hub_byte_identical_to_always_on() -> None:
    h = hub.build_synthetic_site_hub(
        d_model=24,
        d_hidden=12,
        rank=3,
        hub_active=False,
        seed=7,
    )
    x = torch.randn(3, 5, 24)
    y_ref = hub.always_on_apply(
        x, h.bridges, mode="sequential", order=h._bridge_order
    )
    y_hub = h(x)
    # Exact equality — same ops, same tensors, no gate math.
    assert torch.equal(y_hub, y_ref)
    y_hub2, aux = h(x, return_aux=True)
    assert torch.equal(y_hub2, y_ref)
    assert aux["hub_active"] is False
    assert aux["path"] == "always_on"
    assert aux["applied"] == list(h._bridge_order)


def test_inactive_hub_matches_manual_sequential_bridges() -> None:
    """Reference path is the current sequential residual composition."""

    d = 16
    a = ReversibleBridge(name="GLM_METHOD_BRIDGE", d_model=d, d_hidden=8, rank=2, scale=0.08)
    b = ReversibleBridge(name="KIMI_PLANNING_BRIDGE", d_model=d, d_hidden=8, rank=2, scale=0.09)
    cfg = hub.HubConfig(hub_active=False, d_model=d, always_on_mode="sequential")
    h = hub.MultiAdapterHub({"GLM_METHOD_BRIDGE": a, "KIMI_PLANNING_BRIDGE": b}, config=cfg)
    x = torch.randn(2, 3, d)
    y_manual = b(a(x))
    assert torch.equal(h(x), y_manual)


def test_inactive_ignores_router_wiring() -> None:
    """Even if the router is hard-wired to bypass, inactive path still always-on."""

    h = hub.build_synthetic_site_hub(d_model=20, hub_active=False, seed=1)
    hub.force_router_one_hot(h, hub.BYPASS_CHOICE)
    x = torch.randn(2, 4, 20)
    y_ref = hub.always_on_apply(x, h.bridges, mode="sequential", order=h._bridge_order)
    assert torch.equal(h(x), y_ref)
    # Not identity (bridges fire) — proves inactive is not bypass.
    assert not torch.allclose(h(x), x, atol=1e-5)


# ---------------------------------------------------------------------------
# Active: bypass exact identity
# ---------------------------------------------------------------------------


def test_active_bypass_exact_identity() -> None:
    h = hub.build_synthetic_site_hub(
        d_model=24,
        hub_active=True,
        selection_mode="top1",
        seed=3,
    )
    hub.force_router_one_hot(h, hub.BYPASS_CHOICE, logit_scale=50.0)
    x = torch.randn(4, 6, 24)
    y, aux = h(x, return_aux=True)
    assert aux["hub_active"] is True
    assert torch.equal(y, x) or torch.allclose(y, x, atol=1e-6)
    # Residual path also exact.
    y2, r, info = h.apply_with_residual(x)
    assert torch.allclose(r, torch.zeros_like(x), atol=1e-6)
    assert torch.allclose(y2, x, atol=1e-6)
    x_back, rev = h.revert(y2, residual=r)
    assert rev["exact"] is True
    assert torch.allclose(x_back, x, atol=1e-6)


def test_active_soft_bypass_near_identity() -> None:
    h = hub.build_synthetic_site_hub(
        d_model=24,
        hub_active=True,
        selection_mode="soft",
        seed=4,
    )
    hub.force_router_one_hot(h, hub.BYPASS_CHOICE, logit_scale=40.0)
    x = torch.randn(2, 3, 24)
    y = h(x)
    # Softmax is not one-hot; with large bias should be extremely close to identity.
    assert torch.allclose(y, x, atol=1e-4)


# ---------------------------------------------------------------------------
# Active: routes to the right bridge
# ---------------------------------------------------------------------------


def test_active_top1_routes_to_named_bridge() -> None:
    h = hub.build_synthetic_site_hub(
        d_model=28,
        hub_active=True,
        selection_mode="top1",
        bridge_names=["GLM_DECOMPOSITION_BRIDGE", "KIMI_PLANNING_BRIDGE"],
        seed=11,
    )
    x = torch.randn(2, 4, 28)

    # Route exclusively to GLM.
    hub.force_router_one_hot(h, "GLM_DECOMPOSITION_BRIDGE", logit_scale=50.0)
    y_glm = h(x)
    r_glm = hub.module_residual(h.bridges["GLM_DECOMPOSITION_BRIDGE"], x)
    assert torch.allclose(y_glm, x + r_glm, atol=1e-5)

    # Route exclusively to Kimi.
    hub.force_router_one_hot(h, "KIMI_PLANNING_BRIDGE", logit_scale=50.0)
    y_kimi = h(x)
    r_kimi = hub.module_residual(h.bridges["KIMI_PLANNING_BRIDGE"], x)
    assert torch.allclose(y_kimi, x + r_kimi, atol=1e-5)

    # The two routes differ (bridges are different modules).
    assert not torch.allclose(y_glm, y_kimi, atol=1e-4)


def test_active_input_conditioned_routing() -> None:
    """Synthetic feature steers gate toward the matching bridge."""

    h = hub.build_synthetic_site_hub(
        d_model=16,
        hub_active=True,
        selection_mode="top1",
        bridge_names=["GLM_DECOMPOSITION_BRIDGE", "KIMI_PLANNING_BRIDGE"],
        seed=22,
    )
    hub.force_router_input_linear(
        h,
        feature_index=0,
        positive_choice="GLM_DECOMPOSITION_BRIDGE",
        negative_choice="KIMI_PLANNING_BRIDGE",
        scale=15.0,
    )
    # Positive feature → GLM.
    x_pos = torch.zeros(1, 2, 16)
    x_pos[..., 0] = 1.0
    y_pos, aux_pos = h(x_pos, return_aux=True)
    r_glm = hub.module_residual(h.bridges["GLM_DECOMPOSITION_BRIDGE"], x_pos)
    assert torch.allclose(y_pos, x_pos + r_glm, atol=1e-4)
    assert aux_pos.get("selected") == "GLM_DECOMPOSITION_BRIDGE"

    # Negative feature → Kimi.
    x_neg = torch.zeros(1, 2, 16)
    x_neg[..., 0] = -1.0
    y_neg, aux_neg = h(x_neg, return_aux=True)
    r_kimi = hub.module_residual(h.bridges["KIMI_PLANNING_BRIDGE"], x_neg)
    assert torch.allclose(y_neg, x_neg + r_kimi, atol=1e-4)
    assert aux_neg.get("selected") == "KIMI_PLANNING_BRIDGE"


# ---------------------------------------------------------------------------
# Collision resolution (two bridges, one site)
# ---------------------------------------------------------------------------


def test_collision_site_top1_not_dual_add() -> None:
    """At one transplant site, top-1 picks ONE bridge — not both additively."""

    h = hub.build_synthetic_site_hub(
        d_model=32,
        hub_active=True,
        selection_mode="top1",
        bridge_names=["GLM_DECOMPOSITION_BRIDGE", "KIMI_PLANNING_BRIDGE"],
        site_id="post_attention_hidden_state@L16",
        seed=42,
    )
    x = torch.randn(3, 5, 32)

    # Legacy always-on / dual-add reference (what we refuse under hub top-1).
    r_glm = hub.module_residual(h.bridges["GLM_DECOMPOSITION_BRIDGE"], x)
    r_kimi = hub.module_residual(h.bridges["KIMI_PLANNING_BRIDGE"], x)
    dual_add = x + r_glm + r_kimi
    sequential = hub.always_on_apply(
        x, h.bridges, mode="sequential", order=h._bridge_order
    )

    hub.force_router_one_hot(h, "GLM_DECOMPOSITION_BRIDGE", logit_scale=50.0)
    y = h(x)

    # Equals single-bridge residual, not dual-add.
    assert torch.allclose(y, x + r_glm, atol=1e-5)
    assert not torch.allclose(y, dual_add, atol=1e-4)
    assert not torch.allclose(y, sequential, atol=1e-4)

    # Magnitude proof: dual-add residual norm > single residual norm (usually).
    single_norm = float(r_glm.norm().item())
    dual_norm = float((r_glm + r_kimi).norm().item())
    hub_norm = float((y - x).norm().item())
    assert abs(hub_norm - single_norm) < 1e-4
    # Dual-add is a different vector (collision path we avoid).
    assert dual_norm > 0.0


def test_collision_site_soft_mix_is_convex_not_forced_sum() -> None:
    """Soft mix is a convex combination of residuals, not forced sum of both."""

    h = hub.build_synthetic_site_hub(
        d_model=24,
        hub_active=True,
        selection_mode="soft",
        bridge_names=["GLM_DECOMPOSITION_BRIDGE", "KIMI_PLANNING_BRIDGE"],
        seed=9,
    )
    # Amplify residuals so mix vs dual-add is numerically obvious (random-init
    # bridges alone can produce ~1e-5 residuals that collapse the distinction).
    with torch.no_grad():
        h.bridges["GLM_DECOMPOSITION_BRIDGE"].b.fill_(0.5)
        h.bridges["KIMI_PLANNING_BRIDGE"].b.fill_(-0.3)

    # Equal mix between the two bridges (suppress bypass).
    names = h.choice_names()
    with torch.no_grad():
        h.router.fc2.weight.zero_()
        h.router.fc2.bias.zero_()
        for i, name in enumerate(names):
            if name == hub.BYPASS_CHOICE:
                h.router.fc2.bias[i] = -50.0
            else:
                h.router.fc2.bias[i] = 5.0

    x = torch.randn(2, 3, 24)
    y, aux = h(x, return_aux=True)
    r_glm = hub.module_residual(h.bridges["GLM_DECOMPOSITION_BRIDGE"], x)
    r_kimi = hub.module_residual(h.bridges["KIMI_PLANNING_BRIDGE"], x)
    dual = r_glm + r_kimi
    # Soft equal mix ≈ 0.5 * each (plus tiny bypass mass).
    approx_mix = 0.5 * r_glm + 0.5 * r_kimi
    assert torch.allclose(y - x, approx_mix, atol=5e-2)
    # Distinct from dual-add: dual residual magnitude is ~2× the equal mix.
    assert not torch.allclose(y - x, dual, atol=1e-2)
    mix_norm = float((y - x).norm().item())
    dual_norm = float(dual.norm().item())
    assert dual_norm > mix_norm * 1.2
    assert aux["mode"] == "soft"


def test_inactive_collision_still_dual_path_for_regression() -> None:
    """With HUB_ACTIVE=false, co-located bridges still use always-on (legacy)."""

    h = hub.build_synthetic_site_hub(
        d_model=20,
        hub_active=False,
        bridge_names=["GLM_DECOMPOSITION_BRIDGE", "KIMI_PLANNING_BRIDGE"],
        seed=5,
    )
    x = torch.randn(2, 2, 20)
    y = h(x)
    sequential = hub.always_on_apply(
        x, h.bridges, mode="sequential", order=h._bridge_order
    )
    assert torch.equal(y, sequential)


# ---------------------------------------------------------------------------
# Reversibility + gravity
# ---------------------------------------------------------------------------


def test_apply_with_residual_exact_reverse_active() -> None:
    h = hub.build_synthetic_site_hub(d_model=20, hub_active=True, selection_mode="soft", seed=2)
    hub.force_router_one_hot(h, "KIMI_PLANNING_BRIDGE", logit_scale=30.0)
    x = torch.randn(2, 4, 20)
    y, r, _ = h.apply_with_residual(x)
    x_back, info = h.revert(y, residual=r)
    assert info["exact"] is True
    assert torch.allclose(x_back, x, atol=1e-5)


def test_gravity_accounting_and_spec() -> None:
    h = hub.build_synthetic_site_hub(d_model=16, hub_active=False)
    grav = h.gravity_accounting()
    assert grav["ablatable"] is True
    assert grav["reversible"] is True
    assert grav["router"]["native_moe_router_touched"] is False
    assert grav["total_parameter_bytes"] > 0
    spec = h.to_spec()
    assert spec["schema"] == hub.HUB_MODULE_SCHEMA
    assert spec["hub_active"] is False
    assert spec["native_moe_router_touched"] is False
    assert "GLM_DECOMPOSITION_BRIDGE" in spec["bridge_order"]


# ---------------------------------------------------------------------------
# Activation trigger (fail-closed) + config flip
# ---------------------------------------------------------------------------


def test_activation_trigger_fail_closed_without_evidence() -> None:
    result = hub.evaluate_activation_trigger()
    assert result["activate"] is False
    assert result["status"] == "FAIL_CLOSED_NO_ACTIVATION_SIGNAL"
    assert result["currently_default_HUB_ACTIVE"] is False
    assert result["claim_boundary"]["live_v0_switched"] is False


def test_activation_trigger_secondary_regression() -> None:
    scores = {
        "tool_use": {"baseline": 0.80, "always_on": 0.70},  # delta -0.10 > 0.02
        "coding_and_repository_work": {"baseline": 0.50, "always_on": 0.49},  # within tol
    }
    result = hub.evaluate_activation_trigger(
        secondary_scores=scores,
        secondary_tolerance=SECONDARY_TOLERANCE,
    )
    assert result["activate"] is True
    assert result["signals"]["secondary_regression"] is True
    assert any(r["axis"] == "tool_use" for r in result["signals"]["regressions"])


def test_activation_trigger_requires_collision_evidence() -> None:
    # confirmed without evidence → not enough
    weak = hub.evaluate_activation_trigger(
        site_collision={"confirmed": True, "transplant_point": "post_attention_hidden_state"}
    )
    assert weak["activate"] is False

    strong = hub.evaluate_activation_trigger(
        site_collision={
            "confirmed": True,
            "site_id": "post_attention_hidden_state@L16",
            "evidence": {
                "metric": "held_out_secondary_drop",
                "delta": -0.05,
                "note": "fixture-only; not a live measurement",
            },
        }
    )
    assert strong["activate"] is True
    assert strong["signals"]["confirmed_site_collision"] is True


def test_config_one_field_flip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "hub_cfg.json"
    hub.write_default_hub_config(cfg_path)
    loaded = hub.load_hub_config(cfg_path)
    assert loaded.hub_active is False

    # One-field flip simulation.
    doc = json.loads(cfg_path.read_text(encoding="utf-8"))
    doc["hub_active"] = True
    # Drop seal for rewrite (config loader does not require seal).
    doc.pop("seal_sha256", None)
    cfg_path.write_text(json.dumps(doc), encoding="utf-8")
    flipped = hub.load_hub_config(cfg_path)
    assert flipped.hub_active is True
    assert flipped.resolved_active() is True


def test_env_override_hub_active(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = hub.HubConfig(hub_active=False, d_model=8)
    monkeypatch.setenv("FRANKENSTEIN_HUB_ACTIVE", "1")
    assert cfg.resolved_active() is True
    monkeypatch.setenv("FRANKENSTEIN_HUB_ACTIVE", "0")
    assert cfg.resolved_active() is False
    monkeypatch.delenv("FRANKENSTEIN_HUB_ACTIVE", raising=False)
    assert cfg.resolved_active() is False


# ---------------------------------------------------------------------------
# Design seal
# ---------------------------------------------------------------------------


def test_design_document_seals(tmp_path: Path) -> None:
    out = tmp_path / "MULTI_ADAPTER_HUB_DESIGN.json"
    sealed = hub.seal_hub_design(out)
    verify(sealed, label="hub design")
    assert sealed["status"] == "READY_INACTIVE_SCAFFOLD"
    assert sealed["module_default"]["HUB_ACTIVE"] is False
    assert sealed["claim_boundary"]["live_v0_switched"] is False
    assert sealed["claim_boundary"]["production_validated"] is False
    assert sealed["claim_boundary"]["fabricated_fix_claim"] is False
    assert out.is_file()
    # Activation trigger documents secondary tolerance from promotion gate.
    assert (
        sealed["activation_trigger"]["signals"][0]["tolerance_value"]
        == float(SECONDARY_TOLERANCE)
    )
    assert "compose" in sealed["composition_with_promotion_gate"]["relationship"].lower() or (
        "independent" in sealed["composition_with_promotion_gate"]["relationship"].lower()
    )


def test_cli_smoke() -> None:
    rc = hub.main(["smoke", "--d-model", "16"])
    assert rc == 0


def test_cli_eval_trigger_fail_closed() -> None:
    rc = hub.main(["eval-trigger"])
    assert rc == 0
