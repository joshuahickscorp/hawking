#!/usr/bin/env python3.12
"""Integration tests: real B_LINEAR_SUBSPACE_INITIALIZATION → MultiAdapterHub.

Loads the actual BEST_BALANCED.pt checkpoint, builds the hub active via
config/env (never flipping module HUB_ACTIVE), and asserts:

  - forward runs without exception
  - no NaN/Inf
  - output shape matches input
  - active + forced bypass is identity (plain DSV4F path on captured hiddens)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import frankenstein_adapter_hub as hub  # noqa: E402
from lab.operators import frankenstein_b_linear_hub_wire as wire  # noqa: E402

CKPT = wire.DEFAULT_CKPT
EXPORT = wire.DEFAULT_DSV4F_EXPORT


@pytest.fixture(scope="module")
def loaded() -> wire.LoadedBLinear:
    if not CKPT.is_file():
        pytest.skip(f"checkpoint missing: {CKPT}")
    return wire.load_b_linear_checkpoint(CKPT)


def test_module_hub_active_default_still_false() -> None:
    assert hub.HUB_ACTIVE is False


def test_load_real_best_balanced(loaded: wire.LoadedBLinear) -> None:
    assert loaded.load_info["missing_keys"] == []
    assert loaded.load_info["capability_claim"] is False
    assert loaded.stack.linear_init is not None
    assert loaded.stack.use_linear_init is True
    assert loaded.stack.teacher_projector is None
    assert loaded.stack.student_observer is None
    assert len(loaded.stack.interventions) == 0
    # low-rank residual geometry
    assert loaded.stack.linear_init.d_model == 4096
    assert loaded.stack.linear_init.rank == 16


def test_hub_active_via_env_not_module_default(loaded: wire.LoadedBLinear) -> None:
    prev = os.environ.get("FRANKENSTEIN_HUB_ACTIVE")
    try:
        active, prov = wire.resolve_hub_active_via_env_or_config(env_value="1")
        assert active is True
        assert prov["module_default_HUB_ACTIVE"] is False
        assert hub.HUB_ACTIVE is False
        h = wire.build_master_hub(loaded.stack.linear_init, hub_active=True)
        assert h.is_active is True
    finally:
        if prev is None:
            os.environ.pop("FRANKENSTEIN_HUB_ACTIVE", None)
        else:
            os.environ["FRANKENSTEIN_HUB_ACTIVE"] = prev


def test_bypass_identity_and_bridge_on_finite(loaded: wire.LoadedBLinear) -> None:
    prev = os.environ.get("FRANKENSTEIN_HUB_ACTIVE")
    try:
        os.environ["FRANKENSTEIN_HUB_ACTIVE"] = "1"
        h = wire.build_master_hub(loaded.stack.linear_init, hub_active=True)
        x = torch.randn(5, 4096)
        bypass = wire.bypass_identity_check(h, x)
        assert bypass["pass"] is True
        assert bypass["max_abs_diff_vs_input"] == 0.0
        assert bypass["output_stats"]["finite"] is True
        assert bypass["output_stats"]["shape"] == [5, 4096]

        on = wire.bridge_on_forward(h, x)
        assert on["output_stats"]["finite"] is True
        assert on["output_stats"]["shape"] == [5, 4096]
        assert on["residual_stats"]["finite"] is True
        assert on["matches_module_residual"] is True
        assert on["y_equals_x_plus_r"] is True
        # Residual should actually move the representation for nonzero weights
        assert on["differs_from_input"] is True
    finally:
        if prev is None:
            os.environ.pop("FRANKENSTEIN_HUB_ACTIVE", None)
        else:
            os.environ["FRANKENSTEIN_HUB_ACTIVE"] = prev


def test_site_hubs_eight_named_sites(loaded: wire.LoadedBLinear) -> None:
    prev = os.environ.get("FRANKENSTEIN_HUB_ACTIVE")
    try:
        os.environ["FRANKENSTEIN_HUB_ACTIVE"] = "1"
        sites = wire.build_site_hubs(loaded.stack.linear_init, hub_active=True)
        assert set(sites.keys()) == set(wire.V0_BRIDGE_SITES)
        assert len(sites) == 8
        x = torch.randn(2, 4096)
        for name, sh in sites.items():
            assert name in sh.choice_names()
            b = wire.bypass_identity_check(sh, x)
            assert b["pass"] is True, name
            on = wire.bridge_on_forward(sh, x, bridge_name=name)
            assert on["output_stats"]["finite"] is True
            assert on["output_stats"]["shape"] == [2, 4096]
    finally:
        if prev is None:
            os.environ.pop("FRANKENSTEIN_HUB_ACTIVE", None)
        else:
            os.environ["FRANKENSTEIN_HUB_ACTIVE"] = prev


@pytest.mark.skipif(not EXPORT.is_dir(), reason="DSV4F fullseq export missing")
def test_offline_smoke_real_activations(loaded: wire.LoadedBLinear) -> None:
    smoke = wire.run_offline_smoke(loaded, export_dir=EXPORT, n_prompts=5)
    assert smoke["capability_claim"] is False
    assert smoke["fabricated"] is False
    assert smoke["module_HUB_ACTIVE_unchanged"] is True
    assert smoke["health"]["no_exception"] is True
    assert smoke["health"]["all_finite"] is True
    assert smoke["health"]["bypass_identity_pass"] is True
    assert smoke["health"]["norms_bounded"] is True
    assert smoke["n_prompts"] == 5
    assert len(smoke["prompts"]) == 5
    for row in smoke["prompts"]:
        assert row["prompt_text"]  # real prompts resolved
        assert row["comparison"]["bridge_off_stats"]["finite"] is True
        assert row["comparison"]["bridge_on_stats"]["finite"] is True
        assert len(row["per_site"]) == 8
