"""G023 step-1 pins."""
import json
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
R = RH / "MOE_PACK_SLICE.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="slice not run")


def rec():
    return json.load(open(R))


def test_it_tested_the_representation_the_planner_actually_selected():
    """Not a fresh choice: the KernelPlanner downgraded to this one."""
    d = rec()["representation_under_test"]
    kp = json.load(open(RH / "KERNEL_PLANNER_MODEL2.json"))
    sel = next(r for r in kp["organ_plan"] if r["organ"] == "moe_expert")
    assert d["family"] == sel["selected_representation"]
    assert d["downgraded_from"] == "q2_affine"


def test_real_expert_tensors_were_packed_not_synthetic():
    d = rec()["result"]
    assert d["n_tensors_packed"] >= 12
    for t in d["tensors"]:
        assert "mlp.experts." in t["tensor"]
        assert t["shape"] and t["source_bytes"] > 0


def test_the_codec_reconstructs_within_a_sane_bound():
    d = rec()["result"]
    assert d["median_cosine"] > 0.98
    assert d["worst_cosine"] > 0.98
    assert d["median_rel_fro_err"] < 0.2


def test_the_rate_is_what_uniform_q4_group64_should_cost():
    """4 bits per weight plus one f16 scale per 64 -> 4.25 bpw."""
    d = rec()["result"]
    assert abs(d["median_bpw"] - 4.25) < 0.01
    assert d["total_packed_bytes"] < d["total_source_bytes"]


def test_the_composition_law_is_stated_not_buried():
    """A per-tensor cosine is necessary, never sufficient."""
    d = rec()["LOCAL_ADEQUACY_DOES_NOT_COMPOSE"]
    assert "never a sufficient one" in d["law"]
    assert len(d["evidence_from_this_campaign"]) >= 2
    assert "does NOT mean" in d["so_this_result_means"]


def test_the_scope_is_declared_as_a_slice():
    d = rec()["scope"]
    assert d["fraction_of_the_organ"] < 0.01
    assert d["no_catalog_no_container"] is True


def test_the_packer_gap_is_pinned_to_a_specific_resolver_line():
    d = rec()["packer_gap_this_probes"]
    assert "organ_role" in d["resolver"]
    assert "mlp.experts.101.gate_proj.weight" in d["why_experts_fall_through"]
    assert len(d["what_a_real_packer_needs"]) >= 4


def test_the_resolver_gap_this_slice_pinned_is_now_fixed():
    """The slice pinned the defect; the resolver has since been extended. The dense-MLP
    and router paths must be unchanged by that fix."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "headless"))
    import whole_model_native as w
    assert w.organ_role("model.layers.0.mlp.experts.101.gate_proj.weight") == "moe_expert"
    assert w.organ_role("model.layers.0.mlp.gate_proj.weight") == "mlp"
    # the router still falls through to leftover: the planner chose f32 passthrough
    assert w.organ_role("model.layers.0.mlp.gate.weight") == "leftover"
