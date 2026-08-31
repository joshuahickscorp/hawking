import json

import numpy as np
import pytest

from tools.future import router_science as rs
from tools.future._common import RECEIPTS


def test_build_emits_sealed_receipt():
    out = rs.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "ROUTER_SENSITIVE_ALLOCATION.json"
    assert doc["schema"] == "hawking.future.router_science.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["promotion_allowed"] is False
    assert "headline_question" in doc
    assert "WHICH BITS EXIST PRIMARILY TO PRESERVE FUTURE CONTROL FLOW" in doc["headline_question"]
    assert doc["headline_answer"]["bits"]
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["precision_allocation"]
    assert doc["router_jacobian_approximation"]["from_sensitivity_map"]["shape"] == [512, 2560]
    assert doc["bench"]["state"] != "MEASURED"


def test_selftest_emits_sealed_receipt():
    out = rs.selftest()
    doc = json.loads(out.read_text())
    assert doc["schema"] == rs.SCHEMA
    assert doc["seal_sha256"]
    assert doc["router_jacobian_approximation"]["synthetic"]["matches_analytical"] is True


def test_logit_jacobian_fd_matches_analytical():
    rng = np.random.default_rng(rs.SYNTHETIC_SEED)
    W = rng.normal(size=(8, 16))
    x = rng.normal(size=(16,))
    fd = rs.finite_difference_logit_jacobian(W, x, eps=1e-6)
    assert fd.shape == W.shape
    assert float(np.max(np.abs(fd - W))) < 1e-8


def test_routing_inert_direction_has_zero_action():
    rng = np.random.default_rng(rs.SYNTHETIC_SEED)
    W = rng.normal(size=(4, 12))
    _u, _s, vt = np.linalg.svd(W, full_matrices=True)
    inert = vt[4:]
    action = W @ inert.T
    assert float(np.max(np.abs(action))) < 1e-12
    visible = vt[0]
    assert float(np.linalg.norm(W @ visible)) == pytest.approx(float(_s[0]), rel=1e-12)


def test_jacobian_validity_range_stated_from_map():
    doc, _prov = rs.load_sensitivity_map()
    jac = rs.jacobian_from_map(doc)
    assert jac["jacobian_matrix"] is None
    assert jac["shape"] == [512, 2560]
    assert jac["validity_range"]["logit_jacobian"]["range"]
    assert jac["validity_range"]["topk_membership"]["global_linf_overpredicts_flips"] is True
    assert jac["below_operator_norm_bound"] is True
    assert jac["observed_operator_norm_proxy"] < jac["sigma_max"]


def test_softmax_jacobian_fd_matches_analytical():
    rng = np.random.default_rng(rs.SYNTHETIC_SEED)
    z = rng.normal(size=(10,))
    out = rs.softmax_prob_jacobian(z, eps=1e-6)
    assert out["matches_analytical"] is True
    assert "not a top-k" in out["validity_range"].lower()


def test_route_margin_is_kth_minus_kplus1():
    logits = np.array(
        [
            [0.0, 5.0, 1.0, 4.0, 3.0, 2.0],
            [9.0, 8.0, 0.1, 0.0, 7.0, 6.5],
        ]
    )
    # k=2: first row sorted desc 5,4,3,2,1,0 → 4-3=1; second 9,8,7,6.5,0.1,0 → 8-7=1
    m = rs.route_margins(logits, k=2)
    assert list(m) == pytest.approx([1.0, 1.0])
    m3 = rs.route_margins(logits, k=3)
    assert list(m3) == pytest.approx([1.0, 0.5])


def test_boundary_fraction_on_recovered_margins():
    doc, _ = rs.load_sensitivity_map()
    margins = doc["routing"]["dense_top10_top11_margin"]
    table = rs.topk_boundary_sensitivity(margins)
    by = {row["epsilon"]: row for row in table["by_epsilon"]}
    assert by[1e-5]["n_within"] == 1
    assert by[1e-5]["fraction"] == 0.25
    assert by[1e-4]["n_within"] == 2
    assert by[1e-3]["n_within"] == 3
    assert by[1e-2]["n_within"] == 4


def test_recovered_map_flip_is_the_tightest_margin():
    doc, _ = rs.load_sensitivity_map()
    hold = rs.flipped_row_is_min_margin(doc)
    assert hold["holds"] is True
    assert hold["flipped_positions"] == [1]
    assert hold["min_margin_position"] == 1


def test_critical_directions_rank512_repairs_and_coordinates_do_not():
    doc, _ = rs.load_sensitivity_map()
    crit = rs.critical_hidden_directions(doc)
    assert crit["router_visible_rank_upper_bound"] == 512
    assert crit["routing_inert_dim_lower_bound"] == 2048
    assert crit["smallest_visible_rank_that_repaired_membership"] == 512
    assert crit["coordinate_salience_did_not_repair_at_any_tested_fraction"] is True


def test_per_dimension_formula_matches_map_definition():
    # Map: sum_{n,e} |delta[n,j] * W[e,j]| = sum_n |delta[n,j]| * sum_e |W[e,j]|
    rng = np.random.default_rng(0)
    W = rng.normal(size=(5, 7))
    delta = rng.normal(size=(3, 7))
    got = rs.per_dimension_sensitivity(W, delta)
    ref = np.abs(delta[:, :, None] * W.T[None, :, :]).sum(axis=(0, 2))
    assert np.allclose(got, ref)


def test_residual_budget_prefers_high_route_risk():
    margins = [1e-8, 1e-2]
    budget = rs.residual_budget_by_route_risk(margins, flips=[1, 0])
    tight, wide = budget["positions"]
    assert tight["residual_hidden_fraction"] > wide["residual_hidden_fraction"]
    assert tight["residual_hidden_fraction"] == rs.TIGHT_RESIDUAL_FRACTION
    assert wide["residual_hidden_fraction"] == rs.BASE_RESIDUAL_FRACTION
    assert tight["route_risk"] > wide["route_risk"]
    assert tight["basis"] == "router_visible_right_singular_subspace"


def test_allocator_gives_more_precision_to_tiny_margin_than_large():
    """Negative control of the recommendation: margin, not size, not position."""
    tiny = rs.Surface(
        name="tiny",
        kind="position_residual",
        margin_min=1e-8,
        n_params=2560,
        hidden=2560,
        positions=1,
        position_index=0,
        k=10,
    )
    large = rs.Surface(
        name="large",
        kind="position_residual",
        margin_min=1.0,
        n_params=2560,
        hidden=2560,
        positions=1,
        position_index=0,
        k=10,
    )
    a_tiny = rs.assign_bit_class(tiny)
    a_large = rs.assign_bit_class(large)
    assert rs.BIT_CLASS_RANK[a_tiny["bit_class"]] > rs.BIT_CLASS_RANK[a_large["bit_class"]]
    assert a_tiny["recommended_storage_bpw"] > a_large["recommended_storage_bpw"]
    assert a_tiny["residual_budget_fraction"] > a_large["residual_budget_fraction"]
    assert rs.precision_rank(a_tiny) > rs.precision_rank(a_large)
    assert a_large["bit_class"] == "CRUSHED"
    assert a_tiny["bit_class"] == "PREMIUM"


def test_allocator_not_driven_by_size_or_position():
    a = rs.assign_bit_class(
        rs.Surface(
            name="small-late",
            kind="position_residual",
            margin_min=1e-4,
            n_params=3,
            hidden=3,
            positions=1,
            position_index=99,
            k=10,
        )
    )
    b = rs.assign_bit_class(
        rs.Surface(
            name="huge-early",
            kind="position_residual",
            margin_min=1e-4,
            n_params=10_000_000,
            hidden=4096,
            positions=64,
            position_index=0,
            k=10,
        )
    )
    assert a["bit_class"] == b["bit_class"] == "ORDINARY"
    assert a["residual_budget_fraction"] == b["residual_budget_fraction"]
    assert a["recommended_storage_bpw"] == b["recommended_storage_bpw"]


def test_crush_refusal_actually_fires_on_tiny_margin():
    """A guard nobody has watched fail is not a guard."""
    surface = rs.Surface(
        name="tight-control-flow",
        kind="position_residual",
        margin_min=1e-8,
        n_params=16,
        hidden=16,
        positions=1,
        position_index=0,
        k=10,
    )
    assert rs.classify_surface(surface) != "CRUSHED"
    with pytest.raises(rs.ControlFlowCrushError) as ei:
        rs.assign_bit_class(surface, requested="CRUSHED")
    msg = str(ei.value).lower()
    assert "crush" in msg
    assert "tight-control-flow" in msg
    # Natural assignment still succeeds and is premium.
    natural = rs.assign_bit_class(surface)
    assert natural["bit_class"] == "PREMIUM"


def test_crush_allowed_on_large_margin():
    surface = rs.Surface(
        name="wide",
        kind="position_residual",
        margin_min=1.0,
        n_params=16,
        hidden=16,
        positions=1,
        position_index=0,
        k=10,
    )
    a = rs.assign_bit_class(surface, requested="CRUSHED")
    assert a["bit_class"] == "CRUSHED"


def test_crush_refusal_fires_on_router_weight():
    surface = rs.Surface(
        name="flash.moe.gate.weight.family",
        kind="router_weight",
        n_params=512 * 2560,
        hidden=2560,
        k=10,
    )
    with pytest.raises(rs.ControlFlowCrushError):
        rs.assign_bit_class(surface, requested="CRUSHED")
    a = rs.assign_bit_class(surface)
    assert a["bit_class"] == "CONTROL_FLOW_PREMIUM"
    assert a["recommended_storage_bpw"] == 16.0


def test_inert_complement_is_crushed_and_visible_is_not():
    visible = rs.Surface(name="vis", kind="hidden_visible", n_params=100, hidden=16)
    inert = rs.Surface(name="inert", kind="hidden_inert", n_params=100, hidden=16)
    assert rs.assign_bit_class(visible)["bit_class"] == "PREMIUM"
    assert rs.assign_bit_class(inert)["bit_class"] == "CRUSHED"
    with pytest.raises(rs.ControlFlowCrushError):
        rs.assign_bit_class(visible, requested="CRUSHED")
    assert rs.assign_bit_class(inert, requested="CRUSHED")["bit_class"] == "CRUSHED"


def test_map_surfaces_allocate_premium_to_the_flipped_position():
    doc, _ = rs.load_sensitivity_map()
    rows = {a["name"]: a for a in rs.allocate_precision(rs.surfaces_from_map(doc))}
    assert rows["flash.l3_l4.position.1"]["bit_class"] == "PREMIUM"
    assert rows["flash.l3_l4.position.1"]["membership_flip"] is True
    assert rows["flash.l3_l4.position.3"]["bit_class"] == "CRUSHED"
    assert rows["flash.moe.gate.weight.family"]["bit_class"] == "CONTROL_FLOW_PREMIUM"
    assert rows["flash.l3_state.router_inert_complement"]["bit_class"] == "CRUSHED"
    assert (
        rows["flash.l3_l4.position.1"]["recommended_storage_bpw"]
        > rows["flash.l3_l4.position.3"]["recommended_storage_bpw"]
    )


def test_receipt_does_not_claim_hardware_fields():
    from tools.future._common import HARDWARE_FIELDS

    out = rs.build()
    doc = json.loads(out.read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                assert k not in HARDWARE_FIELDS or not isinstance(v, (int, float)), here
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)
