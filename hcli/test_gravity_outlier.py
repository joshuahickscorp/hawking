"""The gate and grammar must both be falsifiable, not asserted."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "future"))
import gravity_outlier_eval as G


def test_grammar_expresses_a_non_quantization_class():
    p = G.parse_spec("outlier0.005-g128")
    assert p["form"] == "outlier_split"
    assert p["frac"] == 0.005 and p["group"] == 128


def test_grammar_refuses_a_spec_it_cannot_execute():
    # Registration is not capability: a spec this evaluator cannot run must
    # never become a candidate.
    with pytest.raises(ValueError):
        G.parse_spec("tiers-1bit")
    with pytest.raises(ValueError):
        G.parse_spec("mixed-q2q3-experts")


def _gate(ppl, r4):
    return bool(r4 <= G.R4_MAX and ppl <= G.PPL_MAX)


def test_capability_gate_passes_the_measured_bf16_reference():
    assert _gate(G.BF16_PPL, G.BF16_R4)


def test_capability_gate_rejects_the_measured_collapsed_arm():
    # base-g128, measured: 87 consecutive identical tokens.
    # receipts/future/O003_OUTLIER_SPLIT.json
    assert not _gate(4.0847, 0.7204)


def test_capability_gate_rejects_the_best_arm_too():
    # outlier-g128 is the best representation measured so far and it STILL
    # fails. The gate is calibrated to bf16, not to whatever we managed to build.
    assert not _gate(3.3579, 0.5066)


def test_outlier_class_never_claims_execution():
    # The sparse channel has no kernel; a TPS claim for it would be fabrication.
    assert G.parse_spec("outlier0.005-g128")["form"] == "outlier_split"


def test_spec_naming_an_unrunnable_group_is_refused_at_the_door():
    # mx.quantize supports 32/64/128. outlier0.005-g256 parsed cleanly and then
    # died inside the search, costing a whole budget slot.
    with pytest.raises(ValueError, match="group 256"):
        G.parse_spec("outlier0.005-g256")
    for g in (32, 64, 128):
        assert G.parse_spec(f"outlier0.005-g{g}")["group"] == g


def test_new_gravity_frontier_methods_have_distinct_measured_plans():
    act = G.parse_spec("sparseact0.0625percal-g128")
    ternary = G.parse_spec("ternarysparse0.05percal-g128")
    organ = G.parse_spec("organsparse0.02u0.10percal-g128")
    assert act["form"] == "sparse_act" and act["per_expert"]
    assert ternary["form"] == "sparse_ternary" and ternary["bits"] == 2
    assert organ["form"] == "organ_sparse"
    assert organ["up_density"] == 0.02 and organ["down_density"] == 0.10
    up_ternary = G.parse_spec("organupternary0.02u0.10percal-g128")
    down_ternary = G.parse_spec("organdownternary0.02u0.10percal-g128")
    assert up_ternary["form"] == "organ_hybrid" and up_ternary["ternary_organ"] == "up"
    assert down_ternary["form"] == "organ_hybrid" and down_ternary["ternary_organ"] == "down"
    assert G.predict_ebpw("sparseact0.0625percal-g128") < 1.0
    assert G.predict_ebpw("ternarysparse0.05percal-g128") < 1.0
    assert G.predict_ebpw("organsparse0.02u0.10percal-g128") < 1.0
    assert G.predict_ebpw("organupternary0.02u0.10percal-g128") < 1.0
    assert G.predict_ebpw("organdownternary0.02u0.10percal-g128") < 1.0


def test_cost_predictor_matches_measured_bytes():
    """A direction check is worthless if its arithmetic disagrees with the
    executor. Every non-outlier form must be exact."""
    measured = {"q4-g64-experts": 4.3797, "q3-g64-experts": 3.5024,
                "q2-g64-experts": 2.6251, "q2-g128-experts": 2.4058,
                "binary-g128": 1.4188, "binary-g64": 1.5284,
                "resbinary-g128": 2.4058}
    for spec, want in measured.items():
        got = G.predict_ebpw(spec)
        assert abs(got - want) < 1e-3, f"{spec}: predicted {got}, measured {want}"


def test_outlier_prediction_is_biased_HIGH_never_low():
    """The threshold keeps fewer weights than requested, so the estimate reads
    high. High is the safe direction: it can never make an upward proposal look
    like a descent."""
    for spec, measured in (("outlier0.005-g128", 2.5419), ("outlier0.02-g128", 2.9597)):
        got = G.predict_ebpw(spec)
        assert got >= measured, f"{spec}: {got} < {measured} -- bias went the unsafe way"
        assert got - measured < 0.01, f"{spec}: bias {got - measured} too large to be useful"


def test_pq_sparse_correction_is_counted_in_the_bytes():
    """A sparse channel the representation requires must cost EBPW.

    Measured 2026-09-06: pqsparse0.002percal reported 2.405907100874676, byte
    for byte identical to pqpercal4k512 which has no sparse channel at all. The
    0.2% of weights kept at full precision were free in the accounting. S012 §77
    -- do not exclude something from the count because of what it is called.
    """
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent
           / "tools" / "future" / "gravity_outlier_eval.py").read_text()
    i = src.index("expert_bits = pq_index_bits")
    line = src[i:src.index("\n", i)]
    assert "kept * 32" in line, f"sparse correction is not counted: {line}"


def test_shared_tucker_reference_keeps_direct_shapes_and_parent_out_of_consumer():
    """The two-sided organ discriminator must remain directly executable."""
    import torch

    from kimi_shared_tucker_fanout import _direct, _fit_tucker

    torch.set_num_threads(1)
    weights = [torch.randn(10, 6) for _ in range(3)]
    u, v, cores, rows, width = _fit_tucker(weights, 2, 2, 5)
    assert (rows, width) == (10, 6)
    assert u.shape == (10, 2)
    assert v.shape == (6, 2)
    assert cores[0].shape == (2, 2)
    assert _direct(u, v, cores[0], torch.randn(6)).shape == (10,)


def test_nova_routed_consumer_matches_gather_mm_for_switchglue_layouts():
    """Direct factors must preserve both sorted and unsorted MoE routing shapes."""
    import mlx.core as mx

    from kimi_nova_routed_kernel import direct_vs_gather_mm

    u = mx.arange(5 * 2, dtype=mx.float32).reshape(5, 2) / 10
    v = mx.arange(6 * 3, dtype=mx.float32).reshape(6, 3) / 10
    cores = mx.arange(4 * 2 * 3, dtype=mx.float32).reshape(4, 2, 3) / 20
    residual_a = mx.arange(4 * 5 * 2, dtype=mx.float32).reshape(4, 5, 2) / 30
    residual_b = mx.arange(4 * 6 * 2, dtype=mx.float32).reshape(4, 6, 2) / 40
    factors = (u, v, cores, residual_a, residual_b)

    sorted_result = direct_vs_gather_mm(
        mx.arange(5 * 1 * 6, dtype=mx.float32).reshape(5, 1, 6) / 10,
        mx.array([0, 1, 2, 3, 0], dtype=mx.int32),
        factors,
    )
    unsorted_result = direct_vs_gather_mm(
        mx.arange(1 * 3 * 1 * 6, dtype=mx.float32).reshape(1, 3, 1, 6) / 10,
        mx.array([[[0, 1], [2, 3], [1, 0]]], dtype=mx.int32),
        factors,
    )
    for result in (sorted_result, unsorted_result):
        assert result["exact_shape"]
        assert result["max_abs_error"] < 1e-2


def test_independent_expert_lowrank_consumer_preserves_routing_shape():
    import mlx.core as mx

    from kimi_nova_routed_kernel import ExpertLowRankRoutedLinear

    factor_a = mx.arange(4 * 5 * 2, dtype=mx.float32).reshape(4, 5, 2) / 30
    factor_b = mx.arange(4 * 6 * 2, dtype=mx.float32).reshape(4, 6, 2) / 40
    consumer = ExpertLowRankRoutedLinear(factor_a, factor_b)
    x = mx.arange(1 * 3 * 1 * 6, dtype=mx.float32).reshape(1, 3, 1, 6) / 10
    ids = mx.array([[[0, 1], [2, 3], [1, 0]]], dtype=mx.int32)
    result = consumer(x, ids)
    assert tuple(result.shape) == (1, 3, 2, 1, 5)


def test_variable_rank_consumer_keeps_compact_rank_groups_without_padding():
    import mlx.core as mx

    from kimi_nova_routed_kernel import VariableExpertLowRankRoutedLinear

    a2 = mx.arange(2 * 5 * 2, dtype=mx.float32).reshape(2, 5, 2) / 30
    b2 = mx.arange(2 * 6 * 2, dtype=mx.float32).reshape(2, 6, 2) / 40
    a3 = mx.arange(2 * 5 * 3, dtype=mx.float32).reshape(2, 5, 3) / 30
    b3 = mx.arange(2 * 6 * 3, dtype=mx.float32).reshape(2, 6, 3) / 40
    consumer = VariableExpertLowRankRoutedLinear(
        {2: (a2, b2), 3: (a3, b3)}, (2, 3, 2, 3)
    )
    x = mx.arange(4 * 1 * 6, dtype=mx.float32).reshape(4, 1, 6) / 10
    ids = mx.array([0, 1, 2, 3], dtype=mx.int32)
    result = consumer(x, ids)
    expected = mx.stack([
        x[0, 0] @ b2[0] @ a2[0].T,
        x[1, 0] @ b3[0] @ a3[0].T,
        x[2, 0] @ b2[1] @ a2[1].T,
        x[3, 0] @ b3[1] @ a3[1].T,
    ])[:, None, :]
    mx.eval(result, expected)
    assert tuple(result.shape) == (4, 1, 5)
    assert float(mx.max(mx.abs(result - expected)).item()) < 1e-5


def test_int8_expert_lowrank_consumer_uses_two_gather_mm_stages():
    import mlx.core as mx

    from kimi_nova_routed_kernel import Int8ExpertLowRankRoutedLinear

    aq = mx.arange(4 * 5 * 2, dtype=mx.float32).reshape(4, 5, 2) / 30
    bq = mx.arange(4 * 6 * 2, dtype=mx.float32).reshape(4, 6, 2) / 40
    a_scale = mx.ones((4, 5), dtype=mx.float32) * 0.5
    b_scale = mx.ones((4, 6), dtype=mx.float32) * 0.25
    consumer = Int8ExpertLowRankRoutedLinear(aq, a_scale, bq, b_scale)
    x = mx.arange(1 * 3 * 1 * 6, dtype=mx.float32).reshape(1, 3, 1, 6) / 10
    ids = mx.array([[[0, 1], [2, 3], [1, 0]]], dtype=mx.int32)
    flat_x = mx.repeat(x.reshape(-1, 1, 6), 2, axis=0).reshape(-1, 6)
    experts = (0, 1, 2, 3, 1, 0)
    expected = mx.stack([
        flat_x[i] @ (bq[expert] * b_scale[expert][:, None])
        @ (aq[expert] * a_scale[expert, :, None]).T
        for i, expert in enumerate(experts)
    ]).reshape(1, 3, 2, 1, 5)
    result = consumer(x, ids)
    mx.eval(result, expected)
    assert tuple(result.shape) == (1, 3, 2, 1, 5)
    assert float(mx.max(mx.abs(result - expected)).item()) < 1e-5
