"""GLOBAL_ALLOCATOR: equal-byte comparison, scales counted, 0.01*W rejected."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from global_allocator import (  # noqa: E402
    CITED_FLOOR_BPW,
    F16_BPW,
    LEADER_EBPW,
    RECEIPT,
    SCALE_TRAP,
    SCHEMA,
    SOURCE_PARAM_COUNT,
    bill_island,
    damage_of,
    greedy_descend,
    grouped_bpw,
    implied_floor_ebpw,
    interpolate_level,
    item_weight,
    non_mlp_floor_ebpw,
    q4_healthy,
    q_inject,
    q_mult,
    score_pair,
    storage_bytes_of,
    summarize_alloc,
    test_greedy_beats_uniform_on_heterogeneous_rd,
    test_grouped_bpw_counts_scales,
    test_implied_floor_arithmetic,
    test_island_billing_counts_ids_and_f16,
    test_metric_rejects_001W_and_cosine_does_not,
    test_q_inject_spans_depth,
    uniform_assignment,
)


def test_accounting_not_naive_bits():
    test_grouped_bpw_counts_scales()
    test_island_billing_counts_ids_and_f16()
    assert grouped_bpw(4, 64) == pytest.approx(4.25)
    assert grouped_bpw(4, 128) == pytest.approx(4.125)
    assert grouped_bpw(2, 64) == pytest.approx(2.25)


def test_implied_floor_matches_the_obligation_arithmetic():
    test_implied_floor_arithmetic()
    ebpw = implied_floor_ebpw()
    assert ebpw == pytest.approx(2.9398, abs=5e-5)
    assert non_mlp_floor_ebpw() == pytest.approx(1.5082, abs=5e-5)
    # Leader sits only ~6% above the implied floor.
    assert 0.05 < (LEADER_EBPW / ebpw - 1.0) < 0.08


def test_ws_scale_trap_and_zero_are_not_go():
    test_metric_rejects_001W_and_cosine_does_not()
    rng = np.random.RandomState(2)
    Y = rng.randn(16, 32).astype(np.float32)
    z = score_pair(Y, np.zeros_like(Y))
    assert z["cosine"] == pytest.approx(0.0)
    assert z["rel_fro"] == pytest.approx(1.0)
    ok, _ = q4_healthy(z)
    assert ok is False
    assert damage_of(z) == pytest.approx(1.0)


def test_depth_and_organ_weights_are_heterogeneous():
    test_q_inject_spans_depth()
    assert q_inject(63) / q_inject(0) == pytest.approx(2.577e-03 / 1.597e-04, rel=1e-6)
    # Late GQA element is worth more than an early MLP element.
    w_gqa = item_weight(1000, "gqa", 63)
    w_mlp = item_weight(1000, "mlp", 0)
    assert w_gqa > w_mlp


def test_greedy_beats_uniform_on_a_toy_rd():
    test_greedy_beats_uniform_on_heterogeneous_rd()


def test_interpolate_hits_equal_bytes_exactly():
    elems = 1024
    levels = [
        {
            "name": "cheap",
            "family": "toy",
            "storage_bpw": 2.0,
            "active_fused_bpw": 2.0,
            "storage_bytes": storage_bytes_of(elems, 2.0),
            "active_bytes": storage_bytes_of(elems, 2.0),
            "scale_aware": 0.5,
            "damage": 0.5,
            "rel_fro": 0.5,
            "cosine": 0.6,
            "gain": 0.5,
            "null": 0.2,
            "n_weights": elems,
            "island": False,
            "control": False,
        },
        {
            "name": "rich",
            "family": "toy",
            "storage_bpw": 4.0,
            "active_fused_bpw": 4.0,
            "storage_bytes": storage_bytes_of(elems, 4.0),
            "active_bytes": storage_bytes_of(elems, 4.0),
            "scale_aware": 0.9,
            "damage": 0.1,
            "rel_fro": 0.1,
            "cosine": 0.95,
            "gain": 0.9,
            "null": 0.2,
            "n_weights": elems,
            "island": False,
            "control": False,
        },
    ]
    mid = interpolate_level(levels, 3.0)
    assert mid["storage_bpw"] == pytest.approx(3.0)
    assert mid["scale_aware"] == pytest.approx(0.7)
    assert mid["storage_bytes"] == pytest.approx(storage_bytes_of(elems, 3.0))


def test_expensive_island_is_identifiable():
    """An island that barely moves scale_aware at a large byte cost is not a win."""
    acc = bill_island(64 * 64, 4 * 64, cheap_bpw=1.85, n_index=4)
    cheap_bytes = storage_bytes_of(64 * 64, 1.85)
    d_bytes = acc["storage_bits"] / 8.0 - cheap_bytes
    d_sa = 1e-6  # almost nothing
    marg = d_sa / d_bytes
    dense_floor = 1e-4
    assert marg < dense_floor
    assert acc["scales_counted"] is True
    assert acc["active_cached_f16_bpw"] == F16_BPW


def _receipt() -> dict:
    assert RECEIPT.is_file(), f"missing {RECEIPT} — run python3 tools/headless/global_allocator.py"
    return json.loads(RECEIPT.read_text())


def test_receipt_schema_and_discipline():
    doc = _receipt()
    assert doc["schema"] == SCHEMA
    assert doc["did_not_load_second_27b"] is True
    assert doc["capture"]["not_gaussian"] is True
    assert doc["accounting_rules"]["scales_counted"] is True
    assert doc["verdict"]["scales_counted"] is True
    assert doc["verdict"]["mlp_floor_not_transferred_as_a_prior"] is True
    trap = doc["scale_trap"]
    assert trap["rejects_scaled_artifact"] is True
    assert trap["scaled_0p01"]["cosine"] > 0.99
    assert trap["scaled_0p01"]["gain"] < 0.05
    assert trap["scaled_0p01"]["rel_fro"] > 0.9
    assert "null" in trap


def test_receipt_reports_storage_and_active_and_nulls():
    doc = _receipt()
    thr = doc["throughput"]
    assert thr["global_storage_bytes"] > 0
    assert thr["global_active_bytes_per_token"] > 0
    assert thr["global_storage_ebpw"] > 0
    assert thr["global_active_ebpw"] > 0
    assert thr["status"] == "DERIVED"
    assert thr["null"]
    for o, rec in doc["marginal_per_organ"].items():
        assert rec["storage_bpw"] is not None, o
        assert rec["active_fused_bpw"] is not None, o
        assert rec["null"], o
        assert rec["weighted_scale_aware"] is not None, o


def test_global_beats_uniform_at_equal_bytes():
    doc = _receipt()
    cmp_ = doc["comparison"]
    assert cmp_["equal_total_model_specific_bytes"] is True
    assert cmp_["relative_byte_gap"] < 1e-6
    assert cmp_["global_beats_uniform"] is True
    assert doc["verdict"]["global_beats_uniform_at_equal_bytes"] is True
    g = cmp_["uniform"]["summary"]["storage_bytes"]
    u = doc["allocation"]["greedy"]["summary"]["storage_bytes"]
    assert g == pytest.approx(u, rel=1e-9)
    assert (
        doc["allocation"]["greedy"]["summary"]["weighted_scale_aware"]
        > cmp_["uniform"]["summary"]["weighted_scale_aware"]
    )


def test_allocation_is_not_a_rounding_of_uniform():
    doc = _receipt()
    sm = doc["allocation"]["greedy"]["summary"]
    assert sm["min_storage_bpw"] is not None
    assert sm["max_storage_bpw"] is not None
    assert sm["max_storage_bpw"] - sm["min_storage_bpw"] > 0.5
    # A layer may be far from another; that is the point.
    organs = set(sm["organs"])
    assert {"mlp", "deltanet", "gqa"} <= organs
    assert "embedding" in organs or "embedding_output" in organs
    # MLP must not have been forced onto the mixer floors, or vice versa.
    mlp_bpw = sm["organs"]["mlp"]["storage_bpw"]
    gqa_bpw = sm["organs"]["gqa"]["storage_bpw"]
    assert mlp_bpw != pytest.approx(gqa_bpw, abs=0.05)


def test_protected_islands_are_measured_and_scored():
    doc = _receipt()
    isl = doc["allocation"]["protected_islands"]
    assert isl["frac"] == pytest.approx(0.01)
    measured = isl["measured"]
    assert len(measured) >= 8, "islands must be fitted on real tensors, not declared"
    for row in measured[:8]:
        assert "marginal_capability_per_byte" in row
        assert "scale_aware" in row
        assert "storage_bpw" in row
    # buys_enough is the island-specific marginal vs the median dense step.
    # Probe-layer lerp can move scale_aware by a hair relative to the cheap
    # base; the decision variable is the marginal, not the interpolated pair.
    for row in isl["selected"]:
        assert "marginal_capability_per_byte" in row
        assert "buys_enough" in row
        if row["buys_enough"]:
            assert (row["marginal_capability_per_byte"] or 0) > 0
        else:
            assert row["reason"]


def test_every_probe_level_states_a_null():
    doc = _receipt()
    n = 0
    for cls, by_l in doc["probes"].items():
        for L, rec in by_l.items():
            for lv in rec["levels"]:
                n += 1
                assert "null" in lv, f"{cls} L{L} {lv.get('name')}"
                assert "scale_aware" in lv
                assert "gain" in lv
                assert lv.get("scales_counted", True) is True
                if lv.get("name") == "scale_001W":
                    assert lv["gain"] < 0.05
                    assert not lv.get("q4_equivalent", True)
    assert n >= 40


def test_capability_and_throughput_marginals_exist():
    doc = _receipt()
    found_cap = found_thr = False
    for cls, by_l in doc["probes"].items():
        for rec in by_l.values():
            for m in rec.get("marginals") or []:
                if m.get("marginal_capability_per_model_specific_byte") is not None:
                    found_cap = True
                if m.get("marginal_tok_s_per_active_byte") is not None:
                    found_thr = True
    assert found_cap
    assert found_thr
    # More active bytes must not be reported as a throughput gain.
    model = doc["throughput"]["model"]
    assert model["d_tok_s_per_active_byte"] < 0
    assert model["null"]


def test_q_inject_and_function_lost_are_cited_not_invented():
    doc = _receipt()
    q = doc["quality"]
    assert "gravity_error_chain" in q["q_inject_source"] or "Q_INJECT" in q["q_inject_source"]
    assert "NOETIC_ORGAN_CENSUS" in q["function_lost_source"]
    assert doc["verdict"]["implied_floor_ebpw"] == pytest.approx(implied_floor_ebpw(), rel=1e-9)
    assert abs(doc["throughput"]["global_storage_ebpw"] * SOURCE_PARAM_COUNT / 8.0
               - doc["throughput"]["global_storage_bytes"]) < 8.0
