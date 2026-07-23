#!/usr/bin/env python3.12
"""Tests for the complete MoE layer executor.

Everything here runs with no Metal device and touches no shard except the two tests that
say so; those skip when the device or the artifacts are unavailable rather than failing a
machine that has neither.  The properties under test are the ones that would let the module
publish a number it did not earn: the router claim, the activation label, the cache-key
discipline, the graph ledger, and whole-layer parity through the nonlinearity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gravity_forge as forge                    # noqa: E402
import gravity_metal_lab_b as labb               # noqa: E402
import gravity_moe_layer as gml                  # noqa: E402
import gravity_real_fixtures as grf              # noqa: E402

GEOM = gml.LayerGeometry(rows_a=2048, cols_a=6144, rows_b=6144, cols_b=2048, D=8, k=128)
SMALL = gml.LayerGeometry(rows_a=64, cols_a=128, rows_b=128, cols_b=64, D=8, k=128)


# ------------------------------------------------------------------- the router claim

def test_router_status_says_absent_everywhere():
    assert gml.ROUTER_STATUS == "ROUTER_ABSENT_FIXED_EXPERT_LIST"
    plan = gml.layer_plan(GEOM)
    assert plan["router_status"] == gml.ROUTER_STATUS
    assert "FIXED LIST" in plan["router_note"]


def test_routing_weights_are_a_fixed_normalised_vector_plus_a_shared_one():
    w = gml.routing_weights(8)
    assert len(w) == 9
    assert abs(float(w[:8].sum()) - 1.0) < 1e-6
    assert w[8] == gml.SHARED_EXPERT_WEIGHT == 1.0
    assert np.array_equal(w, gml.routing_weights(8))            # fixed, not sampled per call
    assert len(set(w[:8].tolist())) == 8                        # non-degenerate


def test_routing_weights_are_not_all_ones():
    """All-ones would hide a per-expert offset bug in the combine kernel."""
    w = gml.routing_weights(8)
    assert not np.allclose(w[:8], 1.0)


# ------------------------------------------------------------------------ the graph shape

def test_one_command_buffer_and_the_dispatch_count_is_the_graph():
    plan = gml.layer_plan(GEOM)
    assert plan["command_buffers_per_layer"] == 1
    assert plan["encoders_per_layer"] == 4
    assert plan["dispatches_by_stage"] == {"wave_a": 18, "swiglu": 9, "wave_b": 9,
                                           "combine": 1}
    assert plan["dispatches_per_layer"] == 37


def test_command_buffer_count_meets_the_moonshot_target():
    plan = gml.layer_plan(GEOM)
    targets = dict(gml.COMMAND_BUFFER_TARGETS)
    assert plan["command_buffers_per_layer"] <= targets["moonshot"]


def test_shared_expert_is_a_slot_not_a_second_graph():
    with_shared = gml.layer_plan(GEOM)
    without = gml.layer_plan(GEOM, shared=False)
    assert with_shared["tensors_executed"] - without["tensors_executed"] == 3
    assert with_shared["command_buffers_per_layer"] == without["command_buffers_per_layer"]
    assert with_shared["dispatches_per_layer"] - without["dispatches_per_layer"] == 4
    assert "routing weight 1.0" in with_shared["shared_expert_integration"]


def test_scratch_fits_the_measured_threadgroup_limit():
    plan = gml.layer_plan(GEOM)
    for value in plan["scratch_bytes"].values():
        assert value <= gml.gravity_metal.DEFAULT_THREADGROUP_MEMORY


def test_plan_refuses_a_config_that_overflows_threadgroup_memory():
    with pytest.raises(labb.TrackBError):
        gml.layer_plan(GEOM, wave_a_cfg={"grammar": "lookup-linear", "cbs": 512,
                                         "tpg": 1024, "half_table": False, "row4": True})


# -------------------------------------------------------------------------- byte ledger

def test_graph_ledger_does_not_bill_the_two_stores_the_graph_never_makes():
    """Wave A and Wave B never write a y: reduce_swiglu and moe_combine consume partials.

    A sum of 27 per-tensor ledgers would bill 27 output stores and 27 partial re-reads that
    this graph does not perform, so the ledger is built for the graph.
    """
    plan = gml.layer_plan(GEOM)
    led = plan["byte_ledger"]
    ca = plan["wave_a"]["per_tensor_cost"]
    naive = 18 * ca["executed_total_bytes"]
    assert led["wave_a_read_bytes"] + led["wave_a_write_bytes"] < naive


def test_ledger_totals_are_self_consistent():
    led = gml.layer_plan(GEOM)["byte_ledger"]
    reads = (led["wave_a_read_bytes"] + led["reduce_swiglu_read_bytes"]
             + led["wave_b_read_bytes"] + led["combine_read_bytes"])
    writes = (led["wave_a_write_bytes"] + led["reduce_swiglu_write_bytes"]
              + led["wave_b_write_bytes"] + led["combine_write_bytes"])
    assert reads == led["executed_read_bytes"]
    assert writes == led["executed_write_bytes"]
    assert reads + writes == led["executed_total_bytes"]


def test_artifact_bytes_use_the_seven_bit_packed_width():
    """The shards store 7-bit indices; the kernel uploads 8.  Both are billed, separately."""
    led = gml.layer_plan(GEOM)["byte_ledger"]
    per_tensor = (2048 * 768 * 7 + 7) // 8 + 128 * 8 * 2
    assert led["layer_artifact_bytes"] == 9 * (2 * per_tensor
                                               + ((6144 * 256 * 7 + 7) // 8 + 128 * 8 * 2))
    assert led["executed_over_artifact"] > 1.0                  # 8-bit upload, plus staging


def test_layer_moves_far_fewer_bytes_than_dense_bf16():
    led = gml.layer_plan(GEOM)["byte_ledger"]
    assert led["executed_over_dense_bf16"] < 0.2


# ----------------------------------------------------------------------- pressure targets

def test_pressure_verdict_picks_the_tightest_target_and_does_not_round():
    assert gml.pressure_verdict(0.5)["reached"] == "moonshot"
    assert gml.pressure_verdict(0.76)["reached"] == "dominance"
    assert gml.pressure_verdict(1.51)["reached"] == "ship"
    assert gml.pressure_verdict(3.01)["reached"] == "viable"
    assert gml.pressure_verdict(6.01)["reached"] == "NONE"


def test_pressure_verdict_is_exclusive_at_the_boundary_upward_only():
    """0.7501 ms is not a moonshot.  Nothing rounds toward a target."""
    assert gml.pressure_verdict(0.7501)["reached"] == "dominance"
    assert gml.pressure_verdict(0.75)["reached"] == "moonshot"


# ------------------------------------------------------------------- per-token projection

def test_per_token_labels_every_term_and_never_zeroes_the_unmeasured():
    proj = gml.per_token_projection(1.0, source="test")
    assert proj["moe_layer_ms"]["status"] == "MEASURED"
    assert proj["attention_per_layer_ms"]["status"] == "DERIVED"
    assert proj["dense_mlp_layer_ms"]["status"] == "DERIVED"
    assert len(proj["unmeasured_terms"]) >= 4
    assert any("lm_head" in t for t in proj["unmeasured_terms"])
    assert proj["layer_counts"]["sparse_moe"] + proj["layer_counts"]["dense_mlp"] == 78


def test_per_token_total_is_the_sum_of_its_labelled_terms():
    proj = gml.per_token_projection(1.25, source="test")
    total = (75 * 1.25 + 3 * proj["dense_mlp_layer_ms"]["value"]
             + 78 * proj["attention_per_layer_ms"]["value"])
    assert abs(proj["implied_token_ms"] - total) < 1e-9
    assert abs(proj["implied_tok_s"] - 1000.0 / total) < 1e-9


def test_per_token_is_an_upper_bound_on_tok_s():
    assert "UPPER bound on tok/s" in gml.per_token_projection(1.0, source="t")["note"]


# ------------------------------------------------------------------------- CPU authority

def _fake_expert(rng, rows_a, cols_a, rows_b, cols_b, D=8, k=128):
    def codes(rows, cols):
        nchunk = cols // D
        return {"rows": rows, "cols": cols, "nchunk": nchunk, "D": D, "S": 1, "sub": D,
                "rotate": False, "seed": 0,
                "codebooks": [rng.standard_normal((k, D)).astype(np.float32)],
                "indices": rng.integers(0, k, size=(rows * nchunk, 1)).astype(np.uint8)}
    return {"gate": codes(rows_a, cols_a), "up": codes(rows_a, cols_a),
            "down": codes(rows_b, cols_b)}


def test_cpu_authority_composes_the_graph_not_a_sum_of_projections():
    rng = np.random.default_rng(7)
    experts = [_fake_expert(rng, 16, 32, 32, 16) for _ in range(2)]
    x = rng.standard_normal(32).astype(np.float32)
    w = np.array([0.75, 0.25], dtype=np.float32)
    out = gml.cpu_layer(experts, x, w)
    # recompute by hand, one expert at a time
    want = x.astype(np.float32).copy()
    for exp, weight in zip(experts, w):
        g = forge.pq_execute(gml.artifact_of(exp["gate"]), x)
        u = forge.pq_execute(gml.artifact_of(exp["up"]), x)
        want = want + weight * forge.pq_execute(
            gml.artifact_of(exp["down"]), (gml.silu(g) * u).astype(np.float32))
    assert np.allclose(out["y"], want, atol=0, rtol=0)


def test_cpu_authority_includes_the_residual():
    rng = np.random.default_rng(11)
    experts = [_fake_expert(rng, 16, 32, 32, 16)]
    x = rng.standard_normal(32).astype(np.float32)
    zero = gml.cpu_layer(experts, x, np.array([0.0], dtype=np.float32))
    assert np.allclose(zero["y"], x)                    # weight 0 leaves only the residual


def test_cpu_authority_refuses_a_weight_count_mismatch():
    rng = np.random.default_rng(3)
    with pytest.raises(gml.MoeLayerError):
        gml.cpu_layer([_fake_expert(rng, 16, 32, 32, 16)], np.zeros(32, np.float32),
                      np.ones(2, np.float32))


def test_silu_matches_the_kernel_expression():
    v = np.array([-4.0, -0.5, 0.0, 0.5, 4.0], dtype=np.float32)
    assert np.allclose(gml.silu(v), v / (1.0 + np.exp(-v)), atol=1e-7)


def test_swiglu_error_composes_nonlinearly_so_whole_layer_parity_is_the_number():
    """The reason the gate is on the layer output and not on a projection.

    A relative perturbation of the gate projection does not pass through silu(g)*u
    unchanged; this asserts the amplification exists rather than assuming it.
    """
    rng = np.random.default_rng(5)
    g = rng.standard_normal(4096).astype(np.float32) * 3.0
    u = rng.standard_normal(4096).astype(np.float32)
    eps = 1e-4
    clean = gml.silu(g) * u
    perturbed = gml.silu(g * (1 + eps)) * u
    rel_in = eps
    rel_out = float(np.linalg.norm(perturbed - clean) / np.linalg.norm(clean))
    assert rel_out > rel_in                             # amplified, not merely carried


# ---------------------------------------------------------------------------- parity math

def test_parity_of_reports_the_gate_it_was_judged_against():
    p = gml.parity_of(np.ones(4, np.float32), np.ones(4, np.float32))
    assert p["gate"] == gml.PARITY_GATE == 2e-3
    assert p["relative_l2"] == 0.0
    assert p["finite"]


def test_parity_gate_is_the_modules_own_tolerance_not_one_over_a_million():
    """1e-6 would fail for the right reason and be misread: the codebook is cast to fp16."""
    assert gml.PARITY_GATE == 2e-3


def test_parity_detects_a_wrong_answer():
    ref = np.arange(8, dtype=np.float32) + 1
    assert gml.parity_of(ref * 1.5, ref)["relative_l2"] > gml.PARITY_GATE


# -------------------------------------------------------------------- refuted claim guard

def test_the_refuted_numbers_cannot_be_republished():
    for name in ("dense_fp16_9.012ms", "speedup_35.9x", "metal_parity_1.4e-6"):
        with pytest.raises(Exception):
            gml.lab.assert_not_refuted(name=name)


def test_module_does_not_quote_the_refuted_dense_baseline():
    text = Path(gml.__file__).read_text()
    assert "9.012" not in text
    assert "35.9x" not in text


# --------------------------------------------------------------------- source properties

def test_module_never_opens_the_artifact_directory_for_writing():
    text = Path(gml.__file__).read_text()
    for forbidden in ("shutil", ".unlink(", ".rename(", ".replace(", "open(", "'w'"):
        assert forbidden not in text or forbidden == "open("
    assert "write_text" in text                 # only under REPORT_DIR
    assert str(grf.ARTIFACT_DIR) not in text    # reached only through gravity_real_fixtures


def test_reports_are_written_only_under_the_breakthrough_report_dir():
    assert gml.REPORT_DIR.name == "breakthrough"
    text = Path(gml.__file__).read_text()
    assert text.count("write_text") == text.count("REPORT_DIR /")


def test_command_queue_depth_and_autorelease_pool_are_inherited_not_reinvented():
    """The 64-in-flight deadlock is handled by lab_b's decoder, which this module holds."""
    labb_text = Path(labb.__file__).read_text()
    assert "newCommandQueueWithMaxCommandBufferCount_(1024)" in labb_text
    assert "objc.autorelease_pool()" in Path(gml.__file__).read_text()


def test_the_extra_kernels_are_only_the_two_the_graph_needs():
    assert gml.EXTRA_METAL_SOURCE.count("kernel void ") == 3      # +the vec4 variant
    for name in ("reduce_swiglu", "reduce_swiglu4", "moe_combine"):
        assert f"kernel void {name}(" in gml.EXTRA_METAL_SOURCE


def test_combine_kernel_starts_from_the_residual():
    """The residual is the combine accumulator's initialiser, which is why it has no cost."""
    assert "float acc = residual[gid];" in gml.EXTRA_METAL_SOURCE


def test_band_verdict_needs_median_and_min_to_agree():
    assert gml._band_verdict(1.20, 1.20) == "VEC4_WINS"
    assert gml._band_verdict(1.20, 1.00) == "NEUTRAL_WITHIN_NOISE"
    assert gml._band_verdict(0.80, 0.80) == "SCALAR_WINS"
    assert gml._band_verdict(1.00, 1.00) == "NEUTRAL_WITHIN_NOISE"


def test_selftest_runs_without_a_device():
    assert gml.selftest() == 0


# ---------------------------------------------------------------- real artifacts / device

def _loaded():
    try:
        return gml.load_layer(3)
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"real artifacts unavailable: {exc}")


def test_real_layer_geometry_and_activation_label():
    loaded = _loaded()
    geom = loaded["geometry"]
    assert (geom.rows_a, geom.cols_a) == (2048, 6144)
    assert (geom.rows_b, geom.cols_b) == (6144, 2048)
    assert geom.D == 8 and geom.k == 128
    assert len(loaded["experts"]) == grf.EXPERTS_PER_TOKEN + 1
    assert loaded["fixture_set"]["router_present"] is False
    for exp in loaded["experts"]:
        for proj in ("gate", "up", "down"):
            assert exp[proj]["fixture"].activation_source == grf.SYNTHETIC


def test_every_cache_key_is_a_distinct_content_address():
    """Never id(), never one literal reused -- gravity_metal now refuses both."""
    loaded = _loaded()
    keys = [e[p]["key"] for e in loaded["experts"] for p in ("gate", "up", "down")]
    assert len(keys) == len(set(keys)) == 27
    for key, exp in zip(keys, (e[p] for e in loaded["experts"]
                               for p in ("gate", "up", "down"))):
        shard, name, digest = key.split("::")
        assert shard.endswith(".gravity") and name == exp["name"] and len(digest) == 64


def test_prepare_refuses_a_cache_key_claimed_by_two_tensors():
    loaded = _loaded()
    try:
        ex = gml.MoeLayerExecutor()
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"no Metal device: {exc}")
    experts = [{p: dict(e[p]) for p in ("gate", "up", "down")} for e in loaded["experts"]]
    experts[1]["gate"]["key"] = experts[0]["gate"]["key"]        # the collision
    with pytest.raises(gml.MoeLayerError, match="claimed by both"):
        ex.prepare(experts, gml.routing_weights(8), plan=gml.layer_plan(loaded["geometry"]))


def test_whole_layer_parity_against_the_cpu_authority():
    loaded = _loaded()
    try:
        ex = gml.MoeLayerExecutor()
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"no Metal device: {exc}")
    geom = loaded["geometry"]
    weights = gml.routing_weights(8)
    x = grf.synthetic_activation(geom.cols_a, seed=gml.SEED)
    ex.prepare(loaded["experts"], weights, plan=gml.layer_plan(geom))
    ex.set_input(x)
    got = ex.run_layer()
    authority = gml.cpu_layer(
        [{p: e[p]["codes"] for p in ("gate", "up", "down")} for e in loaded["experts"]],
        x, weights)
    parity = gml.parity_of(got, authority["y"])
    assert parity["finite"]
    assert parity["relative_l2"] < gml.PARITY_GATE
