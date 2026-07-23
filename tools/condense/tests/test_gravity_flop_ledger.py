"""Tests for the FLOP-and-byte ledger.

The analytic models are checked against hand-computed values at the real R0 production
geometries, and the whole-model ledger is checked against the sealed .gravity shard the
live campaign already wrote.  That shard is opened READ-ONLY and never written.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_flop_ledger as fl  # noqa: E402
import gravity_format  # noqa: E402

SHARD = fl.DEFAULT_SHARD
_needs_shard = pytest.mark.skipif(not SHARD.exists(),
                                  reason=f"sealed fixture shard absent: {SHARD}")


# ---------------------------------------------------------------------------
# Analytic models at the real R0 geometries, against hand-computed values.
# ---------------------------------------------------------------------------
def test_decode_fma_gate_up_hand_computed():
    """gate/up [2048,6144], D=8: nchunk=768 and the dense MAC count, exactly."""
    cost = fl.decode_fma_cost(2048, 6144)
    assert cost["nchunk"] == 768                      # 6144 / 8
    assert cost["macs"] == 12_582_912                 # 2048 * 768 * 8 == 2048 * 6144
    assert cost["macs"] == 2048 * 6144, "decode-FMA executes exactly the dense MAC count"
    assert cost["index_ops"] == 1_572_864             # 2048 * 768
    assert cost["threadgroups"] == 8                  # ceil(2048 / 256)
    assert cost["stage_x"] is True                    # (128*8 + 768*8)*4 = 28672 <= 32768


def test_decode_fma_down_hand_computed():
    """down [6144,2048], D=8: nchunk=256, same dense MAC count, different shape."""
    cost = fl.decode_fma_cost(6144, 2048)
    assert cost["nchunk"] == 256                      # 2048 / 8
    assert cost["macs"] == 12_582_912 == 6144 * 2048
    assert cost["index_ops"] == 1_572_864             # 6144 * 256
    assert cost["threadgroups"] == 24                 # ceil(6144 / 256)


def test_lookup_linear_gate_up_hand_computed():
    """The 5.33x arithmetic reduction the representation permits but no kernel implements."""
    cost = fl.lookup_linear_cost(2048, 6144)
    assert cost["table_macs"] == 786_432              # 768 * 128 * 8
    assert cost["gather_adds"] == 1_572_864           # 2048 * 768
    assert cost["total_ops"] == 2_359_296
    assert cost["reduction_vs_decode_fma"] == pytest.approx(16.0 / 3.0)   # 5.333...
    assert cost["table_bytes"] == 393_216             # 768 * 128 * 4, fp32
    assert fl.lookup_linear_cost(2048, 6144, table_dtype_bytes=2)["table_bytes"] == 196_608


def test_lookup_linear_down_hand_computed():
    cost = fl.lookup_linear_cost(6144, 2048)
    assert cost["table_macs"] == 262_144              # 256 * 128 * 8
    assert cost["gather_adds"] == 1_572_864           # 6144 * 256
    assert cost["total_ops"] == 1_835_008
    assert cost["reduction_vs_decode_fma"] == pytest.approx(12_582_912 / 1_835_008)


def test_per_threadgroup_rereads_are_counted():
    """gravity_metal.bytes_read_per_matvec omits them; this model must not."""
    cost = fl.decode_fma_cost(2048, 6144)
    assert cost["codebook_bytes_single"] == 2048      # 128 * 8 halves
    assert cost["codebook_bytes_reread"] == 8 * 2048  # one staged copy per threadgroup
    assert cost["x_bytes_single"] == 24_576           # 768 * 8 floats
    assert cost["x_bytes_reread"] == 8 * 24_576
    assert cost["index_bytes"] == 1_572_864           # 8-bit upload, not the billed 7-bit
    assert cost["device_bytes_read"] == 1_785_856
    assert cost["reported_by_gravity_metal"] == 1_574_912
    dense = cost["dense_bf16_weight_bytes"]
    assert dense == 25_165_824
    assert cost["device_bytes_read"] / dense == pytest.approx(0.0710, abs=5e-5)
    assert cost["reported_by_gravity_metal"] / dense == pytest.approx(0.0626, abs=5e-5)


def test_kernel_index_stream_is_fatter_than_the_artifact():
    """R0 bills 7-bit indices; the kernel uploads 8-bit.  14.3% fatter, and it must show."""
    kernel = fl.decode_fma_cost(2048, 6144, index_bits=8)["index_bytes"]
    artifact = fl.decode_fma_cost(2048, 6144, index_bits=7)["index_bytes"]
    assert kernel / artifact == pytest.approx(8 / 7, rel=1e-6)


def test_unstaged_x_streams_per_thread_chunk():
    cost = fl.decode_fma_cost(2048, 6144, stage_x=False)
    assert cost["x_bytes_reread"] == 2048 * 768 * 8 * 4


def test_nchunk_requires_divisible_cols():
    with pytest.raises(fl.FlopLedgerError):
        fl.decode_fma_cost(2048, 6145)


# ---------------------------------------------------------------------------
# Ratio arithmetic and the UNMEASURED discipline.
# ---------------------------------------------------------------------------
def _record(**overrides):
    base = dict(workload="w", dense_equivalent_macs=1000, dense_equivalent_bytes=2000,
                executed_flops=1000, executed_int_ops=100,
                representation_overhead_ops=0, bytes_read=100, bytes_written=0,
                dispatches=1, latency_s=None, evidence="TEST")
    base.update(overrides)
    return fl.WorkloadRecord(**base)


def test_ratio_arithmetic():
    record = _record()
    assert record.executed_arithmetic == 1100
    assert record.executed_bytes == 100
    assert record.arithmetic_compression == pytest.approx(1000 / 1100)
    assert record.flop_compression == pytest.approx(1.0)
    assert record.byte_compression == pytest.approx(20.0)
    assert record.arithmetic_intensity_flop_per_byte == pytest.approx(11.0)


@pytest.mark.parametrize("field", ["executed_flops", "executed_int_ops",
                                   "representation_overhead_ops"])
def test_none_arithmetic_field_never_becomes_zero(field):
    record = _record(**{field: None})
    assert record.executed_arithmetic is None
    assert record.arithmetic_compression is None
    # the surviving fields still report, so UNMEASURED is contained, not contagious
    assert record.byte_compression == pytest.approx(20.0)


@pytest.mark.parametrize("field", ["bytes_read", "bytes_written"])
def test_none_byte_field_never_becomes_zero(field):
    record = _record(**{field: None})
    assert record.executed_bytes is None
    assert record.byte_compression is None
    assert record.arithmetic_intensity_flop_per_byte is None


def test_none_numerator_poisons_the_ratio():
    record = _record(dense_equivalent_macs=None, dense_equivalent_bytes=None)
    assert record.arithmetic_compression is None
    assert record.flop_compression is None
    assert record.byte_compression is None


def test_zero_denominator_is_none_not_infinity():
    assert fl._ratio(5, 0) is None
    assert fl._ratio(None, 5) is None
    assert fl._ratio(5, None) is None


def test_record_json_names_its_unmeasured_fields():
    row = _record(latency_s=None, dispatches=None).to_json()
    assert row["schema"] == fl.RECORD_SCHEMA
    assert row["unmeasured_fields"] == ["dispatches", "latency_s"]


# ---------------------------------------------------------------------------
# Classifier rules.
# ---------------------------------------------------------------------------
def test_classifier_fewer_bytes_same_arithmetic():
    """The real R0 case: dense MAC count runs, only the bytes fell."""
    verdict = fl.classify([fl.matvec_record(2048, 6144, workload="gate_proj")])
    assert verdict["classification"] == "FEWER_BYTES_SAME_ARITHMETIC"
    assert verdict["flop_compression"] == pytest.approx(1.0)
    assert verdict["byte_compression"] > 1.0
    # counting index loads, the executed total is strictly WORSE than dense
    assert verdict["arithmetic_compression"] < 1.0


def test_classifier_fewer_flops_and_bytes():
    verdict = fl.classify([_record(executed_flops=100, executed_int_ops=10,
                                   representation_overhead_ops=5)])
    assert verdict["classification"] == "FEWER_FLOPS_AND_BYTES"
    assert verdict["flop_compression"] == pytest.approx(10.0)


def test_classifier_native_functional_runtime_requires_zero_overhead():
    free = _record(executed_flops=100, executed_int_ops=10, representation_overhead_ops=0)
    assert fl.classify([free])["classification"] == "NATIVE_FUNCTIONAL_RUNTIME"
    taxed = _record(executed_flops=100, executed_int_ops=10, representation_overhead_ops=1)
    assert fl.classify([taxed])["classification"] == "FEWER_FLOPS_AND_BYTES"


def test_classifier_conditional_compute_wins_over_everything():
    skipped = _record(workload="skipped", executed_flops=0, executed_int_ops=0,
                      representation_overhead_ops=0, bytes_read=0, bytes_written=0)
    cheap = _record(workload="cheap", executed_flops=100, executed_int_ops=0,
                    representation_overhead_ops=0)
    verdict = fl.classify([skipped, cheap])
    assert verdict["classification"] == "CONDITIONAL_COMPUTE"
    assert "skipped work" in verdict["rule"]


def test_classifier_refuses_to_classify_unmeasured():
    verdict = fl.classify([_record(), _record(workload="blind", bytes_read=None)])
    assert verdict["classification"] == "UNMEASURED_INSUFFICIENT"
    assert verdict["unmeasured_records"] == ["blind"]
    assert verdict["byte_compression"] is None
    assert verdict["flop_compression"] is None


def test_classifier_needs_a_record():
    with pytest.raises(fl.FlopLedgerError):
        fl.classify([])


def test_every_classification_is_in_the_declared_set():
    for case in ([fl.matvec_record(2048, 6144, workload="w")],
                 [_record(executed_flops=100, executed_int_ops=10,
                          representation_overhead_ops=5)],
                 [_record(bytes_read=None)]):
        assert fl.classify(case)["classification"] in fl.CLASSIFICATIONS


# ---------------------------------------------------------------------------
# The real sealed artifact is the authority on per-tensor bytes.
# ---------------------------------------------------------------------------
@_needs_shard
def test_analytic_bytes_match_the_real_shard_header():
    report = fl.validate_against_shard(SHARD)
    assert report["production_rung"] == "R0"
    assert report["tensor_count"] == 211
    assert report["all_analytic_bytes_match_header"], report["distinct_geometries"]
    assert report["all_shapes_match_adapter"], report["distinct_geometries"]
    shapes = {tuple(row["shape"]) for row in report["distinct_geometries"]}
    assert shapes == {(16384, 2048), (6144, 2048), (2048, 6144)}


@_needs_shard
def test_packed_tensor_bytes_is_header_bytes_tensor_by_tensor():
    """Not just per distinct shape: every one of the 211 descriptors, individually."""
    header = gravity_format.read_header(SHARD)
    for tensor in header["tensors"]:
        rows, cols = (int(dim) for dim in tensor["shape"])
        assert fl.packed_tensor_bytes(rows, cols) == int(tensor["bytes"]), tensor["name"]
        assert rows * cols == int(tensor["elements"]), tensor["name"]


@_needs_shard
def test_header_bpw_matches_the_billed_r0_rate():
    header = gravity_format.read_header(SHARD)
    assert header["compression"]["packed_bpw"] == pytest.approx(0.87633225, abs=1e-8)
    for tensor in header["tensors"]:
        predicted = fl.packed_tensor_bytes(*(int(d) for d in tensor["shape"]))
        assert predicted * 8 / int(tensor["elements"]) == pytest.approx(
            float(tensor["bpw"]), rel=1e-12)


def test_shard_is_opened_read_only():
    """The fixture lives in the live campaign's active output directory."""
    before = SHARD.stat().st_mtime_ns if SHARD.exists() else None
    if before is None:
        pytest.skip("fixture absent")
    fl.validate_against_shard(SHARD)
    assert SHARD.stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# Whole-model ledger totals.
# ---------------------------------------------------------------------------
def test_geometry_is_the_pinned_official_one():
    geometry = fl.official_geometry()
    assert geometry.num_hidden_layers == 78
    assert geometry.first_k_dense_replace == 3
    assert geometry.n_routed_experts == 256
    assert geometry.num_experts_per_tok == 8
    assert geometry.n_shared_experts == 1
    assert geometry.moe_intermediate_size == 2048
    assert geometry.hidden_size == 6144
    full = sum(1 for layer in range(78) if geometry.indexer_type(layer) == "full")
    assert (full, 78 - full) == (21, 57)


def test_ladder_admissibility_matches_the_packer():
    """0.875 + (128*8*16 + 512)/elements < 1.0  ->  elements > 135168."""
    assert not fl.ladder_admissible(135_168)
    assert fl.ladder_admissible(135_169)
    assert fl.ladder_admissible(2048 * 6144)


def test_active_rows_are_the_top_k_expert_set_not_all_256():
    geometry = fl.official_geometry()
    rows = fl.active_tensor_rows(geometry)
    experts = [r for r in rows if r["organ"] == "routed_expert"]
    assert len(experts) == 75 * 8 * 3          # sparse layers * top-k * (gate, up, down)
    shared = [r for r in rows if r["organ"] == "shared_expert"]
    assert len(shared) == 75 * 3
    dense_mlp = [r for r in rows if r["organ"] == "dense_mlp"]
    assert len(dense_mlp) == 3 * 3
    attention = [r for r in rows if r["organ"] == "attention"]
    assert len(attention) == 78 * 5
    indexer = [r for r in rows if r["organ"] == "indexer"]
    assert len(indexer) == 21 * 5
    assert {r["name"] for r in rows if r["organ"] == "lm_head"} == {"lm_head.weight"}
    assert not [r for r in rows if ".78." in r["name"]], "MTP layer must be excluded"


def test_control_sensitive_tensors_stay_dense():
    rows = {r["name"]: r for r in fl.active_tensor_rows(fl.official_geometry())}
    router = rows["model.layers.5.mlp.gate.weight"]
    assert router["terminal_state"] == "PROTECTED_SOURCE_NATIVE_WITH_BILLED_BYTES"
    assert router["active_bytes"] == router["dense_bf16_bytes"] == 256 * 6144 * 2
    norm = rows["model.layers.5.input_layernorm.weight"]
    assert norm["active_bytes"] == 6144 * 2
    bias = rows["model.layers.5.mlp.gate.e_score_correction_bias"]
    assert bias["dtype"] == "F32" and bias["active_bytes"] == 256 * 4
    expert = rows["model.layers.5.mlp.experts.0.gate_proj.weight"]
    assert expert["terminal_state"] == "PACKED_IN_CORE_ARTIFACT"
    assert expert["active_bytes"] == fl.packed_tensor_bytes(2048, 6144) == 1_378_368


def test_embedding_is_a_row_gather_not_a_full_read():
    rows = {r["name"]: r for r in fl.active_tensor_rows(fl.official_geometry())}
    embed = rows["model.embed_tokens.weight"]
    assert embed["terminal_state"] == "PACKED_ROW_GATHER"
    assert embed["dense_bf16_bytes"] == 6144 * 2
    assert embed["active_bytes"] == 64 + 128 * 8 * 2 + (768 * 7 + 7) // 8
    assert embed["dense_equivalent_macs"] == 0
    assert rows["lm_head.weight"]["dense_equivalent_macs"] == 154_880 * 6144


@_needs_shard
def test_ledger_totals_are_the_sum_of_their_parts():
    ledger = fl.active_byte_ledger(SHARD)
    rows = fl.active_tensor_rows(fl.official_geometry())
    totals = ledger["totals"]
    assert totals["active_bytes_per_token"] == sum(r["active_bytes"] for r in rows)
    assert totals["dense_bf16_bytes_per_token"] == sum(r["dense_bf16_bytes"] for r in rows)
    assert totals["dense_equivalent_macs_per_token"] == sum(
        r["dense_equivalent_macs"] for r in rows)
    assert sum(b["active_bytes"] for b in ledger["per_organ"]) == totals["active_bytes_per_token"]
    assert totals["byte_compression"] == pytest.approx(
        totals["dense_bf16_bytes_per_token"] / totals["active_bytes_per_token"])
    # every packed tensor is billed under the campaign's 1.0 BPW ceiling
    packed = [r for r in rows if r["terminal_state"] == "PACKED_IN_CORE_ARTIFACT"]
    assert packed and all(r["active_bytes"] * 8 / r["elements"] < 1.0 for r in packed)
    # the kernel moves strictly more than the artifact holds, and the ledger says so
    assert totals["kernel_device_bytes_per_token"] > totals["active_bytes_per_token"]
    assert ledger["artifact_validation"]["all_analytic_bytes_match_header"]


@_needs_shard
def test_x_staging_failures_are_named_and_dominate_the_kernel_amplification():
    """o_proj and the dense down_proj are too wide to stage x, so they re-read it."""
    ledger = fl.active_byte_ledger(SHARD)
    failures = ledger["x_staging_failures"]
    assert failures["threadgroup_memory_limit_bytes"] == 32768
    assert failures["distinct_shapes"] == [[6144, 12288], [6144, 16384]]
    assert failures["tensors_per_token"] == 78 + 3      # every o_proj, every dense down_proj
    # (128*8 + 2048*8)*4 = 69632 > 32768, so gravity_metal.py takes the unstaged branch
    assert fl.decode_fma_cost(6144, 16384)["stage_x"] is False
    assert fl.decode_fma_cost(6144, 12288)["stage_x"] is False
    assert fl.decode_fma_cost(2048, 6144)["stage_x"] is True
    assert failures["kernel_device_bytes"] > 0.8 * (
        ledger["totals"]["kernel_device_bytes_per_token"]
        - ledger["totals"]["active_bytes_per_token"])


@_needs_shard
def test_ledger_hand_checks_one_whole_sparse_layer():
    """Recompute one sparse layer's active bytes independently of the aggregation."""
    ledger = fl.active_byte_ledger(SHARD)
    rows = [r for r in fl.active_tensor_rows(fl.official_geometry())
            if r["name"].startswith("model.layers.5.")]
    gate_up = fl.packed_tensor_bytes(2048, 6144)
    down = fl.packed_tensor_bytes(6144, 2048)
    expected = (
        fl.packed_tensor_bytes(2048, 6144)            # q_a_proj
        + fl.packed_tensor_bytes(16384, 2048)         # q_b_proj
        + fl.packed_tensor_bytes(576, 6144)           # kv_a_proj_with_mqa
        + fl.packed_tensor_bytes(28672, 512)          # kv_b_proj
        + fl.packed_tensor_bytes(6144, 16384)         # o_proj
        + 8 * (2 * gate_up + down)                    # top-8 routed experts
        + (2 * gate_up + down)                        # shared expert
        + 256 * 6144 * 2 + 256 * 4                    # router + f32 correction bias
        + 2 * 6144 * 2 + 2048 * 2 + 512 * 2           # the four norms
    )
    assert sum(r["active_bytes"] for r in rows) == expected
    assert ledger["architecture"]["sparse_moe_layers"] == 75


@_needs_shard
def test_kv_reads_name_their_missing_input_rather_than_inventing_it():
    ledger = fl.active_byte_ledger(SHARD, context_length=8192)
    kv = ledger["kv_reads"]
    assert kv["latent_width_per_position"] == 576            # 512 + 64
    assert kv["attended_positions"] == 2048                  # capped by index_topk
    assert kv["kv_latent_elements"] == 78 * 2048 * 576
    assert kv["index_key_elements"] == 21 * 8192 * 128
    assert kv["kv_latent_bytes"] is None and kv["index_key_bytes"] is None
    assert {u["field"] for u in kv["unknowns"]} == {
        "kv_cache_dtype_bytes", "index_cache_dtype_bytes"}
    assert ledger["totals_including_kv"]["active_bytes_per_token"] is None
    assert all("missing_input" in u for u in ledger["unknowns"])


@_needs_shard
def test_kv_reads_compute_once_the_dtype_is_supplied():
    ledger = fl.active_byte_ledger(SHARD, context_length=4096,
                                   kv_cache_dtype_bytes=2, index_cache_dtype_bytes=2)
    kv = ledger["kv_reads"]
    assert kv["kv_latent_bytes"] == 78 * 2048 * 576 * 2
    assert kv["index_key_bytes"] == 21 * 4096 * 128 * 2
    assert ledger["totals_including_kv"]["active_bytes_per_token"] == (
        ledger["totals"]["active_bytes_per_token"]
        + kv["kv_latent_bytes"] + kv["index_key_bytes"])


# ---------------------------------------------------------------------------
# Roofline.
# ---------------------------------------------------------------------------
@_needs_shard
def test_roofline_reports_both_bounds_and_names_the_binding_one():
    roofline = fl.token_roofline(fl.active_byte_ledger(SHARD))
    bytes_per_token = roofline["active_bytes_per_token"]
    macs = roofline["executed_macs_per_token"]
    assert roofline["bandwidth_bound_s_per_token"] == pytest.approx(
        bytes_per_token / 736.0e9)
    assert roofline["compute_bound_s_per_token"] == pytest.approx(
        macs * 2 / 17_703.0e9)
    assert roofline["binds"] in {"BANDWIDTH", "COMPUTE"}
    assert roofline["binds"] == (
        "COMPUTE" if roofline["compute_bound_s_per_token"]
        > roofline["bandwidth_bound_s_per_token"] else "BANDWIDTH")
    assert roofline["bound_tokens_per_s"] == pytest.approx(
        1.0 / max(roofline["bandwidth_bound_s_per_token"],
                  roofline["compute_bound_s_per_token"]))


@_needs_shard
def test_roofline_uses_the_measured_roof_not_the_vendor_figure():
    assert fl.MEASURED_READ_BYTES_PER_S == 736.0e9
    roofline = fl.token_roofline(fl.active_byte_ledger(SHARD))
    constants = roofline["measured_constants"]
    assert constants["sustained_read_bytes_per_s"] == 736.0e9
    assert 819.0e9 not in [v for v in constants.values() if isinstance(v, (int, float))]
    assert "819" in constants["sustained_read_note"], "say plainly which figure was rejected"
    # the bound must be the one 736 GB/s implies, not the one 819 GB/s would
    assert roofline["bandwidth_bound_s_per_token"] == pytest.approx(
        roofline["active_bytes_per_token"] / 736.0e9)


@_needs_shard
def test_roofline_ridge_agrees_with_the_binding_bound():
    roofline = fl.token_roofline(fl.active_byte_ledger(SHARD))
    intensity = roofline["arithmetic_intensity_flop_per_byte"]
    above = intensity > fl.MEASURED_RIDGE_FLOP_PER_BYTE
    assert roofline["ridge_comparison"] == (
        "ABOVE_RIDGE_COMPUTE_BOUND" if above else "BELOW_RIDGE_BANDWIDTH_BOUND")
    assert (roofline["binds"] == "COMPUTE") == above


def test_roofline_propagates_unmeasured_inputs():
    ledger = {"totals": {"active_bytes_per_token": None,
                         "dense_equivalent_macs_per_token": None}}
    roofline = fl.token_roofline(ledger, ops_per_token=None)
    assert roofline["bandwidth_bound_s_per_token"] is None
    assert roofline["binds"] == "UNMEASURED"
    assert roofline["bound_tokens_per_s"] is None


# ---------------------------------------------------------------------------
# The measured record, and emission.
# ---------------------------------------------------------------------------
def test_measured_down_proj_record_carries_a_real_latency():
    record = fl.measured_down_proj_record()
    assert record.latency_s == pytest.approx(0.2096e-3)
    assert record.dense_equivalent_macs == 6144 * 2048
    assert record.executed_flops == record.dense_equivalent_macs
    assert record.evidence.startswith("MEASURED")
    assert record.to_json()["unmeasured_fields"] == []
    # measured GPU time against the analytic bandwidth bound for the bytes it moves
    modelled = record.bytes_read / fl.MEASURED_READ_BYTES_PER_S
    assert record.latency_s / modelled > 50, "the kernel is far off its own byte bound"


@_needs_shard
def test_emit_writes_both_reports(tmp_path):
    written = fl.emit(tmp_path)
    assert set(written) == {"GLM52_ACTIVE_BYTE_LEDGER.json", "GLM52_TOKEN_ROOFLINE.json"}
    import json
    ledger = json.loads(written["GLM52_ACTIVE_BYTE_LEDGER.json"].read_text())
    roofline = json.loads(written["GLM52_TOKEN_ROOFLINE.json"].read_text())
    assert ledger["schema"] == "hawking.gravity.active_byte_ledger.v1"
    assert roofline["schema"] == "hawking.gravity.token_roofline.v1"
    assert ledger["evidence_level"] == "F0_PHYSICAL_ACCOUNTING_ONLY"
    assert "not_evidence_of" in ledger and "not_evidence_of" in roofline
    assert roofline["classification"]["classification"] == "FEWER_BYTES_SAME_ARITHMETIC"
    assert roofline["grammar_comparison"]["gate_up_lookup_linear"][
        "reduction_vs_decode_fma"] == pytest.approx(16 / 3)
    assert all(math.isfinite(v) for v in (roofline["bandwidth_bound_s_per_token"],
                                          roofline["compute_bound_s_per_token"]))
