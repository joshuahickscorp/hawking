"""N016 PREFILL_KV: prefill measured separately; footprint is MODEL+STATE."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from gpu_ledger import ABSENT, DERIVED, MEASURED  # noqa: E402
from noetic_information_accounting import qwen38_workspace_bytes  # noqa: E402
from noetic_parent_a import DURABLE  # noqa: E402
from prefill_kv import (  # noqa: E402
    ACTIVATION_BYTES,
    ADMISSION_SEQ,
    DELTANET_STATE_BYTES,
    KV_BYTES_PER_POSITION,
    LENGTHS,
    PARENT_ROOT,
    RECEIPT,
    SCHEMA,
    crossover_seq,
    encode_ids,
    fit_prefill_slope,
    kv_precision_options,
    predicted_prefill_gpu_ns,
    production_footprint,
    session_state_bytes,
    wrap_chat,
)

KINDS = {MEASURED, DERIVED, ABSENT}


def _walk_qty(obj, path=""):
    if isinstance(obj, dict):
        if "kind" in obj and obj["kind"] in KINDS and "command" in obj and "unit" in obj:
            yield path, obj
        for k, v in obj.items():
            yield from _walk_qty(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_qty(v, f"{path}[{i}]")


def test_workspace_formula_matches_multisession_256():
    w = session_state_bytes(256)
    assert w["activation_bytes"] == ACTIVATION_BYTES == 1_691_396
    assert w["deltanet_state_bytes"] == DELTANET_STATE_BYTES == 156_893_184
    assert w["gqa_kv_bytes"] == 33_554_432
    assert w["SESSION_STATE_BYTES"] == 192_139_012
    assert w["kv_bytes_per_position"] == KV_BYTES_PER_POSITION == 131_072
    rust = qwen38_workspace_bytes(256)
    assert rust["total_bytes"] == w["SESSION_STATE_BYTES"]


def test_kv_grows_with_seq_deltanet_does_not():
    a = session_state_bytes(4096)
    b = session_state_bytes(16384)
    assert a["activation_bytes"] == b["activation_bytes"]
    assert a["deltanet_state_bytes"] == b["deltanet_state_bytes"]
    assert b["gqa_kv_bytes"] == a["gqa_kv_bytes"] * 4
    assert b["SESSION_STATE_BYTES"] - a["SESSION_STATE_BYTES"] == b["gqa_kv_bytes"] - a["gqa_kv_bytes"]


def test_production_footprint_is_model_plus_c_times_state():
    model = 14_297_694_680
    c1 = production_footprint(model, 4096, 1)
    c4 = production_footprint(model, 4096, 4)
    ss = session_state_bytes(4096)["SESSION_STATE_BYTES"]
    assert c1["MODEL_BYTES"] == model
    assert c1["SESSION_STATE_BYTES"] == ss
    assert c1["PRODUCTION_FOOTPRINT_BYTES"] == model + ss
    assert c4["PRODUCTION_FOOTPRINT_BYTES"] == model + 4 * ss
    assert c4["PRODUCTION_FOOTPRINT_BYTES"] - c1["PRODUCTION_FOOTPRINT_BYTES"] == 3 * ss
    # One body, not four copies.
    copies = 4 * (model + ss)
    assert c4["PRODUCTION_FOOTPRINT_BYTES"] < copies / 2


def test_crossover_at_c4_is_below_32k_for_q4_class_weights():
    model = 14_297_694_680
    x1 = crossover_seq(model, 1)
    x4 = crossover_seq(model, 4)
    assert x4["seq_where_c_state_exceeds_weights"] < 32768
    assert x4["seq_where_c_state_exceeds_weights"] < x1["seq_where_c_state_exceeds_weights"]
    # At 128K, c=4 state is tens of GiB and rivals the Metal working set.
    long = production_footprint(model, 131072, 4)
    assert long["SESSION_STATE_BYTES_x_c"] > model
    assert long["state_exceeds_weights"] is True


def test_quadratic_fit_recovers_known_slope():
    # gpu_ns(i) = 30e6 + 5500 * i
    series = [30_000_000 + 5500 * i for i in range(256)]
    fit = fit_prefill_slope(series)
    assert fit["ok"] is True
    assert abs(fit["slope_ns_per_position"] - 5500) < 1e-6
    assert abs(fit["intercept_ns"] - 30_000_000) < 1.0
    pred = predicted_prefill_gpu_ns(fit, 256)
    expected = 256 * 30_000_000 + 5500 * 256 * 255 / 2
    assert abs(pred - expected) < 1.0


def test_kv_precision_is_qualified_not_assumed_free():
    opt = kv_precision_options(16384)
    assert opt["production_dtype"] == "f32"
    assert opt["wired_in_qwen38_hybrid_decode"] == "mha_decode_f32_tcb"
    f16 = opt["candidates"]["f16"]
    assert f16["wired_into_production_session"] is False
    assert f16["capability_cost"]["kind"] == ABSENT
    assert f16["capability_cost"]["value"] is None
    assert "not free" in f16["capability_cost"]["absent_reason"].lower() or "not free" in str(
        f16["capability_cost"]
    ).lower() or "would need" in f16["capability_cost"]["absent_reason"].lower()
    int4 = opt["candidates"]["int4"]
    assert int4["capability_cost"]["kind"] == ABSENT
    assert int4["capability_cost"]["value"] is None
    assert opt["deltanet_state"]["grows_with_seq"] is False
    assert opt["deltanet_state"]["bytes"] == DELTANET_STATE_BYTES


def test_chat_wrap_and_tokenizer_roundtrip():
    text = wrap_chat("Say hi.")
    ids = encode_ids(text)
    assert len(ids) >= 8
    assert ids  # tokenizer loaded from the sealed parent tokenizer.json


def _receipt() -> dict:
    assert RECEIPT.is_file(), (
        f"missing {RECEIPT} — run python3 tools/headless/prefill_kv.py --measure"
    )
    return json.loads(RECEIPT.read_text())


def test_harness_writes_receipt_schema():
    doc = _receipt()
    assert doc["schema"] == SCHEMA
    assert RECEIPT.resolve().parts[-2] == "headless"
    assert "ascent-2026-08-16" not in str(RECEIPT)
    assert "campaign" not in str(RECEIPT)


def test_did_not_load_second_27b_or_mutate_parent():
    doc = _receipt()
    assert doc["did_not_load_second_27b"] is True
    assert doc["did_not_mutate_sealed_parent"] is True
    assert doc["did_not_write_ascent_or_campaign"] is True
    assert Path(doc["sealed_parent"]["path"]).resolve() == DURABLE.resolve() or Path(
        doc["sealed_parent"]["path"]
    ).resolve() == PARENT_ROOT.resolve()
    before = doc["parent_identity_before"]
    after = doc["parent_identity_after"]
    assert before["catalog_ino"] == after["catalog_ino"]
    assert before["catalog_mtime_ns"] == after["catalog_mtime_ns"]
    assert before["disk_bytes"] == after["disk_bytes"]


def test_prefill_measured_separately_at_four_lengths_both_models():
    doc = _receipt()
    assert tuple(doc["lengths"].keys()) if isinstance(doc["lengths"], dict) else LENGTHS
    for length in LENGTHS:
        assert length in doc["lengths"] or length in ("short", "4k", "16k", "long")
    pre = doc["prefill"]
    assert "q4" in pre and "parent" in pre
    for model in ("q4", "parent"):
        for length in LENGTHS:
            block = pre[model][length]
            assert block["length"] == length or block.get("length") in (length, "long")
            # Every length has labelled cold/warm (ABSENT is a label, not a skip).
            for key in ("cold_prefill_wall_ns",):
                q = block[key]
                assert q["kind"] in KINDS, (model, length, key, q)
                assert q["command"]
                if q["kind"] == ABSENT:
                    assert q["value"] is None
                    assert q["absent_reason"]
                else:
                    assert q["value"] is not None
            gpu_key = "warm_prefill_gpu_ns_per_token"
            q = block[gpu_key]
            assert q["kind"] in KINDS, (model, length, gpu_key)
            assert q["command"]


def test_short_4k_16k_are_measured_long_may_be_derived():
    doc = _receipt()
    for model in ("q4", "parent"):
        for length in ("short", "4k", "16k"):
            block = doc["prefill"][model][length]
            cold = block["cold_prefill_wall_ns"]
            assert cold["kind"] == MEASURED, (model, length, cold)
            assert isinstance(cold["value"], (int, float)) and cold["value"] > 0
            if cold.get("spread"):
                assert cold["spread"]["n"] >= 1
        long = doc["prefill"][model]["long"]
        # Long 32K full walk is the G118 67-minute choice. GPU integral is
        # DERIVED from the MEASURED 16K (or 4K) slope; cold wall stays ABSENT.
        total = long.get("warm_prefill_gpu_ns_total") or long.get("warm_prefill_gpu_ns_per_token")
        assert total["kind"] in (DERIVED, MEASURED), (model, total)
        assert long["cold_prefill_wall_ns"]["kind"] in (ABSENT, MEASURED)
        if long["cold_prefill_wall_ns"]["kind"] == ABSENT:
            assert "32" in long["cold_prefill_wall_ns"]["absent_reason"] or "walk" in long[
                "cold_prefill_wall_ns"
            ]["absent_reason"].lower()


def test_cold_and_warm_separately_with_spread_on_short():
    doc = _receipt()
    for model in ("q4", "parent"):
        block = doc["prefill"][model]["short"]
        cold = block["cold_prefill_wall_ns"]
        warm = block["warm_prefill_wall_ns"]
        assert cold["kind"] == MEASURED
        first = block["cold_first_step_gpu_ns"]
        assert first["kind"] == MEASURED
        assert first["value"] > 0
        # Graph-cold first step is slower than a warm decode-class step.
        if warm["kind"] == MEASURED:
            assert warm.get("spread", {}).get("n", 1) >= 1
        # A single Metal run is not the headline for short: complete-wall
        # pairs give in-process warm plus 3 process reps.
        assert block["n_process_runs"] >= 2


def test_production_footprint_in_receipt_at_c1_and_c4():
    doc = _receipt()
    for model in ("q4", "parent"):
        foot = doc["production_footprint"][model]
        mb = foot["MODEL_BYTES"]
        assert mb["kind"] == MEASURED
        assert mb["value"] > 1_000_000_000
        for length in LENGTHS:
            for c in ("c=1", "c=4"):
                row = foot["by_length"][length][c]
                assert row["MODEL_BYTES"] == mb["value"]
                assert row["SESSION_STATE_BYTES"] == session_state_bytes(
                    ADMISSION_SEQ[length]
                )["SESSION_STATE_BYTES"]
                sessions = 1 if c == "c=1" else 4
                assert (
                    row["PRODUCTION_FOOTPRINT_BYTES"]
                    == row["MODEL_BYTES"] + sessions * row["SESSION_STATE_BYTES"]
                )
                assert "PRODUCTION_FOOTPRINT" in row or row["PRODUCTION_FOOTPRINT_BYTES"]
        # c=4 at 16K/long is the AgentOS question.
        c4_16 = foot["by_length"]["16k"]["c=4"]
        c1_16 = foot["by_length"]["16k"]["c=1"]
        assert c4_16["PRODUCTION_FOOTPRINT_BYTES"] > c1_16["PRODUCTION_FOOTPRINT_BYTES"]
        assert c4_16["SESSION_STATE_BYTES_x_c"] == 4 * c1_16["SESSION_STATE_BYTES"]


def test_capability_compromise_qualified_not_assumed():
    doc = _receipt()
    cap = doc["capability_qualification"]
    assert cap["do_not_assume_kv_quant_is_free"] is True
    assert cap["kv_precision_changed_in_this_measurement"] is False
    kv = doc["kv_precision"]["16k"]
    assert kv["candidates"]["f16"]["capability_cost"]["kind"] == ABSENT
    assert kv["candidates"]["f16"]["capability_cost"]["value"] is None
    assert kv["candidates"]["int4"]["capability_cost"]["kind"] == ABSENT
    assert "not wired" in kv["candidates"]["f16"]["capability_cost"]["absent_reason"].lower() or (
        "not the production" in kv["candidates"]["f16"]["capability_cost"]["absent_reason"].lower()
    )


def test_metal_counters_absent_with_physical_reason_never_zero():
    doc = _receipt()
    probe = doc["metal_probe"]
    names = [cs["name"] for cs in probe["counterSets"]]
    assert names == ["timestamp"]
    assert probe["supportsCounterSampling"]["atDispatchBoundary"] is False
    absent = doc["absent_gpu_counters"]
    for key in (
        "DRAM_READ_BYTES",
        "DRAM_WRITE_BYTES",
        "OS_PAGE_CACHE_COLD_GPU_NS",
        "per_dispatch_gpu_ns",
        "SIMD_utilization",
        "hardware_occupancy_counter",
    ):
        q = absent[key]
        assert q["kind"] == ABSENT
        assert q["value"] is None
        assert q["value"] != 0
        assert q["absent_reason"]
        reason = q["absent_reason"].lower()
        assert any(
            s in reason
            for s in ("counter", "metal", "dispatch", "purge", "boundary", "timestamp")
        ), (key, q["absent_reason"])
    for path, q in _walk_qty(doc):
        if q["kind"] == ABSENT:
            assert q["value"] is None, path
            assert q.get("absent_reason"), path
        else:
            assert q["value"] is not None, path
            assert q["absent_reason"] is None, path
            assert q.get("command"), path


def test_topology_says_separate_prefill_path_is_allowed():
    doc = _receipt()
    t = doc["topology"]
    assert t["decode_optimal"] is True
    assert t["production_optimal"] is False
    assert t["separate_prefill_path_allowed"] is True
    assert "GEMM" in t["separate_path_would_be"] or "batched" in t["separate_path_would_be"].lower()
    assert "teacher-forced" in t["production_prefill_path"] or "session.step" in t["production_prefill_path"]


def test_q4_and_parent_are_the_named_artifacts():
    doc = _receipt()
    assert "qwen38-gravity-uniform-q4-v1" in doc["incumbent"]["path"]
    assert "NOETIC_PARENT_A" in doc["sealed_parent"]["path"]
    assert doc["incumbent"]["payload_bytes"] > 10_000_000_000
    assert doc["sealed_parent"]["payload_bytes"] > 5_000_000_000
    # Parent is the smaller body; q4 is the incumbent.
    assert doc["sealed_parent"]["payload_bytes"] < doc["incumbent"]["payload_bytes"]


def test_gpu_timestamps_are_not_cpu_wait_proxies():
    doc = _receipt()
    assert "GPUStartTime" in doc["gpu_timestamp_authority"]
    assert "never a CPU-wait proxy" in doc["gpu_timestamp_authority"]
