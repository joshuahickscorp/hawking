"""The DeltaNet organ decomposition must raise rather than absorb a hole."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import deltanet_organ_decompose as dod


def _arm(gb_s: float, *, weight_bytes: int, label: str = "arm", gpu_ns: int | None = None) -> dict:
    ns = gpu_ns if gpu_ns is not None else int(round(weight_bytes / gb_s)) if gb_s else 1
    return {
        "label": label,
        "kernel": "k",
        "weight_bytes": weight_bytes,
        "gpu_ns_median": ns,
        "gpu_ns_reps": [ns],
        "dispatches": 48 if label != "production" else 48,
        "encoders": 1,
        "command_buffers": 1,
        "effective_gb_s": gb_s,
    }


def _alu_organ(name: str, prod_gb: float, a_gb: float, b_gb: float, bytes_: int) -> dict:
    half = bytes_ // 2
    return {
        "organ": name,
        "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128",
        "codec": "HQ30UQ4 group64",
        "projection": {"name": name, "weight_bytes": bytes_},
        "production": _arm(prod_gb, weight_bytes=bytes_, label="production"),
        "arm_a_stripped": _arm(a_gb, weight_bytes=bytes_, label="arm_a_stripped"),
        "arm_b_halfk": _arm(b_gb, weight_bytes=half, label="arm_b_halfk"),
        "arm_a_halfk": _arm(a_gb, weight_bytes=half, label="arm_a_halfk"),
        "zero_load": _arm(8000.0, weight_bytes=bytes_, label="zero_load"),
    }


def _fam(ns: int, bytes_: int, dispatches: int, kernel: str = "k") -> dict:
    gb_s = (bytes_ / ns) if ns and bytes_ else 0.0
    return {
        "label": kernel,
        "kernel": kernel,
        "weight_bytes": bytes_,
        "gpu_ns_median": ns,
        "gpu_ns_reps": [ns],
        "dispatches": dispatches,
        "encoders": 1,
        "command_buffers": 1,
        "effective_gb_s": gb_s,
    }


def _raw(**overrides) -> dict:
    # A made-up but internally consistent organ: 8.000 ms, partition sums to 7.600.
    families = {
        "dn_as_executed": _fam(8_000_000, dod.CITED_ORGAN_BYTES, 337, "encode_deltanet"),
        "dn_input_rmsnorm": _fam(40_000, dod.INPUT_RMS_BYTES, 1, "rmsnorm"),
        "dn_inproj": _fam(3_600_000, dod.QKVZ_BYTES + dod.BA_BYTES, 48, "pair_concat"),
        "rearrange_48": _fam(340_000, dod.CONV_BYTES, 48, "rearrange"),
        "ba_to_decay_48": _fam(80_000, dod.A_LOG_DT_BIAS_BYTES, 48, "ba_to_decay"),
        "gated_delta_unfused": _fam(1_600_000, dod.REC_STATE_RW, 48, "gated_delta"),
        "gated_rmsnorm_48": _fam(190_000, dod.NORM_LINEAR_ATTN_BYTES, 48, "gated_rmsnorm"),
        "dn_out_proj": _fam(1_350_000, dod.OUT_BYTES, 48, "out_proj"),
        "dn_residual_rmsnorm": _fam(400_000, dod.POST_ATTN_DN_BYTES, 48, "residual"),
        "dn_qkvz": _fam(3_560_000, dod.QKVZ_BYTES, 48, "qkvz"),
        "dn_ba": _fam(200_000, dod.BA_BYTES, 48, "ba"),
        "gated_delta_fused_ba": _fam(1_590_000, dod.REC_STATE_RW, 48, "fused_ba"),
        "gated_delta_widen_f4": _fam(890_000, dod.REC_STATE_RW, 48, "f4"),
        "rec_state_f32_stream": _fam(220_000, dod.REC_STATE_RESIDENT, 1, "stream"),
        "organ_incomplete_missing_out_proj": _fam(6_360_000, dod.QKVZ_BYTES + dod.BA_BYTES, 288, "incomplete"),
        "noop_empty": _fam(0, 0, 0, "empty"),
    }
    doc = {
        "schema": "hawking.future.deltanet_organ_decompose.raw.v1",
        "layer": 0,
        "warmup": 5,
        "reps": 11,
        "session_warmup": 2,
        "session_reps": 7,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "absolute_gb_s_are_measured_under_load": True,
        "concurrent_load": {"loadavg": "{ 1.0 1.0 1.0 }"},
        "production_fusions": {
            "mlp": "GateUpSwiglu",
            "fuse_gqa_qkv": True,
            "fuse_dn_inproj": True,
            "fuse_add_rmsnorm": True,
            "fuse_ba_delta": False,
        },
        "alu_matched_pair": {
            "in_proj_qkvz": _alu_organ("in_proj_qkvz", 600.9, 943.2, 545.1, 44_564_480),
            "out_proj": _alu_organ("out_proj", 580.0, 900.0, 540.0, 16_711_720),
            "in_proj_ba": _alu_organ("in_proj_ba", 80.0, 90.0, 70.0, 261_160),
        },
        "families": families,
        "as_executed_named": {
            "dispatches": 337,
            "kernel_histogram": [{"kernel": "pair", "count": 48}],
        },
        "dense_w_materialized": 0,
    }
    doc.update(overrides)
    if "families" in overrides:
        doc["families"] = overrides["families"]
    return doc


def test_raises_when_a_partition_family_is_missing():
    raw = _raw()
    del raw["families"]["gated_delta_unfused"]
    with pytest.raises(dod.UnreconciledDecomposition) as caught:
        dod.measurement_from_raw(raw)
    assert "gated_delta_unfused" in str(caught.value)
    assert "missing" in str(caught.value)


def test_raises_when_organ_is_missing():
    raw = _raw()
    del raw["families"]["dn_as_executed"]
    with pytest.raises(dod.UnreconciledDecomposition) as caught:
        dod.measurement_from_raw(raw)
    assert "dn_as_executed" in str(caught.value)


def test_raises_when_a_partition_family_is_zero_ns():
    raw = _raw()
    raw["families"]["dn_out_proj"]["gpu_ns_median"] = 0
    with pytest.raises(dod.UnreconciledDecomposition) as caught:
        dod.measurement_from_raw(raw)
    assert "zero-ns" in str(caught.value)
    assert "dn_out_proj" in str(caught.value)


def test_raises_on_empty_organ():
    with pytest.raises(dod.EmptyGpuSample):
        dod.reconcile(0, {k: 1 for k in dod.PARTITION})


def test_raises_when_families_object_is_absent():
    with pytest.raises(dod.UnreconciledDecomposition):
        dod.measurement_from_raw({"schema": "x"})


def test_residual_is_named_not_absorbed():
    measured = dod.measurement_from_raw(_raw())
    recon = measured["reconciliation"]
    # 8.000 ms organ, 7.600 ms partition → 0.400 ms residual, kept as a field.
    assert recon["organ_ms"] == pytest.approx(8.0, abs=1e-6)
    assert recon["sum_partition_ms"] == pytest.approx(7.6, abs=1e-3)
    assert recon["residual_ms"] == pytest.approx(0.4, abs=1e-3)
    assert "not absorbed" in recon["residual_name"]
    assert "organ_cb_slower_than_isolated_family_sum" in recon["residual_name"]
    assert recon["within_tolerance"] is True
    # The builder must still carry the residual; a receipt without it is a lie.
    doc = dod.build(measured)
    assert doc["reconciliation"]["residual_ms"] == recon["residual_ms"]
    assert "residual" in doc["finding"].lower()


def test_does_not_scale_parts_to_force_8_227():
    measured = dod.measurement_from_raw(_raw())
    cited = measured["cited_organ"]
    # 8.000 vs 8.227: the gap is named. Families are not multiplied to match.
    assert cited["this_run_organ_ms"] == pytest.approx(8.0, abs=1e-6)
    assert cited["ms"] == 8.227
    assert cited["gap_ms"] != 0
    assert "8.227" in cited["gap_name"]
    inproj = next(r for r in measured["ranked"] if r["id"] == "dn_inproj")
    assert inproj["ms"] == pytest.approx(3.6, abs=1e-4)


def test_alu_byte_mismatch_raises():
    raw = _raw()
    raw["alu_matched_pair"]["in_proj_qkvz"]["arm_a_stripped"]["weight_bytes"] = 1
    with pytest.raises(dod.ByteMismatch):
        dod.measurement_from_raw(raw)


def test_gb_s_is_bytes_over_gpu_ns():
    assert dod.effective_gb_s(350_000_000, 1_000_000) == 350.0
    with pytest.raises(dod.EmptyGpuSample):
        dod.effective_gb_s(100, 0)


def test_ranked_list_carries_limiter_and_measurement():
    doc = dod.build(dod.measurement_from_raw(_raw()))
    ids = [r["id"] for r in doc["ranked"] if r["in_partition"]]
    assert ids[0] == "dn_inproj"
    assert "gated_delta_unfused" in ids
    assert "dn_out_proj" in ids
    for row in doc["ranked"]:
        if not row["in_partition"]:
            continue
        assert row["limiter"]
        assert row["measurement_that_forced_it"]["gpu_ns"] == row["gpu_ns"]
        assert row["why"]
    # State update is not byte-bound: stream is faster, f4 jumped.
    delta = next(r for r in doc["ranked"] if r["id"] == "gated_delta_unfused")
    assert delta["limiter"] in (dod.VERDICT_LAYOUT, dod.VERDICT_SERIAL)
    # Small kernels are latency/elementwise, not 360 GB/s.
    small = next(r for r in doc["ranked"] if r["id"] == "gated_rmsnorm_48")
    assert small["limiter"] in (dod.VERDICT_LATENCY, dod.VERDICT_ELEMENTWISE)
    assert doc["largest_addressable_cost"]["id"] == "dn_inproj"
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert "8.227" in doc["finding"]


def test_record_refuses_none(tmp_path: Path):
    dest = tmp_path / "DELTANET_ORGAN_DECOMPOSE.json"
    measured = dod.measurement_from_raw(_raw())
    path = dod.record(measured, path=dest)
    assert path == dest
    doc = json.loads(dest.read_text())
    assert doc["schema"] == dod.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["reconciliation"]["residual_name"]
    with pytest.raises(dod.UnreconciledDecomposition):
        dod.record(None)


def test_committed_receipt_if_present_reconciles():
    if not dod.RECEIPT.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(dod.RECEIPT.read_text())
    assert doc["schema"] == dod.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    recon = doc["reconciliation"]
    organ = recon["organ_ns"]
    summed = sum(recon["parts_ns"][k] for k in dod.PARTITION)
    assert summed + recon["residual_ns"] == organ
    assert recon["residual_name"]
    for row in doc["ranked"]:
        if row["in_partition"]:
            assert row["gpu_ns"] > 0
            assert row["limiter"]
            assert row["measurement_that_forced_it"]
    assert doc["absolute_gb_s_are_measured_under_load"] is True
    assert doc["cited_organ"]["ms"] == 8.227
