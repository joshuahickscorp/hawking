"""The dispatch-size sweep must refuse a verdict on fewer than five points."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import dispatch_size_sweep as dss


def _point(
    mb: float,
    gb_s: float,
    *,
    dispatches: int = 1,
    encoders: int = 1,
    rows: int | None = None,
) -> dict:
    weight_bytes = int(round(mb * 1e6))
    gpu_ns = int(round(weight_bytes / gb_s))
    if rows is None:
        rows = max(2, int(weight_bytes / 1600))
    return {
        "label": f"mb_{mb}",
        "target_mb": mb,
        "weight_bytes": weight_bytes,
        "rows": rows,
        "cols": 5120,
        "n_source_tensors": max(1, int(round(mb / 27.85))),
        "dispatches": dispatches,
        "encoders": encoders,
        "command_buffers": 1,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [gpu_ns],
        "effective_gb_s": gb_s,
        "mb_per_dispatch": mb / dispatches,
    }


def _control(one_gb_s: float = 350.0, many_gb_s: float = 348.0) -> dict:
    mb = 334.2
    return {
        "total_weight_bytes": int(mb * 1e6),
        "n_unit_tensors": 12,
        "unit_rows": 17408,
        "one_large": _point(mb, one_gb_s, dispatches=1),
        "many_small": _point(mb, many_gb_s, dispatches=12),
        "max_abs_diff": 0.0,
        "bit_identical": True,
    }


def _raw(
    *,
    points: list[tuple[float, float]] | None = None,
    pack_ok: bool = True,
) -> dict:
    if points is None:
        points = [
            (5.0, 330.0),
            (20.0, 332.0),
            (50.0, 335.0),
            (100.0, 336.0),
            (200.0, 338.0),
            (337.7, 340.0),
            (700.0, 342.0),
        ]
    curve = [_point(mb, gb) for mb, gb in points]
    return {
        "schema": "hawking.future.dispatch_size_sweep.raw.v1",
        "kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
        "codec": "HGRAVF01 affine2 q2",
        "geometry": "tpr64_tg128",
        "projection": "mlp.gate_proj.weight",
        "group_size": 64,
        "bits": 2,
        "cols": 5120,
        "unit_rows": 17408,
        "unit_weight_bytes": 27_852_800,
        "bytes_per_row": 1600,
        "warmup": 5,
        "reps": 11,
        "pack_parity_max_abs_diff": 0.0 if pack_ok else 1e-3,
        "pack_parity_bit_identical": pack_ok,
        "curve": curve,
        "control": _control(),
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "arithmetic": "y = affine2_q2(W, x)",
    }


def test_refuses_a_verdict_on_fewer_than_five_points():
    four = [(5.0, 330.0), (20.0, 332.0), (50.0, 335.0), (337.7, 340.0)]
    with pytest.raises(dss.TooFewPoints) as caught:
        dss.judge([_point(mb, gb) for mb, gb in four])
    assert "5" in str(caught.value)
    with pytest.raises(dss.TooFewPoints):
        dss.measurement_from_raw(_raw(points=four))


def test_refuses_a_verdict_when_the_packed_unit_tensor_drifted():
    with pytest.raises(dss.PackNotIdentical):
        dss.measurement_from_raw(_raw(pack_ok=False))
    measured = dss.measurement_from_raw(_raw())
    measured["pack_parity_bit_identical"] = False
    measured["pack_parity_max_abs_diff"] = 0.5
    with pytest.raises(dss.PackNotIdentical):
        dss.build(measured)


def test_implicated_when_the_curve_enters_the_lm_head_regime():
    points = [
        (5.0, 320.0),
        (20.0, 332.0),
        (50.0, 380.0),
        (100.0, 430.0),
        (200.0, 470.0),
        (337.7, 497.4),
        (700.0, 520.0),
    ]
    curve = [_point(mb, gb) for mb, gb in points]
    assert dss.judge(curve) == dss.VERDICT_IMPLICATED
    doc = dss.build(dss.measurement_from_raw(_raw(points=points)))
    assert doc["verdict"] == dss.VERDICT_IMPLICATED
    assert doc["gb_s_at_lm_head_mb"] >= dss.IMPLICATE_GB_S
    assert doc["knee_mb"] is not None
    assert doc["keeps_rising_past_lm_head_mb"] is True
    assert "mechanism" in doc["finding"].lower()


def test_refuted_when_the_curve_stays_near_350():
    curve = [_point(mb, gb) for mb, gb in [
        (5.0, 328.0),
        (20.0, 332.0),
        (50.0, 334.0),
        (100.0, 335.0),
        (200.0, 336.0),
        (337.7, 338.0),
        (700.0, 340.0),
    ]]
    assert dss.judge(curve) == dss.VERDICT_REFUTED
    doc = dss.build(dss.measurement_from_raw(_raw()))
    assert doc["verdict"] == dss.VERDICT_REFUTED
    assert doc["gb_s_at_lm_head_mb"] < dss.REFUTE_CEILING_GB_S
    assert "DEAD" in doc["finding"]
    assert doc["remaining_candidates"]["favoured"] == "decode_alu_cost"


def test_a_5mb_overhead_dip_does_not_rescue_the_hypothesis():
    """Fixed launch cost at 5 MB is not the 350→497 gap. Saturation at ~375 is REFUTED."""
    points = [
        (5.0, 223.4),
        (20.0, 325.9),
        (50.0, 354.8),
        (100.0, 365.8),
        (200.0, 371.0),
        (337.7, 374.4),
        (700.0, 376.7),
    ]
    curve = [_point(mb, gb) for mb, gb in points]
    assert dss.judge(curve) == dss.VERDICT_REFUTED
    doc = dss.build(dss.measurement_from_raw(_raw(points=points)))
    assert doc["verdict"] == dss.VERDICT_REFUTED
    assert doc["gb_s_at_lm_head_mb"] == pytest.approx(374.4, abs=0.2)
    assert doc["knee_mb"] is None


def test_refuses_a_forced_binary_in_the_dead_band():
    # 415 GB/s at 338 MB is neither the 350 cluster nor the LM-head regime.
    curve = [_point(mb, gb) for mb, gb in [
        (5.0, 330.0),
        (20.0, 350.0),
        (50.0, 370.0),
        (100.0, 390.0),
        (200.0, 405.0),
        (337.7, 415.0),
        (700.0, 420.0),
    ]]
    with pytest.raises(dss.InconclusiveBandwidth) as caught:
        dss.judge(curve)
    assert "430" in str(caught.value)


def test_gb_s_is_bytes_over_gpu_ns():
    assert dss.effective_gb_s(350_000_000, 1_000_000) == 350.0
    weight = 27_852_800
    ns = int(round(weight / 344.1))
    got = dss.effective_gb_s(weight, ns)
    assert abs(got - 344.1) < 0.1


def test_receipt_records_the_curve_the_control_and_a_binary_verdict():
    doc = dss.build(dss.measurement_from_raw(_raw()))
    assert doc["schema"] == dss.SCHEMA
    assert doc["evidence_class"] == "DIAGNOSTIC_RELATIVE"
    assert doc["verdict"] in (dss.VERDICT_IMPLICATED, dss.VERDICT_REFUTED)
    assert len(doc["curve"]) >= dss.MIN_POINTS
    assert "effective_gb_s" in doc["curve"][0]
    assert "mb_per_dispatch" in doc["curve"][0]
    assert "one_large" in doc["control"]
    assert "many_small" in doc["control"]
    assert doc["pack_parity_bit_identical"] is True
    assert "perfect-locality" in doc["claim_boundary"]
    assert doc["gpu_authority"] is False


def test_record_writes_the_named_receipt(tmp_path: Path):
    dest = tmp_path / "DISPATCH_SIZE_SWEEP.json"
    measured = dss.measurement_from_raw(_raw())
    path = dss.record(measured, path=dest)
    assert path == dest
    doc = json.loads(dest.read_text())
    assert doc["verdict"] == dss.VERDICT_REFUTED
    assert len(doc["curve"]) >= 5


def test_record_without_a_measurement_refuses():
    with pytest.raises(dss.TooFewPoints):
        dss.record(None)


def test_committed_receipt_if_present_is_a_real_verdict():
    """Once recorded, the receipt must name the curve and a binary verdict."""
    if not dss.RECEIPT.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(dss.RECEIPT.read_text())
    assert doc["schema"] == dss.SCHEMA
    assert doc["evidence_class"] == "DIAGNOSTIC_RELATIVE"
    assert doc["verdict"] in (dss.VERDICT_IMPLICATED, dss.VERDICT_REFUTED)
    assert len(doc["curve"]) >= dss.MIN_POINTS
    assert doc["pack_parity_bit_identical"] is True
    assert (
        dss.judge(doc["curve"]) == doc["verdict"]
    )
