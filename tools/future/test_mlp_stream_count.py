"""Stream-count sidecar: refuse a verdict when 38 B/iter is not held."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import mlp_stream_count as msc


WEIGHT = 83_558_400


def _rung(
    ident: str,
    gb_s: float,
    *,
    family: str = "stream_ladder",
    weight_bytes: int = WEIGHT,
    bytes_per_stream: list[int] | None = None,
    stream_count: int | None = None,
) -> dict:
    spec = msc.RUNG_SPEC.get(ident)
    streams = bytes_per_stream if bytes_per_stream is not None else (
        list(spec["bytes_per_stream"]) if spec else [2, 2, 2, 32]
    )
    count = stream_count if stream_count is not None else (
        spec["stream_count"] if spec else len(streams)
    )
    gpu_ns = int(round(weight_bytes / gb_s)) if gb_s > 0 else 0
    return {
        "id": ident,
        "kernel": f"k_{ident}",
        "family": family,
        "stream_count": count,
        "bytes_per_stream": streams,
        "bytes_per_thread_iteration": sum(streams),
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [gpu_ns],
        "gpu_us_median": gpu_ns / 1e3,
        "effective_gb_s": gb_s,
        "dispatches": 3,
        "encoders": 1,
        "command_buffers": 1,
        "threads_per_threadgroup": 128,
        "access_pattern": {
            "operand_alignment_bytes": 2,
            "per_thread_col_stride": 512,
            "codes_contiguous_per_thread": ident == "stride_contig",
            "activation_reused_across_rows": ident != "pack_38",
        },
    }


def _raw(
    *,
    mlp: float = 500.0,
    dn: float = 900.0,
    mid: float = 880.0,
    pack6: float = 950.0,
    pack38: float = 1000.0,
    align2: float | None = None,
    align4: float = 510.0,
    align16: float = 505.0,
    stride: float = 500.0,
    zero: float = 8000.0,
    half: float | None = None,
) -> dict:
    if align2 is None:
        align2 = mlp
    if half is None:
        half = mlp  # same GB/s on half bytes => half the time, loads scale
    rungs = [
        _rung("mlp_2_2_2_32", mlp),
        _rung("dn_4_2_32", dn),
        _rung("mid_2_4_32", mid),
        _rung("pack_6_32", pack6),
        _rung("pack_38", pack38),
    ]
    alignment = [
        _rung("align_2", align2, family="alignment", bytes_per_stream=[2, 2, 2, 32], stream_count=4),
        _rung("align_4", align4, family="alignment", bytes_per_stream=[2, 2, 2, 32], stream_count=4),
        _rung("align_16", align16, family="alignment", bytes_per_stream=[2, 2, 2, 32], stream_count=4),
        _rung("stride_contig", stride, family="stride", bytes_per_stream=[2, 2, 2, 32], stream_count=4),
    ]
    return {
        "schema": "hawking.future.mlp_stream_count.raw.v1",
        "layer": 0,
        "warmup": 5,
        "reps": 11,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "absolute_gb_s_are_measured_under_load": True,
        "bytes_per_thread_iteration_held": 38,
        "pre_registered_interpretation": msc.PRE_REGISTERED,
        "concurrent_load": {"loadavg": "{ 1.0 1.0 1.0 }"},
        "weight_bytes": WEIGHT,
        "dispatches": 3,
        "rungs": rungs,
        "alignment": alignment,
        "zero_load": _rung("zero_load", zero, family="dce", bytes_per_stream=[2, 2, 2, 32], stream_count=0),
        "halfk": _rung(
            "mlp_2_2_2_32_halfk",
            half,
            family="dce",
            weight_bytes=WEIGHT // 2,
            bytes_per_stream=[2, 2, 2, 32],
            stream_count=4,
        ),
    }


def test_raises_if_any_rung_byte_count_differs():
    raw = _raw()
    raw["rungs"][0]["bytes_per_thread_iteration"] = 36
    raw["rungs"][0]["bytes_per_stream"] = [2, 2, 0, 32]
    with pytest.raises(msc.ByteMismatch) as caught:
        msc.measurement_from_raw(raw)
    assert "38" in str(caught.value)

    raw = _raw()
    raw["rungs"][3]["bytes_per_stream"] = [8, 32]
    raw["rungs"][3]["bytes_per_thread_iteration"] = 40
    with pytest.raises(msc.ByteMismatch):
        msc.measurement_from_raw(raw)

    raw = _raw()
    raw["alignment"][1]["bytes_per_thread_iteration"] = 40
    raw["alignment"][1]["bytes_per_stream"] = [4, 4, 0, 32]
    with pytest.raises(msc.ByteMismatch):
        msc.measurement_from_raw(raw)


def test_raises_when_a_ladder_rung_is_missing():
    raw = _raw()
    raw["rungs"] = [r for r in raw["rungs"] if r["id"] != "pack_6_32"]
    with pytest.raises(msc.MissingRung) as caught:
        msc.measurement_from_raw(raw)
    assert "pack_6_32" in str(caught.value)

    raw = _raw()
    raw["alignment"] = [r for r in raw["alignment"] if r["id"] != "align_4"]
    with pytest.raises(msc.MissingRung):
        msc.measurement_from_raw(raw)


def test_raises_on_zero_gpu_ns():
    raw = _raw()
    raw["rungs"][0]["gpu_ns_median"] = 0
    with pytest.raises(msc.EmptyGpuSample):
        msc.measurement_from_raw(raw)


def test_every_fixture_rung_is_exactly_38():
    raw = _raw()
    measured = msc.measurement_from_raw(raw)
    assert len(measured["rungs"]) >= 5
    for ident, row in measured["rungs"].items():
        assert row["bytes_per_thread_iteration"] == 38, ident
        assert sum(row["bytes_per_stream"]) == 38, ident
    for ident, row in measured["alignment"].items():
        assert row["bytes_per_thread_iteration"] == 38, ident


def test_stream_count_bound_when_monotonic_merge():
    doc = msc.build(msc.measurement_from_raw(_raw()))
    assert doc["verdict"] == msc.VERDICT_STREAM
    assert doc["judgement"]["monotone_merge"] is True
    assert doc["judgement"]["same_mlp_dn"] is False
    assert "packing" in doc["finding"].lower() or "STREAM_COUNT_BOUND" in doc["finding"]


def test_alignment_bound_when_222_equals_42_and_align4_lifts():
    raw = _raw(mlp=500.0, dn=505.0, mid=502.0, pack6=510.0, pack38=508.0, align4=900.0, align16=880.0)
    doc = msc.build(msc.measurement_from_raw(raw))
    assert doc["verdict"] == msc.VERDICT_ALIGN
    assert doc["judgement"]["same_mlp_dn"] is True
    assert doc["judgement"]["align_lifts"] is True


def test_alignment_bound_when_width_not_count():
    # 4+2 jumps; 2+4 stays with 2+2+2. Count is 3 on both 4+2 and 2+4.
    raw = _raw(mlp=500.0, dn=900.0, mid=510.0, pack6=920.0, pack38=950.0, align4=505.0)
    doc = msc.build(msc.measurement_from_raw(raw))
    assert doc["verdict"] == msc.VERDICT_ALIGN
    assert doc["judgement"]["width_not_count"] is True


def test_not_stream_count_when_222_equals_42_and_align_does_not_lift():
    raw = _raw(mlp=500.0, dn=510.0, mid=505.0, pack6=508.0, pack38=490.0, align4=512.0, align16=507.0)
    doc = msc.build(msc.measurement_from_raw(raw))
    assert doc["verdict"] == msc.VERDICT_NOT
    assert doc["judgement"]["next_candidate"]


def test_mixed_when_merge_is_not_monotonic():
    raw = _raw(mlp=500.0, dn=900.0, mid=880.0, pack6=400.0, pack38=350.0, align4=510.0)
    doc = msc.build(msc.measurement_from_raw(raw))
    assert doc["verdict"] == msc.VERDICT_MIXED
    assert doc["judgement"]["monotone_merge"] is False


def test_mixed_when_loads_do_not_survive():
    raw = _raw()
    raw["zero_load"]["gpu_ns_median"] = raw["rungs"][0]["gpu_ns_median"]
    raw["zero_load"]["effective_gb_s"] = raw["rungs"][0]["effective_gb_s"]
    raw["halfk"]["gpu_ns_median"] = raw["rungs"][0]["gpu_ns_median"]
    raw["halfk"]["weight_bytes"] = WEIGHT
    raw["halfk"]["effective_gb_s"] = raw["rungs"][0]["effective_gb_s"]
    doc = msc.build(msc.measurement_from_raw(raw))
    assert doc["loads_survived"]["survived"] is False
    assert doc["verdict"] == msc.VERDICT_MIXED


def test_gb_s_is_bytes_over_gpu_ns():
    assert msc.effective_gb_s(350_000_000, 1_000_000) == 350.0
    with pytest.raises(msc.EmptyGpuSample):
        msc.effective_gb_s(100, 0)


def test_record_writes_curve_and_refuses_none(tmp_path: Path):
    dest = tmp_path / "MLP_STREAM_COUNT.json"
    measured = msc.measurement_from_raw(_raw())
    path = msc.record(measured, path=dest)
    assert path == dest
    doc = json.loads(dest.read_text())
    assert doc["schema"] == msc.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["verdict"] in msc.VERDICTS
    assert doc["bytes_per_thread_iteration_held"] == 38
    assert doc["pre_registered_interpretation"]["registered_before_measurement"] is True
    assert len(doc["rungs"]) >= 5
    for ident in msc.STREAM_RUNG_IDS:
        assert doc["rungs"][ident]["bytes_per_thread_iteration"] == 38
        assert "effective_gb_s" in doc["rungs"][ident]
        assert "gpu_us_median" in doc["rungs"][ident]
        assert "access_pattern" in doc["rungs"][ident]
    for ident in msc.ALIGN_IDS:
        assert doc["alignment"][ident]["bytes_per_thread_iteration"] == 38
    assert doc["absolute_gb_s_are_measured_under_load"] is True
    assert "packing_cost" in doc
    with pytest.raises(msc.MissingRung):
        msc.record(None)


def test_committed_receipt_if_present_is_a_real_verdict():
    if not msc.RECEIPT.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(msc.RECEIPT.read_text())
    assert doc["schema"] == msc.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["verdict"] in msc.VERDICTS
    assert doc["bytes_per_thread_iteration_held"] == 38
    assert len(doc["rungs"]) >= 5
    for ident in msc.STREAM_RUNG_IDS:
        row = doc["rungs"][ident]
        assert row["bytes_per_thread_iteration"] == 38
        assert sum(row["bytes_per_stream"]) == 38
        assert row["effective_gb_s"] > 0
        assert row["gpu_ns_median"] > 0
    for ident in msc.ALIGN_IDS:
        assert doc["alignment"][ident]["bytes_per_thread_iteration"] == 38
        assert doc["alignment"][ident]["effective_gb_s"] > 0
    assert "loadavg" in (doc.get("concurrent_load") or {})
    measured = {
        "rungs": doc["rungs"],
        "alignment": doc["alignment"],
        "stride": doc.get("stride") or {},
        "loads_survived": doc.get("loads_survived") or {"survived": True},
    }
    assert msc.judge(measured)["verdict"] == doc["verdict"]
    assert doc["pre_registered_interpretation"]["registered_before_measurement"] is True
    assert doc["absolute_gb_s_are_measured_under_load"] is True
