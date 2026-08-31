"""The ALU-roofline sidecar must refuse a verdict when the matched pair is incomplete."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import mlp_alu_roofline as mar


def _arm(gb_s: float, *, weight_bytes: int, label: str = "arm") -> dict:
    gpu_ns = int(round(weight_bytes / gb_s))
    return {
        "label": label,
        "kernel": "k",
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [gpu_ns],
        "dispatches": 3,
        "encoders": 1,
        "command_buffers": 1,
        "effective_gb_s": gb_s,
        "occupancy": {
            "threads_per_threadgroup": 128,
            "max_total_threads_per_threadgroup": 1024,
            "thread_execution_width": 32,
            "occupancy_of_max_threads": 0.125,
            "registers_per_thread": None,
        },
    }


def _organ(
    *,
    name: str = "mlp",
    prod_gb: float = 344.1,
    a_gb: float = 344.1,
    b_gb: float = 344.1,
    prod_bytes: int = 83_558_400,
    a_bytes: int | None = None,
    b_bytes: int | None = None,
    zero_gb: float | None = 8000.0,
    a_half_gb: float | None = None,
) -> dict:
    if a_bytes is None:
        a_bytes = prod_bytes
    if b_bytes is None:
        b_bytes = prod_bytes // 2
    if a_half_gb is None:
        a_half_gb = a_gb
    kernel = (
        "qwen_affine_q2_group32_matvec_geo_tpr64_tg128"
        if name == "mlp"
        else "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128"
    )
    out = {
        "organ": name,
        "kernel": kernel,
        "codec": "HGRAVF01" if name == "mlp" else "HQ30UQ4",
        "threads_per_threadgroup": 128,
        "bytes_per_thread_iteration": 38,
        "production": _arm(prod_gb, weight_bytes=prod_bytes, label="production"),
        "arm_a_stripped": _arm(a_gb, weight_bytes=a_bytes, label="arm_a_stripped"),
        "arm_b_halfk": _arm(b_gb, weight_bytes=b_bytes, label="arm_b_halfk"),
    }
    if zero_gb is not None:
        # Tiny time (huge GB/s) = launch floor against the same byte count.
        out["zero_load"] = _arm(zero_gb, weight_bytes=prod_bytes, label="zero_load")
    if a_half_gb is not None:
        out["arm_a_halfk"] = _arm(a_half_gb, weight_bytes=b_bytes, label="arm_a_halfk")
    return out


def _raw(**kwargs) -> dict:
    mlp = kwargs.pop("mlp", None) or _organ(name="mlp")
    dn = kwargs.pop("deltanet", None) or _organ(
        name="deltanet", prod_bytes=44_564_480, prod_gb=360.0, a_gb=360.0, b_gb=360.0
    )
    return {
        "schema": "hawking.future.mlp_alu_roofline.raw.v1",
        "layer": 0,
        "warmup": 5,
        "reps": 11,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "absolute_gb_s_are_measured_under_load": True,
        "concurrent_load": {"loadavg": "{ 1.0 1.0 1.0 }"},
        "mlp": mlp,
        "deltanet": dn,
        **kwargs,
    }


def test_raises_when_either_arm_is_missing():
    raw = _raw()
    del raw["mlp"]["arm_a_stripped"]
    with pytest.raises(mar.MissingArm) as caught:
        mar.measurement_from_raw(raw)
    assert "arm_a_stripped" in str(caught.value)

    raw = _raw()
    del raw["deltanet"]["arm_b_halfk"]
    with pytest.raises(mar.MissingArm) as caught:
        mar.measurement_from_raw(raw)
    assert "arm_b_halfk" in str(caught.value)

    raw = _raw()
    del raw["mlp"]
    with pytest.raises(mar.MissingArm) as caught:
        mar.measurement_from_raw(raw)
    assert "mlp" in str(caught.value)

    measured = mar.measurement_from_raw(_raw())
    del measured["deltanet"]
    with pytest.raises(mar.MissingArm):
        mar.build(measured)


def test_raises_when_stripped_bytes_do_not_equal_production_bytes():
    mlp = _organ(name="mlp", a_bytes=83_558_400 // 2)
    with pytest.raises(mar.ByteMismatch) as caught:
        mar.measurement_from_raw(_raw(mlp=mlp))
    assert "weight_bytes" in str(caught.value)
    assert "ARM A" in str(caught.value)


def test_raises_on_zero_gpu_ns():
    mlp = _organ(name="mlp")
    mlp["production"]["gpu_ns_median"] = 0
    with pytest.raises(mar.EmptyGpuSample):
        mar.measurement_from_raw(_raw(mlp=mlp))


def test_memory_system_bound_when_a_stays():
    # A stays at 344; B halves time with bytes (linear) — bandwidth-like.
    mlp = _organ(name="mlp", prod_gb=344.1, a_gb=348.0, b_gb=340.0)
    dn = _organ(
        name="deltanet",
        prod_bytes=44_564_480,
        prod_gb=360.0,
        a_gb=365.0,
        b_gb=355.0,
    )
    doc = mar.build(mar.measurement_from_raw(_raw(mlp=mlp, deltanet=dn)))
    assert doc["verdict"] == mar.VERDICT_MEM
    assert doc["mlp"]["verdict"] == mar.VERDICT_MEM
    assert doc["deltanet"]["verdict"] == mar.VERDICT_MEM
    assert "memory" in doc["finding"].lower() or "ceiling" in doc["finding"].lower()


def test_alu_bound_when_a_jumps_and_b_is_sublinear_and_loads_survived():
    # A: 344 -> 520 (1.51x). B: same time as production despite half bytes
    # (time_ratio=1, byte_ratio=0.5, scale=2) -> sub-linear.
    # stripped-half is faster than stripped so loads scale with bytes.
    mlp = _organ(
        name="mlp",
        prod_gb=344.1,
        a_gb=520.0,
        b_gb=172.0,          # half bytes, same time => gb_s halves
        a_half_gb=520.0,     # half bytes, half time of original stripped? wait
        zero_gb=8000.0,
    )
    # For loads_survived.time_scales_with_bytes: a_half_ns < 0.85 * a_ns.
    # a_gb=520 on full bytes. a_half_gb=520 on half bytes => half the time. Good.
    dn = _organ(
        name="deltanet",
        prod_bytes=44_564_480,
        prod_gb=360.0,
        a_gb=540.0,
        b_gb=180.0,
        a_half_gb=540.0,
        zero_gb=8000.0,
    )
    doc = mar.build(mar.measurement_from_raw(_raw(mlp=mlp, deltanet=dn)))
    assert doc["mlp"]["judgement"]["arm_a_jump"] is True
    assert doc["mlp"]["judgement"]["arm_b_sublinear"] is True
    assert doc["mlp"]["judgement"]["loads_survived"]["survived"] is True
    assert doc["verdict"] == mar.VERDICT_ALU
    assert "arithmetic ceiling" in doc["finding"].lower() or "ALU" in doc["finding"]


def test_mixed_when_a_jumps_but_b_is_linear():
    # A jumps, B halves time with bytes (linear) -> MIXED per the contract.
    mlp = _organ(name="mlp", prod_gb=344.1, a_gb=520.0, b_gb=344.1, a_half_gb=520.0)
    dn = _organ(
        name="deltanet",
        prod_bytes=44_564_480,
        prod_gb=360.0,
        a_gb=540.0,
        b_gb=360.0,
        a_half_gb=540.0,
    )
    doc = mar.build(mar.measurement_from_raw(_raw(mlp=mlp, deltanet=dn)))
    assert doc["mlp"]["judgement"]["arm_a_jump"] is True
    assert doc["mlp"]["judgement"]["arm_b_linear"] is True
    assert doc["verdict"] == mar.VERDICT_MIXED


def test_mixed_when_organs_disagree():
    mlp = _organ(name="mlp", prod_gb=344.1, a_gb=348.0, b_gb=340.0)
    dn = _organ(
        name="deltanet",
        prod_bytes=44_564_480,
        prod_gb=360.0,
        a_gb=540.0,
        b_gb=180.0,
        a_half_gb=540.0,
    )
    doc = mar.build(mar.measurement_from_raw(_raw(mlp=mlp, deltanet=dn)))
    assert doc["mlp"]["verdict"] == mar.VERDICT_MEM
    assert doc["deltanet"]["verdict"] == mar.VERDICT_ALU
    assert doc["verdict"] == mar.VERDICT_MIXED


def test_gb_s_is_bytes_over_gpu_ns():
    assert mar.effective_gb_s(350_000_000, 1_000_000) == 350.0
    with pytest.raises(mar.EmptyGpuSample):
        mar.effective_gb_s(100, 0)


def test_record_writes_both_organs_and_refuses_none(tmp_path: Path):
    dest = tmp_path / "MLP_ALU_ROOFLINE.json"
    measured = mar.measurement_from_raw(_raw())
    path = mar.record(measured, path=dest)
    assert path == dest
    doc = json.loads(dest.read_text())
    assert doc["schema"] == mar.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["verdict"] in (mar.VERDICT_ALU, mar.VERDICT_MEM, mar.VERDICT_MIXED)
    assert "effective_gb_s" in doc["mlp"]["production"]
    assert "effective_gb_s" in doc["mlp"]["arm_a_stripped"]
    assert "effective_gb_s" in doc["mlp"]["arm_b_halfk"]
    assert "effective_gb_s" in doc["deltanet"]["production"]
    assert "effective_gb_s" in doc["deltanet"]["arm_a_stripped"]
    assert "effective_gb_s" in doc["deltanet"]["arm_b_halfk"]
    assert doc["absolute_gb_s_are_measured_under_load"] is True
    assert doc["mlp"]["threads_per_threadgroup"] == 128
    assert "decode_tax" in doc["mlp"]
    with pytest.raises(mar.MissingArm):
        mar.record(None)


def test_committed_receipt_if_present_is_a_real_verdict():
    if not mar.RECEIPT.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(mar.RECEIPT.read_text())
    assert doc["schema"] == mar.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["verdict"] in (mar.VERDICT_ALU, mar.VERDICT_MEM, mar.VERDICT_MIXED)
    for name in ("mlp", "deltanet"):
        organ = doc[name]
        assert organ["production"]["effective_gb_s"] > 0
        assert organ["arm_a_stripped"]["effective_gb_s"] > 0
        assert organ["arm_b_halfk"]["effective_gb_s"] > 0
        assert organ["arm_a_stripped"]["weight_bytes"] == organ["production"]["weight_bytes"]
        assert organ["verdict"] in (mar.VERDICT_ALU, mar.VERDICT_MEM, mar.VERDICT_MIXED)
        # Re-judging the recorded numbers must not disagree with the seal.
        assert mar.judge_organ(organ, name)["verdict"] == organ["verdict"]
    assert mar.judge({"mlp": doc["mlp"], "deltanet": doc["deltanet"]}) == doc["verdict"]
    assert doc["absolute_gb_s_are_measured_under_load"] is True
