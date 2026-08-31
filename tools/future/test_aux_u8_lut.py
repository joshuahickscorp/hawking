"""Tests for the LUT-decode u8-aux consumer.

Load-bearing:
  * the LUT path does not materialize the f16 aux (CPU ledger + Metal source)
  * expanding u8 → f16 aux and feeding the ordinary kernel is refused
  * LUT kernels do not call exp; fill/exp-variant kernels do
  * A/B refuses to report GB/s without measured gpu_ns and loadavg
  * FMA/byte by class is reported for incumbent, exp-variant, LUT-variant
  * executable_economics bills both ways; LUT extra FMA is 0 so the billed
    net sign is the byte-lever sign
  * the module refuses to report a speedup without a byte comparison proving
    LUT output is unchanged from the exp-variant
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.future import aux_u8_lut as lut
from tools.future import aux_u8_native as n8
from tools.future import executable_economics as ee


def _tiny_packed(rows: int = 8, cols: int = 64, seed: int = 0) -> n8.PackedU8Aux:
    rng = np.random.default_rng(seed)
    gpr = cols // n8.GROUP
    q = rng.integers(0, 4, size=(rows, gpr, n8.GROUP)).astype(np.uint8)
    scale = rng.random((rows, gpr)).astype(np.float32) * 0.02 + 0.005
    bias = -1.5 * scale + rng.normal(scale=0.001, size=scale.shape).astype(np.float32)
    return n8.pack_u8_aux(q, scale, bias, rows=rows, cols=cols)


def _arm(gb_s: float, *, weight_bytes: int, label: str, **extra: object) -> dict:
    gpu_ns = int(round(weight_bytes / gb_s))
    out = {
        "label": label,
        "kernel": "k",
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [gpu_ns],
        "dispatches": 3,
        "encoders": 1,
        "command_buffers": 1,
        "effective_gb_s": gb_s,
        "aux_bytes": weight_bytes // 10,
        "code_bytes": weight_bytes - weight_bytes // 10,
    }
    out.update(extra)
    return out


def _placement(gb_s: float, *, weight_bytes: int, name: str, bytes_equal: bool = True) -> dict:
    return _arm(
        gb_s,
        weight_bytes=weight_bytes,
        label=f"lut_u8_{name}",
        placement=name,
        materializes_f16_aux=False,
        lut_table_bytes=6144,
        vs_exp={"bytes_equal": bytes_equal, "n_mismatch": 0 if bytes_equal else 3},
    )


def _raw(**kwargs) -> dict:
    inc_bytes = 83_558_400
    nat_bytes = 75_202_560
    base = {
        "schema": "hawking.future.aux_u8_lut.raw.v1",
        "layer": 3,
        "warmup": 5,
        "reps": 11,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "absolute_gb_s_are_measured_under_load": True,
        "concurrent_load": {"loadavg": "{ 6.87 7.62 8.56 }"},
        "incumbent": _arm(330.0, weight_bytes=inc_bytes, label="incumbent_f16_aux"),
        "exp_variant": {
            **_arm(300.0, weight_bytes=nat_bytes, label="native_u8_aux_exp"),
            "materializes_f16_aux": False,
            "endpoint_bytes": 48,
        },
        "lut_variant": {
            **_arm(320.0, weight_bytes=nat_bytes, label="lut_u8_threadgroup"),
            "materializes_f16_aux": False,
            "placement": "threadgroup",
            "output_bytes_equal_vs_exp": True,
            "lut_table_bytes": 6144,
        },
        "placements": {
            "constant": _placement(310.0, weight_bytes=nat_bytes, name="constant"),
            "threadgroup": _placement(320.0, weight_bytes=nat_bytes, name="threadgroup"),
            "device": _placement(315.0, weight_bytes=nat_bytes, name="device"),
        },
        "chosen_placement": "threadgroup",
        "lut_vs_exp_output": {
            "bytes_equal": True,
            "chosen": {"bytes_equal": True, "n_mismatch": 0},
        },
        "output_cosine_mean": 0.999,
        "native_consumer": {
            "materializes_f16_aux": False,
            "scale_buffer": "u8",
            "bias_buffer": "u8",
        },
        "projections": [
            {
                "name": "language_model.model.layers.3.mlp.down_proj.weight",
                "output_relfro_exp_vs_incumbent": 0.0263,
                "output_relfro_lut_threadgroup_vs_incumbent": 0.0263,
            }
        ],
    }
    base.update(kwargs)
    return base


def test_lut_matvec_does_not_materialize_f16_aux():
    packed = _tiny_packed(rows=16, cols=256)
    n_groups = packed.n_groups
    x = np.random.default_rng(2).normal(size=(4, 256)).astype(np.float32)
    ledger = n8.AuxAllocLedger()
    y = lut.lut_matvec_u8(x, packed, ledger=ledger)
    assert y.shape == (4, 16)
    assert ledger.wrote_f16_aux_buffer is False
    assert ledger.max_decoded_aux_elems == 16 * 2
    assert ledger.max_decoded_aux_elems < 2 * n_groups
    assert (
        lut.classify_consumer(
            materializes_f16_aux=False,
            binds_u8_aux=True,
            peak_decoded_aux_elems=ledger.max_decoded_aux_elems,
            n_groups=n_groups,
        )
        == lut.DIRECT_CONSUME
    )


def test_remat_then_ordinary_is_refused():
    packed = _tiny_packed()
    ledger = n8.AuxAllocLedger()
    x = np.ones(packed.cols, dtype=np.float32)
    _ = n8.remat_then_ordinary_matvec(x, packed, ledger=ledger)
    assert ledger.wrote_f16_aux_buffer is True
    with pytest.raises(n8.MaterializeF16AuxRefuse):
        lut.classify_consumer(
            materializes_f16_aux=True,
            binds_u8_aux=False,
            peak_decoded_aux_elems=ledger.max_decoded_aux_elems,
            n_groups=packed.n_groups,
        )


def test_lut_matches_exp_variant_numerically_on_tiny():
    packed = _tiny_packed(rows=8, cols=64, seed=3)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, 64)).astype(np.float32)
    y_lut = lut.lut_matvec_u8(x, packed)
    y_exp = n8.native_matvec_u8(x, packed)
    assert y_lut.tobytes() == y_exp.tobytes()
    scale_lut, bias_lut = lut.build_luts(packed)
    s_lut = lut.decode_u8_scale_column_lut(packed, 0, scale_lut)
    s_exp = n8.decode_u8_scale_column(packed, 0)
    np.testing.assert_array_equal(s_lut, s_exp)
    b_lut = lut.decode_u8_bias_column_lut(packed, 0, bias_lut)
    b_exp = n8.decode_u8_bias_column(packed, 0)
    np.testing.assert_array_equal(b_lut, b_exp)


def test_metal_shader_does_not_materialize_f16_aux_and_lut_does_not_exp():
    inv = lut.lut_shader_invariants()
    assert inv["ok"] is True
    assert inv["materializes_f16_aux"] is False
    assert inv["binds_u8_aux"] is True
    assert inv["lut_kernels_contain_exp"] == []
    assert inv["lut_kernels_bind_half_aux"] == []
    assert inv["indexed_lut"] is True
    assert inv["in_register_exp"] is False
    assert inv["fill_kernel_uses_metal_exp"] is True
    assert inv["exp_variant_uses_exp"] is True
    src = lut.lut_shader_source()
    assert "device const uchar* scales_u8" in src
    for name in lut.LUT_CONSUMER_KERNELS:
        assert f"kernel void {name}" in src
        header = src.split(f"kernel void {name}", 1)[1].split("{", 1)[0]
        assert "half*" not in header
        assert "uchar* scales_u8" in header
    assert "kernel void aux_u8_native_affine_q2_geo_tpr64_tg128" in src
    assert "device const half*" in src  # incumbent arm


def test_fma_per_byte_by_class_three_way():
    table = lut.fma_per_byte_by_class(count_exp_as=0.0)
    inc = table["incumbent"]
    exp = table["exp_variant"]
    lu = table["lut_variant"]
    assert inc["weight_bytes_per_iteration"] == 6
    assert exp["weight_bytes_per_iteration"] == 4
    assert lu["weight_bytes_per_iteration"] == 4
    # Incumbent: 16 FMA / 6 B.
    assert inc["class"]["fma"] == pytest.approx(16 / 6, rel=1e-4)
    assert inc["class"]["integer"] == pytest.approx(16 / 6, rel=1e-4)
    assert inc["class"]["conversion"] == pytest.approx(10 / 6, rel=1e-4)  # 8 q + 2 half
    # Exp: 18 FMA / 4 B (8 dequant + 8 mac + 2 aux-decode).
    assert exp["class"]["fma"] == pytest.approx(18 / 4, rel=1e-4)
    assert exp["decode_fma_per_weight_byte"] == pytest.approx(10 / 4, rel=1e-4)
    assert exp["exp"] == 1
    # LUT: 16 FMA / 4 B; decode is dequant only; two indexed loads.
    assert lu["class"]["fma"] == pytest.approx(16 / 4, rel=1e-4)
    assert lu["decode_fma_per_weight_byte"] == pytest.approx(8 / 4, rel=1e-4)
    assert lu["exp"] == 0
    assert lu["aux_decode_fma"] == 0
    assert lu["lut_loads"] == 2
    assert lu["int_to_float"] == 8
    assert lu["class"]["conversion"] == pytest.approx(8 / 4, rel=1e-4)
    assert lu["class"]["memory"]["lut_loads_per_weight_byte"] == pytest.approx(2 / 4, rel=1e-4)
    assert lu["fma_per_weight_byte"] < exp["fma_per_weight_byte"]
    assert table["delta_lut_minus_exp"]["aux_decode_fma"] == -2
    assert table["delta_lut_minus_exp"]["exp"] == -1
    with_exp = lut.fma_per_byte_by_class(count_exp_as=1.0)
    assert with_exp["exp_variant"]["decode_fma_per_weight_byte"] == pytest.approx(11 / 4, rel=1e-4)
    assert with_exp["lut_variant"]["decode_fma_per_weight_byte"] == pytest.approx(8 / 4, rel=1e-4)


def test_economics_bills_both_ways_lut_extra_fma_is_zero():
    with pytest.raises(ee.IncompleteEconomics):
        ee.score(bytes_removed=lut.byte_model()["bytes_removed"])
    bundle = lut.economics_bundle()
    only = bundle["bytes_only"]
    billed = bundle["with_aux_decode_fma"]
    assert only["extra_flops_per_output_element"] == 0.0
    assert billed["extra_flops_per_output_element"] == 0.0
    assert billed["terms"]["flop_ms_delta"] == 0.0
    # These pinned POSITIVE, which was the OLD model's over-credit and the exact
    # contradiction this lane exposed: billed +1.5530 ms with ZERO added decode
    # FMA, and measured SLOWER than the incumbent. The calibrated model prices
    # BROADCAST AUX bytes at their measured marginal value - the per-group scale
    # and bias are read by many threads, cache-served, never on the critical
    # path - so the credit is now ZERO. The model and the wall clock no longer
    # disagree, which is the point; asserting POSITIVE would re-enshrine the bug.
    assert only["predicted_ms_saved"] == pytest.approx(0.0, abs=5e-3)
    assert billed["predicted_ms_saved"] == pytest.approx(0.0, abs=5e-3)
    assert bundle["net_sign_bytes_only"] == "ZERO"
    assert bundle["net_sign_with_aux_decode_fma"] == "ZERO"
    # The LUT's real achievement survives and is still asserted below: it adds no
    # decode FMA, so both billings agree with each other.
    # Exp-variant still bills the 2 FMA/8w tax that flipped the native sign.
    exp_billed = bundle["exp_variant_with_aux_decode_fma"]
    assert exp_billed["terms"]["flop_ms_delta"] > 0.0
    assert exp_billed["predicted_ms_saved"] < 0.0
    assert lut.extra_flops_per_output_element_lut() == 0.0


def test_refuses_speedup_without_byte_comparison_against_exp():
    with pytest.raises(lut.LutSpeedupWithoutByteMatchRefuse) as caught:
        lut.report_speedup(
            incumbent_gpu_ns=250_000,
            exp_gpu_ns=280_000,
            lut_gpu_ns=240_000,
        )
    assert "byte" in str(caught.value).lower()

    with pytest.raises(lut.LutSpeedupWithoutByteMatchRefuse):
        lut.report_speedup(
            incumbent_gpu_ns=250_000,
            exp_gpu_ns=280_000,
            lut_gpu_ns=240_000,
            bytes_equal=False,
        )

    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.0, 3.0], dtype=np.float32)
    with pytest.raises(lut.LutSpeedupWithoutByteMatchRefuse):
        lut.report_speedup(
            incumbent_gpu_ns=250_000,
            exp_gpu_ns=280_000,
            lut_gpu_ns=240_000,
            lut_output=a,
            exp_output=b,
        )

    ok = lut.report_speedup(
        incumbent_gpu_ns=250_000,
        exp_gpu_ns=280_000,
        lut_gpu_ns=240_000,
        lut_output=a,
        exp_output=a,
    )
    assert ok["unchanged_from_exp"] is True
    assert ok["lut_faster_than_incumbent"] is True
    assert ok["lut_faster_than_exp"] is True

    proof = lut.prove_output_unchanged_from_exp(bytes_equal=False, n_mismatch=4)
    assert proof["unchanged"] is False
    with pytest.raises(lut.LutSpeedupWithoutByteMatchRefuse):
        lut.report_speedup(
            incumbent_gpu_ns=250_000,
            exp_gpu_ns=280_000,
            lut_gpu_ns=240_000,
            unchanged_proof=proof,
        )


def test_ab_refuses_missing_loadavg_or_gpu_ns():
    raw = _raw()
    del raw["concurrent_load"]
    with pytest.raises(lut.IncompleteAB) as caught:
        lut.measurement_from_raw(raw)
    assert "loadavg" in str(caught.value).lower()

    raw = _raw()
    raw["concurrent_load"] = {}
    with pytest.raises(lut.IncompleteAB):
        lut.measurement_from_raw(raw)

    raw = _raw()
    del raw["lut_variant"]["gpu_ns_median"]
    with pytest.raises(lut.IncompleteAB):
        lut.measurement_from_raw(raw)

    raw = _raw()
    del raw["placements"]["device"]
    with pytest.raises(lut.IncompleteAB):
        lut.measurement_from_raw(raw)


def test_ab_from_raw_three_way_and_refuses_speedup_if_output_moved():
    measured = lut.measurement_from_raw(_raw())
    assert measured["absolute_gb_s_are_measured_under_load"] is True
    assert measured["concurrent_load"]["loadavg"]
    assert measured["incumbent"]["effective_gb_s"] > 0
    assert measured["exp_variant"]["effective_gb_s"] > 0
    assert measured["lut_variant"]["effective_gb_s"] > 0
    assert set(measured["placements"]) == {"constant", "threadgroup", "device"}
    assert measured["output_unchanged_from_exp"]["unchanged"] is True
    assert measured["speedup"] is not None
    assert measured["speedup_refused"] is None
    inc = measured["incumbent"]
    assert inc["effective_gb_s"] == pytest.approx(
        lut.effective_gb_s(inc["weight_bytes"], inc["gpu_ns_median"]), rel=1e-6
    )

    moved = _raw()
    moved["lut_vs_exp_output"] = {"bytes_equal": False, "chosen": {"n_mismatch": 12}}
    moved["lut_variant"]["output_bytes_equal_vs_exp"] = False
    measured_moved = lut.measurement_from_raw(moved)
    assert measured_moved["speedup"] is None
    assert measured_moved["speedup_refused"]["refused"] is True
    assert "byte" in measured_moved["speedup_refused"]["reason"].lower()
    # Times are still a measurement; they are not a speedup claim.
    assert measured_moved["lut_variant"]["gpu_us_median"] > 0


def test_record_refuses_without_gpu_ab(tmp_path: Path):
    with pytest.raises(lut.IncompleteAB):
        lut.record(measurement=None, path=tmp_path / "x.json")
    dest = tmp_path / "blocked.json"
    path = lut.record(
        measurement=None,
        gpu_blocked_reason="MTLCreateSystemDefaultDevice returned nil",
        path=dest,
    )
    doc = json.loads(path.read_text())
    assert doc["gpu_ab"]["status"] == "NOT_MEASURED"
    assert doc["gpu_ab"]["incumbent"] is None
    assert doc["gpu_ab"]["lut_variant"] is None
    assert doc["gpu_ab"]["fabricated_hardware_numbers"] is False
    assert doc["gpu_ab"]["concurrent_load"]["loadavg"]


def test_record_writes_class_table_loadavg_economics_and_speedup_gate(tmp_path: Path):
    dest = tmp_path / "AUX_U8_LUT.json"
    measured = lut.measurement_from_raw(_raw())
    path = lut.record(measurement=measured, path=dest)
    doc = json.loads(path.read_text())
    assert doc["schema"] == lut.SCHEMA
    assert doc["gpu_ab"]["concurrent_load"]["loadavg"]
    classes = doc["fma_per_byte_by_class"]["fma_only"]
    assert "incumbent" in classes and "exp_variant" in classes and "lut_variant" in classes
    assert classes["lut_variant"]["decode_fma_per_weight_byte"] == pytest.approx(2.0)
    assert doc["economics"]["net_sign_with_aux_decode_fma"] == "ZERO"
    assert doc["economics"]["with_aux_decode_fma"]["terms"]["flop_ms_delta"] == 0.0
    assert doc["shader_invariants"]["materializes_f16_aux"] is False
    assert doc["gpu_ab"]["speedup"]["unchanged_from_exp"] is True
    assert doc["error_did_not_move"]["ok"] is True


def test_committed_receipt_if_present_is_a_real_three_way():
    if not lut.RECEIPT.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(lut.RECEIPT.read_text())
    assert doc["schema"] == lut.SCHEMA
    ab = doc["gpu_ab"]
    assert ab is not None
    classes = doc["fma_per_byte_by_class"]["fma_only"]
    assert "incumbent" in classes and "exp_variant" in classes and "lut_variant" in classes
    load = ab.get("concurrent_load") or {}
    assert load.get("loadavg"), "A/B must carry loadavg even when GPU is blocked"
    if ab.get("status") == "NOT_MEASURED":
        assert ab.get("fabricated_hardware_numbers") is False
        assert ab.get("incumbent") is None
        assert ab.get("lut_variant") is None
        assert ab.get("effective_gb_s") is None
        return
    assert ab["incumbent"]["effective_gb_s"] > 0
    assert ab["exp_variant"]["effective_gb_s"] > 0
    assert ab["lut_variant"]["effective_gb_s"] > 0
    assert ab["incumbent"]["gpu_ns_median"] > 0
    assert set(ab["placements"]) == {"constant", "threadgroup", "device"}
    for name, arm in ab["placements"].items():
        assert arm["effective_gb_s"] > 0, name
        assert arm["gpu_ns_median"] > 0, name
    assert ab["absolute_gb_s_are_measured_under_load"] is True
    inc = ab["incumbent"]
    assert inc["effective_gb_s"] == pytest.approx(
        lut.effective_gb_s(int(inc["weight_bytes"]), int(inc["gpu_ns_median"])),
        rel=1e-4,
    )
    assert "net_sign_with_aux_decode_fma" in doc["economics"]
    assert doc["shader_invariants"]["materializes_f16_aux"] is False
    if ab.get("speedup") is not None:
        assert ab["speedup"]["unchanged_from_exp"] is True
    else:
        assert ab.get("speedup_refused", {}).get("refused") is True
