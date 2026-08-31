"""Tests for the native u8-aux consumer.

Load-bearing:
  * argmax agreement alone is not equivalence (KL required)
  * the native path does not materialize the f16 aux (CPU ledger + Metal source)
  * expanding u8 → f16 aux and feeding the ordinary kernel is refused
  * A/B refuses to report GB/s without measured gpu_ns and loadavg
  * FMA/byte is reported before and after
  * executable_economics bills extra decode arithmetic (and bytes_added)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.future import aux_capability_screen as acs
from tools.future import aux_u8_native as n8
from tools.future import executable_economics as ee


def _tiny_packed(rows: int = 8, cols: int = 64, seed: int = 0) -> n8.PackedU8Aux:
    rng = np.random.default_rng(seed)
    gpr = cols // n8.GROUP
    q = rng.integers(0, 4, size=(rows, gpr, n8.GROUP)).astype(np.uint8)
    scale = rng.random((rows, gpr)).astype(np.float32) * 0.02 + 0.005
    bias = -1.5 * scale + rng.normal(scale=0.001, size=scale.shape).astype(np.float32)
    return n8.pack_u8_aux(q, scale, bias, rows=rows, cols=cols)


def _arm(gb_s: float, *, weight_bytes: int, label: str) -> dict:
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
        "aux_bytes": weight_bytes // 10,
        "code_bytes": weight_bytes - weight_bytes // 10,
    }


def _raw(**kwargs) -> dict:
    inc_bytes = 83_558_400
    nat_bytes = 75_202_560
    base = {
        "schema": "hawking.future.aux_u8_native.raw.v1",
        "layer": 3,
        "warmup": 5,
        "reps": 11,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "absolute_gb_s_are_measured_under_load": True,
        "concurrent_load": {"loadavg": "{ 6.87 7.62 8.56 }"},
        "incumbent": _arm(330.0, weight_bytes=inc_bytes, label="incumbent_f16_aux"),
        "native_u8": {
            **_arm(300.0, weight_bytes=nat_bytes, label="native_u8_aux"),
            "materializes_f16_aux": False,
            "endpoint_bytes": 48,
        },
        "output_cosine_mean": 0.999,
        "native_consumer": {
            "materializes_f16_aux": False,
            "scale_buffer": "u8",
            "bias_buffer": "u8",
        },
    }
    base.update(kwargs)
    return base


def test_argmax_alone_raises_as_equivalence():
    """NEGATIVE CONTROL: keeping argmax is not a native-equivalence screen."""
    with pytest.raises(n8.ArgmaxAloneParityRefuse) as caught:
        n8.report_equivalence(
            organ_cosine=0.999,
            kl_nats=None,
            argmax_agreement=1.0,
        )
    assert "REFUSED" in str(caught.value)
    assert "argmax" in str(caught.value).lower()
    assert "kl" in str(caught.value).lower() or "logit" in str(caught.value).lower()

    with pytest.raises(n8.ArgmaxAloneParityRefuse):
        n8.report_equivalence(
            organ_cosine=None,
            kl_nats=None,
            argmax_agreement=1.0,
        )


def test_equivalence_requires_kl_and_flags_argmax():
    ok = n8.report_equivalence(
        organ_cosine=0.999,
        kl_nats=0.002,
        argmax_agreement=1.0,
        top_k_agreement=0.95,
        n_rows=16,
    )
    assert ok["kl_nats"] == pytest.approx(0.002)
    assert ok["organ_cosine"] == pytest.approx(0.999)
    assert ok["argmax_agreement"] == pytest.approx(1.0)
    assert ok["argmax_is_not_parity"] is True
    assert "kl_nats" in ok["parity_quantities"]
    assert ok["clears_task_bar"] is True

    drifted = n8.report_equivalence(
        organ_cosine=0.999,
        kl_nats=0.4,
        argmax_agreement=1.0,
    )
    assert drifted["argmax_agreement"] == pytest.approx(1.0)
    assert drifted["kl_nats"] > n8.TASK_LOGIT_KL_BAR
    assert drifted["clears_task_bar"] is False


def test_native_matvec_does_not_materialize_f16_aux():
    packed = _tiny_packed(rows=16, cols=256)
    n_groups = packed.n_groups
    assert n_groups == 16 * (256 // 64)  # gpr=4 → 64 groups
    x = np.random.default_rng(2).normal(size=(4, 256)).astype(np.float32)
    ledger = n8.AuxAllocLedger()
    y = n8.native_matvec_u8(x, packed, ledger=ledger)
    assert y.shape == (4, 16)
    assert ledger.wrote_f16_aux_buffer is False
    # Peak is one group-column of scale+bias = 2*rows, not 2*n_groups.
    assert ledger.max_decoded_aux_elems == 16 * 2
    assert ledger.max_decoded_aux_elems < 2 * n_groups
    assert (
        n8.classify_consumer(
            materializes_f16_aux=False,
            binds_u8_aux=True,
            peak_decoded_aux_elems=ledger.max_decoded_aux_elems,
            n_groups=n_groups,
        )
        == n8.DIRECT_CONSUME
    )


def test_remat_then_ordinary_is_refused():
    packed = _tiny_packed()
    ledger = n8.AuxAllocLedger()
    x = np.ones(packed.cols, dtype=np.float32)
    _ = n8.remat_then_ordinary_matvec(x, packed, ledger=ledger)
    assert ledger.wrote_f16_aux_buffer is True
    assert ledger.max_decoded_aux_elems >= packed.n_groups
    with pytest.raises(n8.MaterializeF16AuxRefuse) as caught:
        n8.classify_consumer(
            materializes_f16_aux=True,
            binds_u8_aux=False,
            peak_decoded_aux_elems=ledger.max_decoded_aux_elems,
            n_groups=packed.n_groups,
        )
    assert "f16" in str(caught.value).lower()
    with pytest.raises(n8.MaterializeF16AuxRefuse):
        n8.classify_consumer(
            materializes_f16_aux=False,
            binds_u8_aux=True,
            peak_decoded_aux_elems=2 * packed.n_groups,
            n_groups=packed.n_groups,
        )


def test_native_matches_forbidden_remat_numerically_on_tiny():
    """Same encode, two consumes: in-register vs remat. Remat is not the runtime."""
    packed = _tiny_packed(rows=8, cols=64, seed=3)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, 64)).astype(np.float32)
    y_nat = n8.native_matvec_u8(x, packed)
    y_rem = n8.remat_then_ordinary_matvec(x, packed)
    assert n8.cosine(y_nat, y_rem) == pytest.approx(1.0, abs=1e-5)
    # Incumbent reconstructed W (f16 aux) is the screened-fit target, not identity.
    s = packed.scale_u8
    assert s.dtype == np.uint8


def test_native_tracks_screened_u8_encode():
    """Native group-column decode equals the screen's u8_log/linear decode."""
    packed = _tiny_packed(rows=4, cols=64, seed=4)
    ep = packed.endpoints()
    s_full = acs.u8_log_decode(packed.scale_u8.reshape(-1), ep["scale_lmin"], ep["scale_lmax"])
    b_full = acs.u8_linear_decode(packed.bias_u8.reshape(-1), ep["bias_min"], ep["bias_max"])
    s_col = n8.decode_u8_scale_column(packed, 0)
    b_col = n8.decode_u8_bias_column(packed, 0)
    np.testing.assert_allclose(s_col, s_full.reshape(4, 1)[:, 0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(b_col, b_full.reshape(4, 1)[:, 0], rtol=1e-5, atol=1e-6)


def test_metal_shader_does_not_materialize_f16_aux():
    inv = n8.native_shader_invariants()
    assert inv["ok"] is True
    assert inv["materializes_f16_aux"] is False
    assert inv["binds_u8_aux"] is True
    assert inv["in_register_exp"] is True
    assert inv["native_binds_half_aux"] is False
    assert inv["forbidden_hits"] == []
    src = n8.native_shader_source()
    assert "device const uchar* scales_u8" in src
    assert "kernel void aux_u8_native_affine_q2_geo_tpr64_tg128" in src
    # Incumbent arm still uses half aux — that is the A, not the native B.
    assert "device const half*" in src
    native_header = src.split("kernel void aux_u8_native_affine_q2_geo_tpr64_tg128", 1)[1].split(
        "{", 1
    )[0]
    assert "half*" not in native_header
    assert "uchar* scales_u8" in native_header


def test_fma_per_byte_before_and_after():
    table = n8.fma_per_byte_table(count_exp_as=0.0)
    before = table["before"]
    after = table["after"]
    assert before["weight_bytes_per_iteration"] == 6
    assert after["weight_bytes_per_iteration"] == 4
    # Incumbent: 8 dequant + 8 mac / 6 B = 2.6667 FMA/byte, 8/6 decode.
    assert before["fma_per_weight_byte"] == pytest.approx(16 / 6, rel=1e-4)
    assert before["decode_fma_per_weight_byte"] == pytest.approx(8 / 6, rel=1e-4)
    # Native: 8 dequant + 8 mac + 2 aux-decode FMA / 4 B = 4.5; decode 10/4 = 2.5.
    assert after["fma_per_weight_byte"] == pytest.approx(18 / 4, rel=1e-4)
    assert after["decode_fma_per_weight_byte"] == pytest.approx(10 / 4, rel=1e-4)
    assert after["fma_per_weight_byte"] > before["fma_per_weight_byte"]
    assert table["delta"]["weight_bytes_per_iteration"] == -2
    # Counting exp as 1 FMA raises native decode intensity further.
    with_exp = n8.fma_per_byte_table(count_exp_as=1.0)
    assert with_exp["after"]["decode_fma_per_weight_byte"] == pytest.approx(11 / 4, rel=1e-4)


def test_extra_flops_per_output_element_is_the_inner_loop_tax():
    # 2 FMA per 8-weight inner iter, mean over gate+up+down rows.
    fma_only = n8.extra_flops_per_output_element(count_exp_as=0.0)
    with_exp = n8.extra_flops_per_output_element(count_exp_as=1.0)
    n_out = 2 * n8.INTERMEDIATE + n8.HIDDEN
    n_params_layer = 3 * n8.INTERMEDIATE * n8.HIDDEN
    assert fma_only == pytest.approx((n_params_layer / 8.0) * 2.0 / n_out)
    assert with_exp == pytest.approx((n_params_layer / 8.0) * 3.0 / n_out)
    assert with_exp > fma_only


def test_economics_bills_bytes_added_and_extra_decode_arithmetic():
    with pytest.raises(ee.IncompleteEconomics):
        ee.score(bytes_removed=n8.byte_model()["bytes_removed"])
    bundle = n8.economics_bundle()
    only = bundle["bytes_only_screen_style"]
    billed = bundle["with_aux_decode_fma"]
    assert only["bytes_added"]["metadata"] == n8.byte_model()["bytes_added_metadata"]
    assert only["extra_flops_per_output_element"] == 0.0
    assert billed["extra_flops_per_output_element"] == pytest.approx(
        n8.extra_flops_per_output_element(count_exp_as=0.0)
    )
    # Arithmetic cost is visible: flop term is positive, so saved ms drops.
    assert billed["terms"]["flop_ms_delta"] > 0.0
    assert billed["predicted_ms_saved"] < only["predicted_ms_saved"]
    # 1.5541 was the OLD model's over-credit and pinning it here is what made
    # this test fail once the model was corrected. quantize_aux_u8 removes
    # BROADCAST AUX bytes - a per-group scale and bias many threads read, so
    # cache-served and never on the critical path - and the calibrated model
    # prices them at their measured marginal value, which is ~0. That the
    # bytes-only figure collapsed from +1.5541 to 0.0 IS the finding: the lever
    # billed positive and measured 29.04 us SLOWER, and the model no longer
    # disagrees with the wall clock.
    assert only["stream_class"] == "broadcast_aux"
    assert only["predicted_ms_saved"] == pytest.approx(0.0, abs=5e-3)
    assert only["terms"]["byte_removed_ms"] == pytest.approx(0.0, abs=5e-3)
    # Nominal extra decode FMA exceeds the byte saving: the lever can be a slowdown.
    assert billed["terms"]["flop_ms_delta"] > abs(only["terms"]["byte_removed_ms"])
    assert billed["predicted_ms_saved"] < 0.0


def test_ab_refuses_missing_loadavg_or_gpu_ns():
    raw = _raw()
    del raw["concurrent_load"]
    with pytest.raises(n8.IncompleteAB) as caught:
        n8.measurement_from_raw(raw)
    assert "loadavg" in str(caught.value).lower()

    raw = _raw()
    raw["concurrent_load"] = {}
    with pytest.raises(n8.IncompleteAB):
        n8.measurement_from_raw(raw)

    raw = _raw()
    del raw["native_u8"]["gpu_ns_median"]
    with pytest.raises(n8.IncompleteAB):
        n8.measurement_from_raw(raw)

    with pytest.raises(n8.EmptyGpuSample):
        n8.effective_gb_s(1000, 0)


def test_ab_from_raw_carries_measured_gb_s_and_loadavg():
    measured = n8.measurement_from_raw(_raw())
    assert measured["absolute_gb_s_are_measured_under_load"] is True
    assert measured["concurrent_load"]["loadavg"]
    assert measured["incumbent"]["effective_gb_s"] > 0
    assert measured["native_u8"]["effective_gb_s"] > 0
    assert measured["incumbent"]["gpu_us_median"] > 0
    assert measured["delta"]["weight_bytes"] < 0  # native moves fewer bytes
    assert measured["token_projection"]["kind"] == "PROJECTION_ARITHMETIC_OVER_PROBE"
    assert measured["token_projection"]["not_a_resident_measurement"] is True
    assert measured["token_projection"]["layers"] == 64
    # GB/s is bytes/ns, not invented.
    inc = measured["incumbent"]
    assert inc["effective_gb_s"] == pytest.approx(
        n8.effective_gb_s(inc["weight_bytes"], inc["gpu_ns_median"]), rel=1e-6
    )


def test_record_refuses_without_gpu_ab(tmp_path: Path):
    with pytest.raises(n8.IncompleteAB):
        n8.record(measurement=None, path=tmp_path / "x.json")
    # Explicit blocked-GPU path still must not invent a bandwidth.
    dest = tmp_path / "blocked.json"
    path = n8.record(
        measurement=None,
        gpu_blocked_reason="MTLCreateSystemDefaultDevice returned nil",
        path=dest,
    )
    doc = json.loads(path.read_text())
    assert doc["gpu_ab"]["status"] == "NOT_MEASURED"
    assert doc["gpu_ab"]["incumbent"] is None
    assert doc["gpu_ab"]["fabricated_hardware_numbers"] is False
    assert doc["gpu_ab"]["concurrent_load"]["loadavg"]


def test_record_writes_fma_table_loadavg_and_economics(tmp_path: Path):
    dest = tmp_path / "AUX_U8_NATIVE.json"
    measured = n8.measurement_from_raw(_raw())
    path = n8.record(measurement=measured, path=dest)
    doc = json.loads(path.read_text())
    assert doc["schema"] == n8.SCHEMA
    assert doc["argmax_alone_is_not_parity"] is True
    assert doc["dense_rematerialization"] == n8.DIRECT_CONSUME
    assert doc["gpu_ab"]["concurrent_load"]["loadavg"]
    assert doc["gpu_ab"]["incumbent"]["effective_gb_s"] > 0
    assert doc["gpu_ab"]["native_u8"]["effective_gb_s"] > 0
    assert "before" in doc["fma_per_byte"]["fma_only"]
    assert "after" in doc["fma_per_byte"]["fma_only"]
    assert doc["fma_per_byte"]["fma_only"]["before"]["decode_fma_per_weight_byte"] == pytest.approx(
        8 / 6, rel=1e-4
    )
    assert doc["fma_per_byte"]["fma_only"]["after"]["decode_fma_per_weight_byte"] == pytest.approx(
        10 / 4, rel=1e-4
    )
    assert doc["economics"]["with_aux_decode_fma"]["terms"]["flop_ms_delta"] > 0
    assert doc["shader_invariants"]["materializes_f16_aux"] is False
    assert doc["shader_invariants"]["binds_u8_aux"] is True
    assert doc["gpu_ab"]["absolute_gb_s_are_measured_under_load"] is True


def test_committed_receipt_if_present_is_a_real_ab():
    if not n8.RECEIPT.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(n8.RECEIPT.read_text())
    assert doc["schema"] == n8.SCHEMA
    ab = doc["gpu_ab"]
    assert ab is not None
    assert "before" in doc["fma_per_byte"]["fma_only"]
    assert "after" in doc["fma_per_byte"]["fma_only"]
    load = ab.get("concurrent_load") or {}
    assert load.get("loadavg"), "A/B must carry loadavg even when GPU is blocked"
    if ab.get("status") == "NOT_MEASURED":
        # Honest refuse: no fabricated GB/s or gpu_ns.
        assert ab.get("fabricated_hardware_numbers") is False
        assert ab.get("incumbent") is None
        assert ab.get("native_u8") is None
        assert ab.get("effective_gb_s") is None
        assert ab.get("delta") is None
        return
    assert ab["incumbent"]["effective_gb_s"] > 0
    assert ab["native_u8"]["effective_gb_s"] > 0
    assert ab["incumbent"]["gpu_ns_median"] > 0
    assert ab["native_u8"]["gpu_ns_median"] > 0
    assert ab["absolute_gb_s_are_measured_under_load"] is True
    inc = ab["incumbent"]
    assert inc["effective_gb_s"] == pytest.approx(
        n8.effective_gb_s(int(inc["weight_bytes"]), int(inc["gpu_ns_median"])),
        rel=1e-4,
    )
