"""Decode-cheapen sidecar: no bit-identical stamp without a byte comparison."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import mlp_decode_cheapen as mdc


def _compare(*, n_bytes: int = 160_000, mismatch_bytes: int = 0, mismatch_floats: int = 0,
             cosine: float = 1.0, rel_fro: float = 0.0, max_abs: float = 0.0) -> dict:
    return {
        "n_bytes_compared": n_bytes,
        "n_floats_compared": n_bytes // 4,
        "n_mismatch_bytes": mismatch_bytes,
        "n_float_mismatch": mismatch_floats,
        "max_abs_err": max_abs,
        "cosine": cosine,
        "rel_fro": rel_fro,
        "compared_against": "production kernel output buffers after the timed command buffer",
    }


def _arm(
    ident: str,
    *,
    gb_s: float,
    weight_bytes: int = 83_558_400,
    klass: str = "exact_candidate",
    compare: dict | None = None,
    bit_identical: bool | None = None,
) -> dict:
    gpu_ns = int(round(weight_bytes / gb_s))
    out = {
        "id": ident,
        "kernel": f"k_{ident}",
        "class": klass,
        "mechanisms": ["test"],
        "note": "fixture",
        "weight_bytes": weight_bytes,
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [gpu_ns],
        "gpu_us_median": gpu_ns / 1e3,
        "effective_gb_s": gb_s,
        "dispatches": 3,
        "encoders": 1,
        "command_buffers": 1,
        "threads_per_threadgroup": 128,
        "occupancy": {
            "threads_per_threadgroup": 128,
            "max_total_threads_per_threadgroup": 1024,
            "thread_execution_width": 32,
            "occupancy_of_max_threads": 0.125,
            "threadgroups_per_core": 145.06,
        },
        "byte_compare": compare if compare is not None else _compare(),
    }
    if bit_identical is not None:
        out["bit_identical"] = bit_identical
    return out


def _raw(variants: list[dict] | None = None) -> dict:
    if variants is None:
        variants = []
        for ident in mdc.VARIANT_IDS:
            if ident == "production":
                variants.append(_arm(ident, gb_s=329.6, klass="control"))
            elif ident in mdc.APPROX_CANDIDATE_IDS:
                variants.append(
                    _arm(
                        ident,
                        gb_s=400.0,
                        klass="approx_candidate",
                        compare=_compare(
                            mismatch_bytes=4,
                            mismatch_floats=1,
                            cosine=0.999999,
                            rel_fro=1e-7,
                            max_abs=1e-5,
                        ),
                    )
                )
            else:
                variants.append(_arm(ident, gb_s=360.0, klass="exact_candidate"))
    return {
        "schema": "hawking.future.mlp_decode_cheapen.raw.v1",
        "layer": 0,
        "warmup": 5,
        "reps": 11,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "absolute_gb_s_are_measured_under_load": True,
        "concurrent_load": {"loadavg": "{ 1.0 1.0 1.0 }"},
        "organ": "mlp",
        "codec": "HGRAVF01 affine2 q2 group64",
        "geometry": "geo_tpr64_tg128",
        "production_kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
        "weight_bytes": 83_558_400,
        "variants": variants,
    }


def test_refuses_bit_identical_without_byte_comparison():
    v = _arm("lut4_select", gb_s=360.0)
    del v["byte_compare"]
    v["bit_identical"] = True
    with pytest.raises((mdc.NoByteComparison, mdc.BitIdentityClaimWithoutCompare)) as caught:
        mdc.bit_identical_from_compare(v)
    msg = str(caught.value).lower()
    assert "bit-identical" in msg
    assert "byte" in msg


def test_refuses_bit_identical_when_n_bytes_compared_is_zero():
    v = _arm("lut4_select", gb_s=360.0, compare=_compare(n_bytes=0))
    v["bit_identical"] = True
    with pytest.raises(mdc.NoByteComparison) as caught:
        mdc.bit_identical_from_compare(v)
    assert "n_bytes_compared" in str(caught.value)


def test_refuses_to_build_a_claimed_identical_mismatch():
    variants = _raw()["variants"]
    for v in variants:
        if v["id"] == "lut4_select":
            v["bit_identical"] = True
            v["byte_compare"] = _compare(mismatch_bytes=12, mismatch_floats=3, cosine=0.9)
    with pytest.raises(mdc.BitIdentityClaimWithoutCompare):
        mdc.measurement_from_raw(_raw(variants))


def test_derive_identical_only_from_zero_mismatch():
    ok = _arm("lut4_select", gb_s=360.0, compare=_compare())
    assert mdc.bit_identical_from_compare(ok) is True
    bad = _arm(
        "fold",
        gb_s=400.0,
        compare=_compare(mismatch_bytes=4, mismatch_floats=1, cosine=0.999),
    )
    assert mdc.bit_identical_from_compare(bad) is False


def test_projection_is_labelled_and_arithmetic():
    prod_ns = 253_500
    var_ns = 168_000
    p = mdc.project_from_probe(prod_ns, var_ns)
    assert p["kind"] == "projection"
    assert "not a resident" in p["label"].lower()
    assert "PROJECTION" in p["note"]
    speedup = prod_ns / var_ns
    assert p["mlp_ms_projected"] == round(mdc.MLP_MS / speedup, 3)
    saved = mdc.MLP_MS - mdc.MLP_MS / speedup
    assert p["token_ms_projected"] == round(mdc.TOKEN_MS - saved, 3)
    assert p["baseline_mlp_ms"] == 15.541
    assert p["baseline_token_ms"] == 28.722


def test_projection_refuses_zero_ns():
    with pytest.raises(mdc.EmptyGpuSample):
        mdc.project_from_probe(0, 168_000)


def test_affine_fold_is_exact_over_reals():
    ident = mdc.affine_fold_identity_over_reals()
    assert ident["exact_over_reals"] is True
    assert ident["abs_err"] < 1e-12


def test_affine_fold_is_not_always_f32_bit_identical():
    cex = mdc.f32_counterexample_for_fold()
    assert cex["fold_matches_production_f32"] is False
    assert cex["lut4_matches_production_f32"] is True
    assert cex["fold_abs_err"] > 0.0


def test_lut4_matches_production_on_many_tiles():
    for packed in (0, 1, 0xFFFF, 0x6C6C, 0xA5A5):
        for scale, bias in ((0.3, 0.1), (-0.5, 0.25), (1.0, 0.0)):
            x = [((i * 3) % 17) * 0.125 - 1.0 for i in range(8)]
            assert mdc.production_unpack8(packed, scale, bias, x) == mdc.lut4_unpack8(
                packed, scale, bias, x
            )


def test_gb_s_is_bytes_over_gpu_ns():
    assert mdc.effective_gb_s(350_000_000, 1_000_000) == 350.0
    with pytest.raises(mdc.EmptyGpuSample):
        mdc.effective_gb_s(100, 0)


def test_missing_production_or_variant_refuses():
    raw = _raw()
    raw["variants"] = [v for v in raw["variants"] if v["id"] != "fold"]
    with pytest.raises(mdc.MissingVariant) as caught:
        mdc.measurement_from_raw(raw)
    assert "fold" in str(caught.value)

    raw = _raw()
    raw["variants"] = [v for v in raw["variants"] if v["id"] != "production"]
    with pytest.raises(mdc.MissingVariant):
        mdc.measurement_from_raw(raw)


def test_zero_gpu_ns_refuses():
    raw = _raw()
    raw["variants"][0]["gpu_ns_median"] = 0
    with pytest.raises(mdc.EmptyGpuSample):
        mdc.measurement_from_raw(raw)


def test_record_writes_classes_separately_and_refuses_none(tmp_path: Path):
    dest = tmp_path / "MLP_DECODE_CHEAPEN.json"
    measured = mdc.measurement_from_raw(_raw())
    path = mdc.record(measured, path=dest)
    assert path == dest
    doc = json.loads(dest.read_text())
    assert doc["schema"] == mdc.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["absolute_gb_s_are_measured_under_load"] is True
    ids = [v["id"] for v in doc["variants"]]
    assert ids == list(mdc.VARIANT_IDS)
    for v in doc["variants"]:
        assert v["effective_gb_s"] > 0
        assert "fma_per_weight_byte" in v
        assert "decode_fma_per_weight_byte" in v
        assert "byte_compare" in v
        assert v["bit_identical"] == mdc.bit_identical_from_compare(v)
        assert v["projection"]["kind"] == "projection"
        assert "not a resident" in v["projection"]["label"].lower()
    prod = next(v for v in doc["variants"] if v["id"] == "production")
    assert prod["decode_fma_per_weight_byte"] == round(8 / 6, 4)
    assert prod["fma_per_weight_byte"] == round(16 / 6, 4)
    fold = next(v for v in doc["variants"] if v["id"] == "fold")
    assert fold["bit_identical"] is False
    assert "capability_score" in fold
    assert fold["capability_score"]["cosine_bar"] == 0.99
    lut = next(v for v in doc["variants"] if v["id"] == "lut4_select")
    assert lut["bit_identical"] is True
    assert "capability_score" not in lut
    assert doc["best_exact"]["id"] in mdc.EXACT_CANDIDATE_IDS
    assert doc["best_approx"]["id"] in mdc.APPROX_CANDIDATE_IDS
    assert "Do not blend" in doc["finding"] or "not bit-identical" in doc["finding"].lower()
    assert doc["algebra"]["over_reals"]["exact_over_reals"] is True
    with pytest.raises(mdc.MissingVariant):
        mdc.record(None)


def test_gap_fraction_uses_this_run_production():
    g = mdc.gap_to_arm_a(329.6, 497.4)
    assert g["fraction_of_gap_closed"] == 1.0
    assert g["arm_a_target_gb_s"] == 497.4
    g2 = mdc.gap_to_arm_a(329.6, 329.6)
    assert g2["fraction_of_gap_closed"] == 0.0


def test_committed_receipt_if_present_is_measured():
    if not mdc.RECEIPT.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(mdc.RECEIPT.read_text())
    assert doc["schema"] == mdc.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["absolute_gb_s_are_measured_under_load"] is True
    assert "PROJECTION" in doc["finding"] or "projection" in doc["finding"].lower()
    ids = [v["id"] for v in doc["variants"]]
    assert "production" in ids
    for v in doc["variants"]:
        assert v["effective_gb_s"] > 0
        assert "fma_per_weight_byte" in v
        assert "decode_fma_per_weight_byte" in v
        assert int(v["byte_compare"]["n_bytes_compared"]) > 0
        assert v["bit_identical"] == mdc.bit_identical_from_compare(v)
        assert v["projection"]["kind"] == "projection"
    exact = [v for v in doc["variants"] if v["id"] != "production" and v["bit_identical"]]
    approx = [v for v in doc["variants"] if v["id"] != "production" and not v["bit_identical"]]
    if exact:
        assert doc["best_exact"]["id"] == max(exact, key=lambda v: v["effective_gb_s"])["id"]
    if approx:
        assert doc["best_approx"]["id"] == max(approx, key=lambda v: v["effective_gb_s"])["id"]
        assert "capability_score" in doc["best_approx"]
    # Re-judging the recorded numbers must not disagree with the seal.
    measured = mdc.measurement_from_raw(
        {
            "layer": doc["layer"],
            "warmup": doc["warmup"],
            "reps": doc["reps"],
            "git_head": doc["git_head"],
            "artifact_root": doc["artifact_root"],
            "timing": doc["timing"],
            "concurrent_load": doc["concurrent_load"],
            "organ": doc["organ"],
            "codec": doc["codec"],
            "geometry": doc["geometry"],
            "production_kernel": doc["production_kernel"],
            "projections": doc["projections"],
            "weight_bytes": doc["weight_bytes"],
            "variants": doc["variants"],
        }
    )
    rebuilt = mdc.build(measured)
    assert rebuilt["production_gb_s"] == doc["production_gb_s"]
    for a, b in zip(rebuilt["variants"], doc["variants"]):
        assert a["id"] == b["id"]
        assert a["bit_identical"] == b["bit_identical"]
        assert a["effective_gb_s"] == b["effective_gb_s"]
