"""The granularity falsifier must refuse a verdict when the arms diverge."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import mlp_region_falsifier as mrf


def _arm(gb_s: float, *, gpu_ns: int | None = None, dispatches: int = 4, encoders: int = 4) -> dict:
    weight_bytes = mrf.CATALOG_BYTES_PER_LAYER
    if gpu_ns is None:
        gpu_ns = int(round(weight_bytes / gb_s))
    return {
        "label": "arm",
        "gpu_ns_median": gpu_ns,
        "gpu_ns_reps": [gpu_ns],
        "dispatches": dispatches,
        "encoders": encoders,
        "command_buffers": 1,
        "effective_gb_s": gb_s,
    }


def _raw(*, prod_gb_s: float = 344.1, cont_gb_s: float = 350.0, max_abs_diff: float = 0.0) -> dict:
    weight_bytes = mrf.CATALOG_BYTES_PER_LAYER
    prod_ns = int(round(weight_bytes / prod_gb_s))
    cont_ns = int(round(weight_bytes / cont_gb_s))
    return {
        "schema": "hawking.future.mlp_region_falsifier.raw.v1",
        "layer": 0,
        "kernel": "qwen_affine_q2_group32_matvec_geo_tpr64_tg128",
        "weight_bytes": weight_bytes,
        "catalog_bytes": weight_bytes,
        "warmup": 5,
        "reps": 11,
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_gate": max_abs_diff,
        "max_abs_diff_up": max_abs_diff,
        "max_abs_diff_act": max_abs_diff,
        "bit_identical": max_abs_diff == 0.0,
        "subfactors_changed": list(mrf.SUBFACTORS),
        "production": {
            "label": "production_scattered",
            "gpu_ns_median": prod_ns,
            "gpu_ns_reps": [prod_ns],
            "dispatches": 4,
            "encoders": 4,
            "command_buffers": 1,
        },
        "contiguous": {
            "label": "contiguous_serial_region",
            "gpu_ns_median": cont_ns,
            "gpu_ns_reps": [cont_ns],
            "dispatches": 4,
            "encoders": 1,
            "command_buffers": 1,
        },
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "arithmetic": "F(x)=down(silu(gate(x))*up(x))",
    }


def test_refuses_a_verdict_when_arms_are_not_bit_identical():
    with pytest.raises(mrf.ArmsNotIdentical) as caught:
        mrf.judge(344.1, 497.4, max_abs_diff=1.0)
    assert "bit-identical" in str(caught.value)
    with pytest.raises(mrf.ArmsNotIdentical):
        mrf.measurement_from_raw(_raw(cont_gb_s=497.4, max_abs_diff=1e-6))
    bad = mrf.measurement_from_raw(_raw(cont_gb_s=350.0, max_abs_diff=0.0))
    bad["bit_identical"] = False
    bad["max_abs_diff"] = 0.5
    with pytest.raises(mrf.ArmsNotIdentical):
        mrf.build(bad)


def test_implicated_only_when_contiguous_enters_lm_head_regime():
    assert mrf.judge(344.1, 431.0, 0.0) == mrf.VERDICT_IMPLICATED
    assert mrf.judge(344.1, 497.4, 0.0) == mrf.VERDICT_IMPLICATED
    doc = mrf.build(mrf.measurement_from_raw(_raw(prod_gb_s=344.1, cont_gb_s=497.4)))
    assert doc["verdict"] == mrf.VERDICT_IMPLICATED
    assert doc["contiguous"]["effective_gb_s"] >= mrf.IMPLICATE_GB_S
    assert "implicated" in doc["finding"].lower()


def test_refuted_when_contiguous_stays_near_350():
    assert mrf.judge(344.1, 350.0, 0.0) == mrf.VERDICT_REFUTED
    assert mrf.judge(344.1, 399.9, 0.0) == mrf.VERDICT_REFUTED
    doc = mrf.build(mrf.measurement_from_raw(_raw(prod_gb_s=344.1, cont_gb_s=351.0)))
    assert doc["verdict"] == mrf.VERDICT_REFUTED
    assert doc["contiguous"]["effective_gb_s"] < mrf.REFUTE_CEILING_GB_S
    assert "DEAD" in doc["finding"]


def test_refuses_a_forced_binary_in_the_dead_band():
    with pytest.raises(mrf.InconclusiveBandwidth) as caught:
        mrf.judge(344.1, 415.0, 0.0)
    assert "430" in str(caught.value)


def test_gb_s_is_bytes_over_gpu_ns():
    # 1 byte/ns = 1 GB/s under the 10^9 convention used by ORGAN_BANDWIDTH.
    assert mrf.effective_gb_s(350_000_000, 1_000_000) == 350.0
    weight = mrf.CATALOG_BYTES_PER_LAYER
    ns = int(round(weight / 344.1))
    got = mrf.effective_gb_s(weight, ns)
    assert abs(got - 344.1) < 0.1


def test_receipt_records_both_arms_and_the_identity_gate():
    doc = mrf.build(mrf.measurement_from_raw(_raw(prod_gb_s=344.1, cont_gb_s=348.0)))
    assert doc["schema"] == mrf.SCHEMA
    assert doc["evidence_class"] == "DIAGNOSTIC_RELATIVE"
    assert doc["verdict"] in (mrf.VERDICT_IMPLICATED, mrf.VERDICT_REFUTED)
    assert "effective_gb_s" in doc["production"]
    assert "effective_gb_s" in doc["contiguous"]
    assert doc["bit_identical"] is True
    assert doc["max_abs_diff"] == 0.0
    assert doc["layer"] == 0
    assert len(doc["subfactors_changed"]) >= 1
    assert "perfect-locality" in doc["claim_boundary"]


def test_record_writes_the_named_receipt(tmp_path: Path):
    dest = tmp_path / "MLP_REGION_FALSIFIER.json"
    measured = mrf.measurement_from_raw(_raw(prod_gb_s=344.1, cont_gb_s=349.0))
    path = mrf.record(measured, path=dest)
    assert path == dest
    doc = json.loads(dest.read_text())
    assert doc["verdict"] == mrf.VERDICT_REFUTED
    assert doc["production"]["effective_gb_s"] == pytest.approx(344.1, abs=0.2)
    assert doc["contiguous"]["effective_gb_s"] == pytest.approx(349.0, abs=0.2)


def test_record_without_a_measurement_refuses():
    with pytest.raises(mrf.ArmsNotIdentical):
        mrf.record(None)


def test_committed_receipt_if_present_is_a_real_verdict():
    """Once recorded, the receipt must name both arms and a binary verdict."""
    if not mrf.RECEIPT.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(mrf.RECEIPT.read_text())
    assert doc["schema"] == mrf.SCHEMA
    assert doc["evidence_class"] == "DIAGNOSTIC_RELATIVE"
    assert doc["verdict"] in (mrf.VERDICT_IMPLICATED, mrf.VERDICT_REFUTED)
    assert doc["bit_identical"] is True
    assert doc["max_abs_diff"] == 0.0
    assert doc["production"]["effective_gb_s"] > 0
    assert doc["contiguous"]["effective_gb_s"] > 0
    # Re-judging the recorded numbers must not disagree with the seal.
    assert (
        mrf.judge(
            doc["production"]["effective_gb_s"],
            doc["contiguous"]["effective_gb_s"],
            doc["max_abs_diff"],
        )
        == doc["verdict"]
    )
