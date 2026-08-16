"""Arithmetic and census gates for DSV4F resident-gravity feasibility."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.operators.dsv4f_resident_gravity import (
    HOST_RAM_BYTES,
    Q80_MIXED_EXPERT_BPW,
    complete_bpw,
    complete_physical_bytes,
    gib,
    indexer_slots,
    kv_slots_per_layer,
    max_routed_bpw,
    packaging_bytes,
    q80_rate_hypothesis_bytes,
    working_set_bytes,
)
from lab.operators.dsv4f_tensor_schedule import (
    EXPECTED_TENSOR_COUNT,
    EXPECTED_TOTAL_TENSOR_BYTES,
    resolve_artifact_root,
)

REPO = Path(__file__).resolve().parents[2]


def test_complete_physical_bytes_is_ceil_of_params_times_bpw_over_8() -> None:
    assert complete_physical_bytes(8, 1.0) == 1
    assert complete_physical_bytes(9, 1.0) == 2
    assert complete_physical_bytes(290_942_289_362, 1.5) == 54_551_679_256


def test_gib_uses_1024_cubed() -> None:
    assert gib(1024**3) == 1.0
    assert abs(gib(54_551_679_256) - 50.80521) < 1e-4


def test_kv_slots_match_official_model_py_formula() -> None:
    assert kv_slots_per_layer(1_048_576, 0) == 128
    assert kv_slots_per_layer(64, 0) == 64
    assert kv_slots_per_layer(1_048_576, 4) == 128 + 1_048_576 // 4
    assert kv_slots_per_layer(1_048_576, 128) == 128 + 1_048_576 // 128
    assert indexer_slots(1_048_576, 4) == 1_048_576 // 4


def test_working_set_1m_is_single_digit_gib_not_tens() -> None:
    ws = working_set_bytes(1_048_576)
    assert ws["attention_kv_bytes"] + ws["indexer_kv_bytes"] < 8 * 1024**3
    assert ws["compressor_state_bytes"] < 32 * 1024**2
    # Sliding layers stay window-sized; compressed layers dominate.
    short = working_set_bytes(128)
    assert short["total_kv_plus_index_bytes"] < ws["total_kv_plus_index_bytes"]


def test_q80_analogous_split_holds_1_5_with_protect_at_source() -> None:
    f_routed = 0.974310469
    protect_bpw = 9.651
    assert complete_bpw(f_routed, 1.22957, protect_bpw) < 1.5
    assert complete_bpw(f_routed, 1.22957, 8.0) < 1.5
    assert max_routed_bpw(f_routed=f_routed, protect_bpw=protect_bpw, target=1.5) < 1.3
    assert max_routed_bpw(f_routed=f_routed, protect_bpw=8.0, target=1.5) > 1.32


def test_q80_hypothesis_refuses_unequal_thirds() -> None:
    with pytest.raises(ValueError):
        q80_rate_hypothesis_bytes(100)


def test_q80_hypothesis_bills_three_families() -> None:
    routed = 94_489_280_512 * 3
    hyp = q80_rate_hypothesis_bytes(routed)
    assert hyp["status"] == "HYPOTHESIS_UNFITTED"
    assert hyp["w1_bytes"] < hyp["w3_bytes"]
    assert hyp["routed_complete_bytes"] == hyp["w1_bytes"] + hyp["w3_bytes"] + hyp["w2_bytes"]
    assert abs(hyp["routed_complete_bpw_from_bytes"] - Q80_MIXED_EXPERT_BPW) < 1e-4


def test_source_precision_exceeds_host_ram() -> None:
    assert EXPECTED_TOTAL_TENSOR_BYTES > HOST_RAM_BYTES
    assert complete_physical_bytes(290_942_289_362, 1.5) + packaging_bytes() < HOST_RAM_BYTES


def test_live_census_and_residency_invariants() -> None:
    try:
        resolve_artifact_root()
    except FileNotFoundError as exc:
        pytest.fail(f"sealed DSV4F artifact is required: {exc}")

    from lab.operators.dsv4f_resident_gravity import analyze

    report = analyze()
    cov = report["coverage"]
    assert cov["covers_all_tensors"] is True
    assert cov["tensor_count"] == EXPECTED_TENSOR_COUNT
    assert cov["byte_mass"] == EXPECTED_TOTAL_TENSOR_BYTES
    assert cov["byte_residual"] == 0
    assert cov["logical_params"] == 290_942_289_362
    assert report["organs"]["routed_expert"]["logical_params"] == 283_467_841_536
    split = report["mass_split"]["q80_analogous"]
    assert split["f_routed"] == pytest.approx(0.9743095173878279, abs=1e-12)
    assert report["verdict"]["source_precision_resident"] is False
    assert report["verdict"]["target_1_5_resident_if_rate_achieved"] is True
    assert report["verdict"]["quality"] == "UNPROVEN"
    assert report["claim_boundary"]["artifact_packed"] is False
    src_1m = report["residency"]["at_source_precision"]["1048576"]
    t15_1m = report["residency"]["at_1_5_uniform"]["1048576"]
    assert src_1m["fits_clean_exclusive"] is False
    assert t15_1m["fits_clean_exclusive"] is True
    assert t15_1m["margin_gib"] > 20.0
    receipt = REPO / "receipts/ascent-2026-08-16/dsv-resident-gravity.json"
    if receipt.is_file():
        import json

        written = json.loads(receipt.read_text(encoding="utf-8"))
        assert written["coverage"]["logical_params"] == cov["logical_params"]
        assert written["coverage"]["byte_residual"] == 0
        assert written["claim_boundary"]["coherence_generation_tested"] is False
