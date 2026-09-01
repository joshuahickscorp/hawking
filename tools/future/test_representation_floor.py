"""The incumbent floor is derived from receipts, and REFUTED bytes cannot shrink it.

The two ways this module could lie are a typed payload and a REFUTED group-size
cut counted as a measured-safe win. Both are pinned here.
"""
from __future__ import annotations

import json

import pytest

from tools.future import representation_floor as rf


def test_a_missing_receipt_refuses_rather_than_defaulting(monkeypatch):
    monkeypatch.setattr(rf, "GAP_REL", "receipts/future/NO_SUCH_GAP.json")
    with pytest.raises(rf.FloorRefused, match="NO_SUCH_GAP"):
        rf.sixty_tps()


def test_a_missing_mix_report_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(rf, "MIX_REPORT", tmp_path / "absent_MIX_REPORT.json")
    with pytest.raises(rf.FloorRefused, match="incumbent artifact report"):
        rf.mix_report()


def test_byte_classes_sum_to_the_mix_payload():
    mix = rf.mix_report()
    bc = rf.byte_classes()
    assert bc["payload_bytes"] == mix["payload_bytes"]
    assert sum(c["bytes"] for c in bc["classes"]) == mix["payload_bytes"]
    assert bc["mlp_storage_bytes"] == mix["affine_bytes"]
    assert bc["q4_code_bytes"] + bc["q4_aux_bytes"] == mix["q4_bytes"]
    assert bc["f32_bytes"] == mix["f32_bytes"]


def test_broadcast_aux_is_billed_at_zero_and_weight_codes_are_not():
    rates = rf.stream_rates()
    assert rates[rf.STREAM_BROADCAST_AUX]["ms_per_gb"] == 0.0
    assert rates[rf.STREAM_WEIGHT_CODES]["ms_per_gb"] == pytest.approx(0.547282, abs=1e-9)
    assert rates[rf.STREAM_ACTIVATION]["ms_per_gb"] == pytest.approx(2.906132, abs=1e-9)
    aux = next(c for c in rf.byte_classes()["classes"] if c["id"] == "mlp_broadcast_aux")
    assert aux["ms_per_token_if_streamed"] == 0.0


def test_every_candidate_has_evidence_status_and_a_receipt_source():
    rows = rf.candidates()
    assert len(rows) >= 10
    for c in rows:
        assert c["evidence_status"] in rf.EVIDENCE_STATUSES
        assert c["source"].startswith("receipts/")
        assert "bytes_saved" in c and "ms_saved" in c
        assert c["stream_class"] in {
            rf.STREAM_WEIGHT_CODES, rf.STREAM_BROADCAST_AUX, rf.STREAM_ACTIVATION,
        }


def test_group_size_256_and_1024_are_refuted():
    rows = {c["id"]: c for c in rf.candidates()}
    assert rows["larger_affine_group_256"]["evidence_status"] == rf.REFUTED
    assert rows["larger_affine_group_1024"]["evidence_status"] == rf.REFUTED
    assert rows["composite_mlp_simple_linear_low_rank"]["evidence_status"] == rf.REFUTED


def test_a_refuted_candidate_cannot_count_toward_the_measured_safe_floor():
    rows = rf.candidates()
    fl = rf.floor_from(rows)
    refuted = [c for c in rows if c["evidence_status"] == rf.REFUTED]
    assert refuted, "the screen must have produced at least one REFUTED cut"
    for c in refuted:
        assert c["id"] not in fl["measured_safe_moves"]
        assert c["counts_toward_measured_safe"] is False
        assert c["id"] in fl["refuted_ids_excluded"]
    stuffed = []
    for c in rows:
        if c["id"] == "larger_affine_group_256":
            stuffed.append({**c, "bytes_saved": 10**12, "gb_saved": 1000.0})
        else:
            stuffed.append(c)
    stuffed_floor = rf.floor_from(stuffed)
    assert stuffed_floor["measured_safe_bytes"] == fl["measured_safe_bytes"]
    assert "larger_affine_group_256" not in stuffed_floor["measured_safe_moves"]
    assert stuffed_floor["if_every_untested_move_worked_bytes"] == fl[
        "if_every_untested_move_worked_bytes"
    ]


def test_measured_safe_is_the_incumbent_minus_only_measured_code_entropy():
    mix = rf.mix_report()
    codes = json.loads((rf.REPO / rf.CODE_REL).read_text())
    saved = int(codes["floor"]["iid_redundant_bytes_rounded"])
    fl = rf.floor_from()
    assert fl["measured_safe_moves"] == ["entropy_code_mlp_codes"]
    assert fl["measured_safe_bytes"] == mix["payload_bytes"] - saved
    assert fl["measured_safe_gb"] == pytest.approx(fl["measured_safe_bytes"] / 1e9, abs=1e-6)
    assert fl["measured_safe_bpw"] < fl["incumbent_bpw"]
    assert fl["if_every_untested_move_worked_bytes"] < fl["measured_safe_bytes"]
    assert fl["if_every_untested_move_worked_gb"] < fl["measured_safe_gb"]


def test_floor_object_has_the_required_keys():
    fl = rf.build()["floor"]
    for k in (
        "measured_safe_gb", "measured_safe_bpw",
        "if_every_untested_move_worked_gb", "if_every_untested_move_worked_bpw",
    ):
        assert k in fl
        assert isinstance(fl[k], float)
        assert fl[k] > 0


def test_conventional_compression_cannot_reach_60_tps_and_shows_the_arithmetic():
    doc = rf.build()
    assert doc["can_conventional_compression_reach_60_tps"] is False
    assert doc["shortfall_gb"] > 0
    a = doc["sixty"]["arithmetic"]
    assert a["fraction_of_matvec_bytes_to_remove"] == pytest.approx(0.1677, abs=1e-4)
    assert a["streaming_ms_at_arm_a"] == pytest.approx(15.269, abs=1e-3)
    assert a["measured_conventional_code_bytes"] < a["bytes_required_at_fraction_of_codes"]
    assert a["untested_conventional_code_bytes"] < a["bytes_required_at_fraction_of_codes"]
    assert "INFORMATION ELIMINATION" in doc["sixty"]["what_class_of_change_could"]
    assert a["source"] == rf.GAP_REL


def test_worth_it_is_no_under_the_s025_ms_bar():
    w = rf.worth_it()
    assert w["verdict"] == "NO"
    assert w["materiality_threshold_ms"] == 1.0
    assert w["expected_ms_per_token"] < w["materiality_threshold_ms"]
    assert w["expected_ms_per_token_if_every_untested_worked"] < w["materiality_threshold_ms"]
    assert "0.000 ms/GB" in w["why"] or "0.000" in w["cost_basis"]


def test_entropy_bytes_and_independence_are_read_not_typed():
    codes = json.loads((rf.REPO / rf.CODE_REL).read_text())
    ent = next(c for c in rf.candidates() if c["id"] == "entropy_code_mlp_codes")
    assert ent["bytes_saved"] == codes["floor"]["iid_redundant_bytes_rounded"]
    assert ent["H_q_bits"] == codes["floor"]["iid_shannon_bits_per_code"]
    assert ent["independent_fraction"] == pytest.approx(0.935, abs=1e-3)
    assert ent["evidence_status"] == rf.MEASURED
    assert ent["ms_saved"] == pytest.approx(
        (ent["bytes_saved"] / 1e9) * rf.stream_rates()[rf.STREAM_WEIGHT_CODES]["ms_per_gb"],
        abs=1e-6,
    )


def test_aux_cuts_save_size_not_time():
    rows = {c["id"]: c for c in rf.candidates()}
    for ident in (
        "mlp_to_2_25_coherent_floor", "larger_affine_group_128",
        "larger_affine_group_256", "quantize_aux_u8", "larger_q4_group_128",
    ):
        assert rows[ident]["ms_saved"] == 0.0
        assert rows[ident]["bytes_saved"] > 0
        assert rows[ident]["stream_class"] == rf.STREAM_BROADCAST_AUX


def test_bitcast_is_measured_time_with_zero_bytes():
    rows = {c["id"]: c for c in rf.candidates()}
    assert rows["q2_bitcast_dequant"]["bytes_saved"] == 0
    assert rows["q2_bitcast_dequant"]["gpu_ms_saved_measured"] == pytest.approx(3.8541, abs=1e-4)
    assert rows["q2_bitcast_dequant"]["token_identical"] is True
    assert rows["q4_bitcast_dequant"]["gpu_ms_saved_measured"] == pytest.approx(0.6836, abs=1e-4)
    assert rows["q4_bitcast_dequant"]["bit_identical"] is True
    assert rows["q2_bitcast_dequant"]["id"] not in rf.floor_from()["measured_safe_moves"]


def test_self_inconsistent_mix_payload_is_refused(monkeypatch):
    real_load = rf._load

    def fake_load(rel, *, why):
        d = real_load(rel, why=why)
        if str(rel) == str(rf.MIX_REPORT):
            return {**d, "f32_bytes": int(d["f32_bytes"]) + 1}
        return d

    monkeypatch.setattr(rf, "_load", fake_load)
    with pytest.raises(rf.FloorRefused, match="does not add"):
        rf.mix_report()


def test_build_receipt_shape_and_load_bearing_sources():
    doc = rf.build()
    assert doc["can_conventional_compression_reach_60_tps"] is False
    assert "shortfall_gb" in doc
    assert doc["worth_it"]["materiality_threshold_ms"] == 1.0
    assert doc["worth_it"]["verdict"] == "NO"
    for row in doc["load_bearing"]:
        assert "source" in row and "value" in row and "id" in row
    for c in doc["candidates"]:
        assert c["source"].startswith("receipts/")
        assert c["evidence_status"] in rf.EVIDENCE_STATUSES
