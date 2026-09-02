"""Pins for tools/future/complete_ebpw.py.

The resident must never compute BPW itself. An unreconciled candidate is
REFUSED, a dense-parent rematerializer is flagged rather than reported as
sub-2, and an aux-only byte cut reports 0.000 ms saved.
"""
from __future__ import annotations

import copy
import json

import pytest

from tools.future import complete_ebpw as ce
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def _inc() -> dict:
    return ce.incumbent_candidate()


def test_incumbent_reproduces_3_1393_bpw_from_its_parts():
    mix = ce.mix_report()
    row = ce.cost(_inc())
    assert row["reconciled"] is True
    assert row["stored_bytes"] == mix["payload_bytes"]
    assert row["executable_bytes"] == mix["payload_bytes"]
    assert row["parent_params"] == mix["parent_params"]
    assert row["complete_ebpw"] == pytest.approx(3.1393, abs=0.001)
    assert row["complete_ebpw"] == pytest.approx(mix["storage_bpw"], abs=1e-12)
    assert row["complete_ebpw"] == pytest.approx(
        mix["payload_bytes"] * 8.0 / mix["parent_params"]
    )
    parts_sum = sum(p["bytes"] for p in row["parts"])
    assert parts_sum == mix["payload_bytes"]
    mlp_body = ce.packed_bytes(
        elements=mix["mlp_elements"],
        bitwidth=mix["affine_bpw_billing"]["total_bpw"],
        what="test.mlp",
    )
    assert mlp_body + mix["q4_bytes"] + mix["f32_bytes"] + (
        mix["affine_bytes"] - mlp_body
    ) == mix["payload_bytes"]


def test_unreconciled_candidate_is_refused():
    cand = _inc()
    cand["id"] = "unreconciled"
    cand["stated_total_bytes"] = int(cand["stated_total_bytes"]) + 1
    with pytest.raises(ce.CompleteEbpwRefused, match="unreconciled"):
        ce.cost(cand)


def test_resident_mlp_only_2_25_against_payload_is_refused():
    """17.113e9 x 2.25 / 8 is not a complete executable and must not bill."""
    mix = ce.mix_report()
    cand = _inc()
    cand["id"] = "resident_rounded_mlp_only"
    cand["regions"] = [
        {
            "name": "mlp_rounded",
            "elements": 17_113_000_000,
            "bitwidth": 2.25,
            "stream_class": ce.STREAM_WEIGHT_CODES,
        }
    ]
    cand["metadata"] = []
    cand["stated_total_bytes"] = mix["payload_bytes"]
    with pytest.raises(ce.CompleteEbpwRefused, match="unreconciled|not an integer"):
        ce.cost(cand)


def test_dense_parent_rematerializing_candidate_is_flagged():
    inc = _inc()
    cand = {
        **inc,
        "id": "tiny_store_dense_parent",
        "stated_total_bytes": 1_000_000,
        "regions": [
            {
                "name": "codes",
                "bytes": 1_000_000,
                "stream_class": ce.STREAM_WEIGHT_CODES,
            }
        ],
        "generators": [],
        "metadata": [],
        "tables": [],
        "residuals": [],
        "runtime_auxiliaries": [],
        "reconstructs_dense_parent": True,
        "consumes_representation_directly": False,
        "parent_executable_bytes": int(inc["stated_total_bytes"]),
        "parent_stream_class": ce.STREAM_WEIGHT_CODES,
    }
    row = ce.cost(cand)
    assert ce.DENSE_PARENT_REMATERIALIZATION in row["flags"]
    assert row["dense_parent_rematerialization"]["flagged"] is True
    assert row["is_sub2_executable"] is False
    assert row["stored_bpw"] < 2.0
    assert row["complete_ebpw"] == pytest.approx(3.1393, abs=0.001)
    assert row["complete_ebpw"] > row["stored_bpw"]
    assert "not a sub-2 executable" in row["dense_parent_rematerialization"]["reason"]


def test_remat_complete_ebpw_is_not_the_stored_figure():
    inc = _inc()
    parent_bytes = int(inc["parent_params"]) * 2  # declared dense f16 parent
    cand = {
        **inc,
        "id": "dense_f16_remat",
        "stated_total_bytes": 4_000,
        "regions": [],
        "generators": [
            {
                "name": "seed",
                "bytes": 4_000,
                "stream_class": ce.STREAM_WEIGHT_CODES,
            }
        ],
        "metadata": [],
        "tables": [],
        "residuals": [],
        "runtime_auxiliaries": [],
        "reconstructs_dense_parent": True,
        "consumes_representation_directly": False,
        "parent_executable_bytes": parent_bytes,
        "parent_stream_class": ce.STREAM_WEIGHT_CODES,
    }
    row = ce.cost(cand)
    assert row["stored_bytes"] == 4_000
    assert row["executable_bytes"] == parent_bytes
    assert row["complete_ebpw"] == pytest.approx(16.0)
    assert row["is_sub2_executable"] is False
    assert ce.DENSE_PARENT_REMATERIALIZATION in row["flags"]


def test_aux_only_byte_cut_reports_zero_ms_saved():
    inc = _inc()
    header = int(inc["metadata"][0]["bytes"])
    assert header > 0
    cand = copy.deepcopy(inc)
    cand["id"] = "aux_only_header_cut"
    cand["metadata"] = [
        {
            "name": "mlp_headers",
            "bytes": 0,
            "stream_class": ce.STREAM_BROADCAST_AUX,
        }
    ]
    cand["stated_total_bytes"] = int(inc["stated_total_bytes"]) - header
    row = ce.cost(cand, versus=inc)
    assert row["versus"]["bytes_saved"] == header
    assert row["versus"]["ms_saved"] == 0.0
    assert row["versus"]["ms_saved"] == pytest.approx(0.000, abs=1e-12)
    assert row["by_stream_class"][ce.STREAM_BROADCAST_AUX]["ms"] == 0.0


def test_weight_codes_cut_saves_nonzero_ms():
    inc = _inc()
    codes = next(r for r in inc["regions"] if r["name"] == "mlp_codes")
    cut = copy.deepcopy(inc)
    cut["id"] = "mlp_codes_half"
    new_bytes = ce.packed_bytes(
        elements=codes["elements"], bitwidth=1.0, what="test.half"
    )
    saved = ce.packed_bytes(
        elements=codes["elements"], bitwidth=codes["bitwidth"], what="test.full"
    ) - new_bytes
    cut["regions"] = [
        (
            {
                **r,
                "bitwidth": 1.0,
            }
            if r["name"] == "mlp_codes"
            else r
        )
        for r in inc["regions"]
    ]
    cut["stated_total_bytes"] = int(inc["stated_total_bytes"]) - saved
    row = ce.cost(cut, versus=inc)
    assert row["versus"]["bytes_saved"] == saved
    assert row["versus"]["ms_saved"] > 0.0
    assert row["versus"]["ms_saved"] == pytest.approx(
        (saved / 1e9) * ce.CITED_WEIGHT_CODES_MS_PER_GB, abs=1e-6
    )


def test_missing_part_category_is_refused_not_defaulted():
    cand = _inc()
    del cand["generators"]
    with pytest.raises(ce.CompleteEbpwRefused, match="missing"):
        ce.cost(cand)


def test_missing_representation_category_is_refused():
    cand = _inc()
    del cand["representation"]
    with pytest.raises(ce.CompleteEbpwRefused, match="missing"):
        ce.cost(cand)


def test_missing_model_specific_code_category_is_refused():
    cand = _inc()
    del cand["model_specific_code"]
    with pytest.raises(ce.CompleteEbpwRefused, match="missing"):
        ce.cost(cand)


def test_empty_generators_list_is_explicit_and_ok():
    cand = _inc()
    assert cand["generators"] == []
    row = ce.cost(cand)
    assert row["reconciled"] is True
    assert all(p["category"] != "generators" for p in row["parts"])


def test_missing_stream_class_is_refused():
    cand = _inc()
    cand["tables"] = [{"name": "codebook", "bytes": 0}]
    with pytest.raises(ce.CompleteEbpwRefused, match="stream_class"):
        ce.cost(cand)


def test_unknown_stream_class_is_refused():
    cand = _inc()
    cand["tables"] = [
        {
            "name": "codebook",
            "bytes": 0,
            "stream_class": "organ_average",
        }
    ]
    with pytest.raises(ce.CompleteEbpwRefused, match="unknown stream_class"):
        ce.cost(cand)


def test_generator_codebook_and_lookup_table_all_bill():
    inc = _inc()
    extra = [
        ("generators", "block_generator", 10_000_000),
        ("tables", "codebook", 3_000_000),
        ("tables", "lookup_table", 2_000_000),
    ]
    cand = copy.deepcopy(inc)
    cand["id"] = "nothing_is_free"
    added = 0
    for category, name, n in extra:
        cand[category] = list(cand[category]) + [
            {"name": name, "bytes": n, "stream_class": ce.STREAM_WEIGHT_CODES}
        ]
        added += n
    cand["stated_total_bytes"] = int(inc["stated_total_bytes"]) + added
    row = ce.cost(cand, versus=inc)
    assert row["stored_bytes"] == int(inc["stated_total_bytes"]) + added
    assert row["versus"]["bytes_saved"] == -added
    assert row["versus"]["ms_saved"] < 0.0
    names = {p["name"] for p in row["parts"]}
    assert {"block_generator", "codebook", "lookup_table"} <= names


def test_unbilled_component_is_refused():
    """Hidden-free-information guard: a part-like sidecar must not skip billing.

    CALL SITE: ce.cost -> ce.refuse_unbilled_components. An import is not this.
    """
    cand = _inc()
    cand["sidecar_codebook"] = [
        {
            "name": "hidden_free_codebook",
            "bytes": 1_000,
            "stream_class": ce.STREAM_WEIGHT_CODES,
        }
    ]
    with pytest.raises(ce.CompleteEbpwRefused, match="unbilled component|hidden free"):
        ce.refuse_unbilled_components(cand)
    with pytest.raises(ce.CompleteEbpwRefused, match="unbilled component|hidden free"):
        ce.cost(cand)


def test_unbilled_guard_mutation_makes_refusal_fail(monkeypatch):
    """Load-bearing mutation check: removing refuse_unbilled_components lets a sidecar pass.

    test_unbilled_component_is_refused would FAIL if this guard were deleted.
    The monkeypatch is the mutation; it does not write the source file.
    """
    cand = _inc()
    cand["sidecar_codebook"] = [
        {
            "name": "hidden_free_codebook",
            "bytes": 1_000,
            "stream_class": ce.STREAM_WEIGHT_CODES,
        }
    ]
    with pytest.raises(ce.CompleteEbpwRefused, match="unbilled component|hidden free"):
        ce.refuse_unbilled_components(cand)

    monkeypatch.setattr(ce, "refuse_unbilled_components", lambda _c: None)
    # Guard gone: cost ACCEPTS the sidecar and never bills it. That is the
    # hidden-free-information defect the guard exists to stop.
    row = ce.cost(cand)
    assert row["reconciled"] is True
    names = {p["name"] for p in row["parts"]}
    assert "hidden_free_codebook" not in names


def test_candidate_from_parts_refuses_extra_category():
    parts = ce.empty_parts()
    parts["representation"] = [
        {"name": "codes", "bytes": 16, "stream_class": ce.STREAM_WEIGHT_CODES}
    ]
    parts["sidecar_codebook"] = [
        {"name": "hidden", "bytes": 8, "stream_class": ce.STREAM_WEIGHT_CODES}
    ]
    with pytest.raises(ce.CompleteEbpwRefused, match="unbilled component|hidden free"):
        ce.candidate_from_parts(family_id="probe", parent_params=16, parts=parts)


def test_compare_to_incumbent_uses_the_same_axes():
    inc = _inc()
    row = ce.compare_to_incumbent(inc)
    assert row["same_axes"] == list(ce.COMPARE_AXES)
    assert row["candidate_axes"].keys() == row["incumbent_axes"].keys()
    assert set(row["candidate_axes"]) == set(ce.COMPARE_AXES)
    assert row["versus"]["bytes_saved"] == 0
    assert row["versus"]["ms_saved"] == 0.0


def test_representation_and_model_specific_code_bill():
    inc = _inc()
    cand = copy.deepcopy(inc)
    cand["id"] = "family_payload_and_decoder"
    cand["representation"] = [
        {
            "name": "family_codes",
            "bytes": 6_000,
            "stream_class": ce.STREAM_WEIGHT_CODES,
        }
    ]
    cand["model_specific_code"] = [
        {
            "name": "decoder_stub",
            "bytes": 2_000,
            "stream_class": ce.STREAM_BROADCAST_AUX,
        }
    ]
    cand["stated_total_bytes"] = int(inc["stated_total_bytes"]) + 8_000
    row = ce.cost(cand, versus=inc)
    assert row["versus"]["bytes_saved"] == -8_000
    by_name = {p["name"]: p for p in row["parts"]}
    assert by_name["family_codes"]["bytes"] == 6_000
    assert by_name["family_codes"]["ms"] > 0.0
    assert by_name["decoder_stub"]["bytes"] == 2_000
    assert by_name["decoder_stub"]["ms"] == 0.0
    assert by_name["family_codes"]["category"] == "representation"
    assert by_name["decoder_stub"]["category"] == "model_specific_code"


def test_residual_and_runtime_aux_bill():
    inc = _inc()
    cand = copy.deepcopy(inc)
    cand["id"] = "residual_and_aux"
    cand["residuals"] = [
        {"name": "sparse_residual", "bytes": 8_000, "stream_class": ce.STREAM_WEIGHT_CODES}
    ]
    cand["runtime_auxiliaries"] = [
        {
            "name": "runtime_table",
            "bytes": 4_000,
            "stream_class": ce.STREAM_BROADCAST_AUX,
        }
    ]
    cand["stated_total_bytes"] = int(inc["stated_total_bytes"]) + 12_000
    row = ce.cost(cand, versus=inc)
    assert row["versus"]["bytes_saved"] == -12_000
    by_name = {p["name"]: p for p in row["parts"]}
    assert by_name["sparse_residual"]["bytes"] == 8_000
    assert by_name["sparse_residual"]["ms"] > 0.0
    assert by_name["runtime_table"]["bytes"] == 4_000
    assert by_name["runtime_table"]["ms"] == 0.0


def test_missing_mix_report_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "MIX_REPORT", tmp_path / "absent_MIX_REPORT.json")
    with pytest.raises(ce.CompleteEbpwRefused, match="incumbent artifact report"):
        ce.mix_report()


def test_missing_calibration_refuses(monkeypatch):
    monkeypatch.setattr(ce, "ECON_REL", "receipts/future/NO_SUCH_CALIBRATION.json")
    with pytest.raises(ce.CompleteEbpwRefused, match="stream-class ms/GB"):
        ce.stream_rates()


def test_stream_rates_are_weight_codes_0_547282_and_aux_0():
    rates = ce.stream_rates()
    assert rates[ce.STREAM_WEIGHT_CODES]["ms_per_gb"] == pytest.approx(
        0.547282, abs=1e-9
    )
    assert rates[ce.STREAM_BROADCAST_AUX]["ms_per_gb"] == 0.0
    assert rates[ce.STREAM_BROADCAST_AUX]["ms_per_gb"] == pytest.approx(
        0.000, abs=1e-12
    )


def test_disagreeing_bytes_and_bitwidth_refused():
    cand = _inc()
    cand["regions"] = [
        {
            "name": "mlp_codes",
            "elements": 64,
            "bitwidth": 2.0,
            "bytes": 17,
            "stream_class": ce.STREAM_WEIGHT_CODES,
        }
    ]
    cand["metadata"] = []
    cand["stated_total_bytes"] = 16
    with pytest.raises(ce.CompleteEbpwRefused, match="does not reconcile"):
        ce.cost(cand)


def test_remat_without_parent_bytes_is_refused():
    inc = _inc()
    cand = {
        **inc,
        "id": "remat_no_parent",
        "reconstructs_dense_parent": True,
        "consumes_representation_directly": False,
    }
    with pytest.raises(ce.CompleteEbpwRefused, match="parent_executable_bytes"):
        ce.cost(cand)


def test_build_writes_parseable_receipt():
    rc = ce.main(["--build"])
    assert rc == 0
    path = RECEIPTS / ce.RECEIPT
    doc = json.loads(path.read_text())
    assert doc["schema"] == ce.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["selftest"]["incumbent_bpw_within_0_001_of_3_1393"] is True
    assert doc["selftest"]["unreconciled_refused"] is True
    assert doc["selftest"]["dense_parent_rematerialization_flagged"] is True
    assert doc["selftest"]["aux_only_cut_ms_saved_is_zero"] is True
    assert doc["selftest"]["unbilled_component_refused"] is True
    assert doc["incumbent"]["complete_ebpw"] == pytest.approx(3.1393, abs=0.001)
    _assert_no_hardware_claims(doc)
