"""Tests for the MLP auxiliary-information attack on the 1.07 GB.

A guard nobody has watched fail is not a guard. The load-bearing refusal:
scale+bias+header that do not sum to 1,069,605,696 raise, they do not
silently pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import mlp_auxiliary_information as mai
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def test_unreconciled_auxiliary_raises_and_does_not_silently_pass():
    """NEGATIVE CONTROL: broken scale/bias/header accounting must refuse."""
    with pytest.raises(mai.UnreconciledAuxiliary) as caught:
        mai.reconcile_auxiliary(0, 0, 0)
    assert caught.value.got == 0
    assert caught.value.want == mai.AUXILIARY_BYTES_TARGET
    assert "REFUSED" in str(caught.value)
    assert str(mai.AUXILIARY_BYTES_TARGET) in str(caught.value)

    with pytest.raises(mai.UnreconciledAuxiliary):
        mai.reconcile_auxiliary(mai.AUXILIARY_BYTES_TARGET - 1, 0, 0)

    with pytest.raises(mai.UnreconciledAuxiliary):
        mai.reconcile_auxiliary(
            mai.AUXILIARY_BYTES_TARGET,
            1,
            0,
            want=mai.AUXILIARY_BYTES_TARGET,
        )

    # Real rows, then one extra scale byte. Must not pass.
    rows = mai.auxiliary_rows()
    mutated = [dict(r) for r in rows]
    mutated[0]["scale_bytes"] = int(mutated[0]["scale_bytes"]) + 1
    with pytest.raises(mai.UnreconciledAuxiliary) as caught2:
        mai.accounting_from_rows(mutated)
    assert caught2.value.got == mai.AUXILIARY_BYTES_TARGET + 1
    assert caught2.value.want == mai.AUXILIARY_BYTES_TARGET


def test_accounting_reconciles_to_recorded_auxiliary_total():
    snap = mai.accounting()
    assert snap["auxiliary_bytes"] == mai.AUXILIARY_BYTES_TARGET
    assert snap["header_bytes"] + snap["scale_bytes"] + snap["bias_bytes"] == (
        mai.AUXILIARY_BYTES_TARGET
    )
    assert snap["code_bytes"] == mai.CODE_BYTES_TARGET
    assert snap["stored_bytes"] == mai.MLP_ACTIVE_TARGET
    assert snap["n_tensors"] == 192
    assert snap["group_size"] == 64
    assert snap["header_bytes"] == 58_176
    assert snap["scale_bytes"] == snap["bias_bytes"] == 534_773_760
    assert snap["reconciled"] is True
    # Headers are the small slice, not the 1.07 GB.
    assert snap["header_bytes"] < 100_000
    assert snap["header_share_of_auxiliary"] < 0.001


def test_real_hgrafv01_header_is_json_plus_f16_aux_plus_codes():
    rows = mai.auxiliary_rows()
    assert len(rows) == 192
    assert {r["header_bytes"] for r in rows} == {303}
    assert {r["group_size"] for r in rows} == {64}
    gate0 = next(r for r in rows if r["layer"] == 0 and r["organ"] == "mlp.gate")
    assert gate0["shape"] == [17408, 5120]
    assert gate0["groups"] == 1_392_640
    assert gate0["scale_bytes"] == 2_785_280
    assert gate0["code_bytes"] == 22_282_240
    down0 = next(r for r in rows if r["layer"] == 0 and r["organ"] == "mlp.down")
    assert down0["shape"] == [5120, 17408]


def test_group_size_curve_is_exact_and_includes_incumbent():
    acc = mai.accounting()
    geo = acc["identity"]["geometry"]
    curve = mai.group_size_byte_curve(
        n_parameters=acc["n_parameters"],
        n_tensors=acc["n_tensors"],
        header_bytes_per_tensor=acc["header_bytes"] // acc["n_tensors"],
        hidden=geo["hidden_size"],
        intermediate=geo["intermediate_size"],
    )
    by_g = {r["group_size"]: r for r in curve}
    assert 64 in by_g and 128 in by_g and 256 in by_g and 1024 in by_g
    inc = by_g[64]
    assert inc["incumbent"] is True
    assert inc["auxiliary_bytes"] == mai.AUXILIARY_BYTES_TARGET
    assert inc["bytes_eliminated_vs_incumbent"] == 0
    assert inc["capability"] == "UNMEASURED"
    # Larger G strictly reduces aux bytes; smaller G grows them.
    assert by_g[32]["auxiliary_bytes"] > inc["auxiliary_bytes"]
    assert by_g[128]["auxiliary_bytes"] < inc["auxiliary_bytes"]
    assert by_g[128]["bytes_eliminated_vs_incumbent"] == (
        inc["auxiliary_bytes"] - by_g[128]["auxiliary_bytes"]
    )
    # Codes are independent of G.
    assert by_g[32]["code_bytes"] == by_g[1024]["code_bytes"] == mai.CODE_BYTES_TARGET


def _snap() -> dict:
    return mai.snapshot(consult_index=False)


def test_measurements_come_from_real_arrays_not_a_model():
    meas = _snap()["measurements"]
    assert meas["n_tensors_measured"] == 192
    # Independent-random f16 would sit near 16 bits with tens of thousands
    # of unique patterns per tensor. The arrays do not.
    assert 1.0 < meas["scale_shannon_bits"] < 14.0
    assert meas["scale_unique_f16"] < 20_000
    assert meas["bias_unique_f16"] < 30_000
    gate = meas["by_organ"]["mlp.gate"]
    assert gate["n_tensors"] == 64
    assert gate["lag1_along_groups"] is not None
    assert abs(gate["bias_over_scale_median"] - (-1.5)) < 0.05
    # DC dominates uncentered rank-1; variation is not low-rank.
    assert gate["uncentered_svd_rank1"] > 0.9
    assert gate["centered_svd_rank1"] < 0.5
    recon = meas["reconstruction_relfro"]
    assert recon["drop_bias_relfro"] > 1.0
    assert recon["tied_bias_relfro"] > 0.1
    assert recon["u8_log_scale_keep_bias_relfro"] < 0.05
    assert recon["u8_linear_bias_keep_scale_relfro"] < 0.05
    code = meas["code_prediction"]
    assert abs(code["corr_scale_qmean"]) < 0.05
    assert code["scale_r2_qmean_qvar"] < 0.6


def test_candidates_cover_the_contract_questions():
    cands = _snap()["candidates"]
    ids = [c["id"] for c in cands]
    assert ids == list(mai.REQUIRED_CANDIDATE_IDS)
    for row in cands:
        assert row["mechanism"]
        assert row["byte_model"]
        assert row["cheapest_falsifier"]
        assert row["dense_rematerialization"] in {
            mai.DIRECT_CONSUME,
            mai.REJECTED_DENSE_REMAT,
            mai.DEPENDS_ON_LOWERING,
        }
        assert row["status"] in {
            mai.ALREADY_FALSIFIED,
            mai.MEASURED_NEGATIVE,
            mai.OPEN,
        }
        assert row["evidence_class"] == "STATIC_ONLY"
        assert row["gpu_authority"] is False
        if row["dense_rematerialization"] != mai.REJECTED_DENSE_REMAT:
            assert row["physical_primitive"] in ATLAS_PRIMITIVES


def test_measured_negatives_are_the_array_kills():
    by_id = {c["id"]: c for c in _snap()["candidates"]}
    for cid in (
        "shared_scale_basis",
        "per_tensor_curve_plus_residual",
        "predict_scale_from_code_stats",
        "low_rank_scale_matrix",
        "parametric_scale_program",
        "tie_bias_to_minus_half_codes",
        "drop_bias",
        "collapse_to_global_scale",
        "cross_layer_scale_delta",
    ):
        assert by_id[cid]["status"] == mai.MEASURED_NEGATIVE, cid
    assert by_id["quantize_aux_u8"]["status"] == mai.OPEN
    assert by_id["larger_group_size"]["status"] == mai.OPEN
    assert by_id["pack_headers"]["status"] == mai.OPEN
    # Header pack is real and tiny.
    assert by_id["pack_headers"]["bytes_eliminated_if_true"] == 52_032
    # u8 both halves the two f16 arrays.
    assert by_id["quantize_aux_u8"]["bytes_eliminated_if_true"] == 534_773_760
    # Dropping bias would be half a gig if it were legal. It is not.
    assert by_id["drop_bias"]["bytes_eliminated_if_true"] == 534_773_760


def test_no_candidate_is_silent_dense_w_remat():
    cands = _snap()["candidates"]
    remat = {c["id"] for c in cands if c["dense_rematerialization"] == mai.REJECTED_DENSE_REMAT}
    assert remat == set()
    # Generating a full scale buffer is called out as not an active-byte win.
    parametric = next(c for c in cands if c["id"] == "parametric_scale_program")
    assert parametric["dense_rematerialization"] == mai.DEPENDS_ON_LOWERING
    assert "active" in parametric["dense_rematerialization_reason"].lower()


def test_answers_section_matches_candidates():
    snap = _snap()
    ans = snap["answers"]
    assert ans["are_scales_independently_random"]["answer"].startswith("NO")
    assert ans["can_scales_be_predicted_from_codes"]["status"] == mai.MEASURED_NEGATIVE
    assert ans["are_biases_necessary"]["drop_status"] == mai.MEASURED_NEGATIVE
    assert ans["can_headers_be_packed_harder"]["header_bytes"] == 58_176
    assert ans["can_group_size_change"]["capability"] == "UNMEASURED"


def test_negative_index_is_queried_and_does_not_launder_w_scars():
    snap = mai.snapshot(consult_index=True)
    by_id = {c["id"]: c for c in snap["candidates"]}
    shared = by_id["shared_scale_basis"]
    # A W-space shared-basis scar may appear as a cousin. It must not flip
    # this scale-array candidate to ALREADY_FALSIFIED.
    assert shared["status"] == mai.MEASURED_NEGATIVE
    assert shared.get("cousin_not_this_object") is True
    hits = shared.get("index_refusals") or []
    if hits:
        assert shared.get("index_hits_are_cousins") is True
        for h in hits:
            assert h.get("applies_to_this_object") is False
    collapse = by_id["collapse_to_global_scale"]
    assert any(c.get("scar_id") == "NNS-029" for c in collapse.get("citations") or [])


def test_build_emits_sealed_receipt():
    out = mai.build(consult_index=True)
    assert out.parent == RECEIPTS
    assert out.name == "MLP_AUXILIARY_INFORMATION.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == mai.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    _assert_no_hardware_claims(doc)
    assert doc["accounting"]["auxiliary_bytes"] == mai.AUXILIARY_BYTES_TARGET
    assert doc["accounting"]["reconciled"] is True
    assert [c["id"] for c in doc["candidates"]] == list(mai.REQUIRED_CANDIDATE_IDS)
    assert doc["candidate_counts"]["n"] == len(mai.REQUIRED_CANDIDATE_IDS)
    assert any(c["id"] == "quantize_aux_u8" for c in doc["open_byte_levers"])


def test_module_entrypoint_runs_and_emits_sealed_receipt():
    rc = mai.main(["--build"])
    assert rc == 0
    doc = json.loads((RECEIPTS / mai.RECEIPT).read_text())
    assert doc["schema"] == mai.SCHEMA
    assert doc["seal_sha256"]


def test_selftest_aliases_build():
    assert mai.selftest is mai.build or mai.selftest().name == mai.RECEIPT


def test_hardware_fields_stay_non_numeric_on_the_receipt():
    out = mai.build(consult_index=True)
    doc = json.loads(out.read_text())
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:

        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    here = f"{path}.{k}" if path else k
                    if k in HARDWARE_FIELDS:
                        assert not isinstance(v, (int, float)) or isinstance(v, bool), here
                    walk(v, here)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(doc)


def test_cli_entrypoint_actually_runs():
    import subprocess as _sp

    root = Path(__file__).resolve().parents[2]
    p = _sp.run(
        ["python3", "tools/future/mlp_auxiliary_information.py", "--accounting-only"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr[-800:]
    doc = json.loads(p.stdout)
    assert doc["auxiliary_bytes"] == mai.AUXILIARY_BYTES_TARGET
    assert doc["reconciled"] is True
