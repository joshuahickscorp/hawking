"""Tests for the MLP code-body information attack on the 4.28 GB of 2-bit codes.

A guard nobody has watched fail is not a guard. The load-bearing refusal:
code bytes that do not sum to 4,278,190,080 raise, they do not silently pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import mlp_code_information as mci
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def test_unreconciled_code_bytes_raises_and_does_not_silently_pass():
    """NEGATIVE CONTROL: broken code-byte accounting must refuse."""
    with pytest.raises(mci.UnreconciledCode) as caught:
        mci.reconcile_code_bytes(0)
    assert caught.value.got == 0
    assert caught.value.want == mci.CODE_BYTES_TARGET
    assert "REFUSED" in str(caught.value)
    assert str(mci.CODE_BYTES_TARGET) in str(caught.value)

    with pytest.raises(mci.UnreconciledCode):
        mci.reconcile_code_bytes(mci.CODE_BYTES_TARGET - 1)

    with pytest.raises(mci.UnreconciledCode):
        mci.reconcile_code_bytes(
            mci.CODE_BYTES_TARGET,
            want=mci.CODE_BYTES_TARGET - 1,
        )

    rows = mci.code_rows()
    mutated = [dict(r) for r in rows]
    mutated[0]["code_bytes"] = int(mutated[0]["code_bytes"]) + 1
    with pytest.raises(mci.UnreconciledCode) as caught2:
        mci.accounting_from_rows(mutated)
    assert caught2.value.got == mci.CODE_BYTES_TARGET + 1
    assert caught2.value.want == mci.CODE_BYTES_TARGET


def test_accounting_reconciles_to_recorded_code_total():
    snap = mci.accounting()
    assert snap["code_bytes"] == mci.CODE_BYTES_TARGET
    assert snap["n_tensors"] == 192
    assert snap["group_size"] == 64
    assert snap["code_bits"] == 2
    assert snap["n_parameters"] == 17_112_760_320
    assert snap["n_groups"] * snap["group_size"] == snap["n_parameters"]
    assert (snap["n_parameters"] * 2) // 8 == mci.CODE_BYTES_TARGET
    assert snap["stored_bytes"] == mci.MLP_ACTIVE_TARGET
    assert snap["auxiliary_bytes"] == mci.AUXILIARY_BYTES_TARGET
    assert snap["header_bytes"] + snap["scale_bytes"] + snap["bias_bytes"] + snap["code_bytes"] == (
        snap["stored_bytes"]
    )
    assert snap["reconciled"] is True
    assert snap["code_share_of_mlp"] == pytest.approx(0.8, abs=0.001)


def test_real_hgrafv01_code_bytes_match_shape():
    rows = mci.code_rows()
    assert len(rows) == 192
    gate0 = next(r for r in rows if r["layer"] == 0 and r["organ"] == "mlp.gate")
    assert gate0["shape"] == [17408, 5120]
    assert gate0["code_bytes"] == 22_282_240
    assert mci.expected_code_bytes(gate0["shape"]) == gate0["code_bytes"]
    down0 = next(r for r in rows if r["layer"] == 0 and r["organ"] == "mlp.down")
    assert down0["shape"] == [5120, 17408]
    assert down0["code_bytes"] == gate0["code_bytes"]


def _snap() -> dict:
    return mci.snapshot(consult_index=False)


def test_measurements_come_from_real_code_arrays_not_a_model():
    meas = _snap()["measurements"]
    assert meas["n_tensors_measured"] == 192
    assert meas["code_bytes_read"] == mci.CODE_BYTES_TARGET
    assert meas["n_parameters"] == 17_112_760_320
    # Affine-Q2 of roughly-Gaussian W is biased, not uniform, not degenerate.
    assert 1.80 < meas["H_q_bits"] < 1.95
    assert meas["q_hist"][1] > meas["q_hist"][0]
    assert meas["q_hist"][2] > meas["q_hist"][3]
    assert abs(meas["p_q"][0] - meas["p_q"][3]) < 0.01
    # Neighbours add almost nothing.
    assert meas["mi_within_byte_bits"] < 0.01
    assert abs(meas["H_q_given_prev_within_byte"] - meas["H_q_bits"]) < 0.01
    # Packed-byte entropy tracks 4 independent codes, not a product codebook.
    assert abs(meas["H_byte_bits"] - meas["H_byte_if_iid_q"]) < 0.05
    assert meas["byte_occupied_of_256"] > 200
    # Sharing is not in the arrays.
    gate_mi = meas["cross_layer"]["mlp.gate"]["mi_bits"]["mean"]
    down_mi = meas["cross_layer"]["mlp.down"]["mi_bits"]["mean"]
    gu_mi = meas["cross_tensor_gate_vs_up"]["mi_bits"]["mean"]
    assert gate_mi < 0.01
    assert down_mi < 0.01
    assert gu_mi < 0.01
    # Independent baseline explains the match rate.
    match = meas["cross_layer"]["mlp.gate"]["match"]["mean"]
    indep = meas["cross_layer"]["mlp.gate"]["independent_match"]["mean"]
    assert abs(match - indep) < 0.01
    # Groups do not repeat; rows are not 1-bit islands.
    assert meas["sample_unique_frac"]["min"] > 0.999
    assert meas["sample_rowH_frac_lt_1_5"]["max"] == 0.0
    assert meas["sample_rowH_min"]["min"] > 1.5
    assert 0.90 < meas["sample_zlib_ratio"]["mean"] < 0.98
    # NNS-022 reopen does not fire.
    assert meas["H_q_over_uniform"] > mci.NNS022_REOPEN_UNIFORM_FRAC
    assert meas["nns022_reopen_fires"] is False


def test_entropy_floor_is_four_gigabytes_not_a_vibe():
    snap = _snap()
    floor = snap["floor"]
    meas = snap["measurements"]
    assert floor["stored_code_bytes"] == mci.CODE_BYTES_TARGET
    assert floor["iid_shannon_bytes_rounded"] == meas["iid_shannon_bytes_rounded"]
    assert floor["iid_redundant_bytes_rounded"] == meas["iid_redundant_bytes_rounded"]
    # 4.00 of 4.28 GB independent; ~278 MB histogram bias.
    assert 3_950_000_000 < floor["iid_shannon_bytes_rounded"] < 4_100_000_000
    assert 200_000_000 < floor["iid_redundant_bytes_rounded"] < 350_000_000
    assert floor["iid_shannon_bytes_rounded"] + floor["iid_redundant_bytes_rounded"] == (
        mci.CODE_BYTES_TARGET
    ) or abs(
        floor["iid_shannon_bytes_rounded"] + floor["iid_redundant_bytes_rounded"] - mci.CODE_BYTES_TARGET
    ) <= 1
    assert 0.92 < floor["independent_fraction"] < 0.95
    assert "independent information" in floor["verdict"]


def test_candidates_cover_the_contract_questions():
    cands = _snap()["candidates"]
    ids = [c["id"] for c in cands]
    assert ids == list(mci.REQUIRED_CANDIDATE_IDS)
    assert not (set(ids) & set(mci.AUXILIARY_DEAD_IDS))
    legal_status = {
        mci.ALREADY_FALSIFIED,
        mci.MEASURED_NEGATIVE,
        mci.OPEN,
        mci.UNMEASURED,
        mci.REJECTED_DENSE_REMAT,
    }
    legal_remat = {
        mci.DIRECT_CONSUME,
        mci.REJECTED_DENSE_REMAT,
        mci.DEPENDS_ON_LOWERING,
    }
    for row in cands:
        assert row["mechanism"]
        assert row["byte_model"]
        assert row["cheapest_falsifier"]
        assert row["dense_rematerialization"] in legal_remat
        assert row["status"] in legal_status
        assert row["evidence_class"] == "STATIC_ONLY"
        assert row["gpu_authority"] is False
        assert row.get("support") in {
            "MEASURED",
            "UNMEASURED",
            "SCAR",
            "SHANNON_GAP_MEASURED_KERNEL_UNMEASURED",
        }
        if row["dense_rematerialization"] != mci.REJECTED_DENSE_REMAT:
            assert row["physical_primitive"] in ATLAS_PRIMITIVES


def test_sharing_and_dictionary_families_are_measured_negative():
    by_id = {c["id"]: c for c in _snap()["candidates"]}
    for cid in (
        "heterogeneous_bit_allocation",
        "generated_programs",
        "shared_code_bases",
        "factorized_programs",
        "dictionary_of_code_blocks",
        "product_codebooks",
        "lowrank_plus_sparse_residual",
        "block_generators",
        "cross_layer_code_prediction",
    ):
        assert by_id[cid]["status"] == mci.MEASURED_NEGATIVE, cid
        assert by_id[cid]["support"] == "MEASURED", cid
    # Generated W is refused as a production path before a fit.
    assert by_id["generated_tensors"]["status"] == mci.REJECTED_DENSE_REMAT
    assert by_id["generated_tensors"]["dense_rematerialization"] == mci.REJECTED_DENSE_REMAT
    # Dictionary / product cheap lowering expands W.
    assert by_id["dictionary_of_code_blocks"]["dense_rematerialization"] == mci.REJECTED_DENSE_REMAT
    assert by_id["product_codebooks"]["dense_rematerialization"] == mci.REJECTED_DENSE_REMAT


def test_scars_are_cited_not_laundered_and_aux_families_are_absent():
    snap = _snap()
    ids = {c["id"] for c in snap["candidates"]}
    for dead in mci.AUXILIARY_DEAD_IDS:
        assert dead not in ids
    by_id = {c["id"]: c for c in snap["candidates"]}
    assert by_id["lower_bit_native"]["status"] == mci.ALREADY_FALSIFIED
    assert by_id["capability_sensitive_literal_islands"]["status"] == mci.ALREADY_FALSIFIED
    assert by_id["latent_routed_accumulation"]["status"] == mci.ALREADY_FALSIFIED

    def _ids(cid: str) -> set[str]:
        return {c["scar_id"] for c in by_id[cid].get("citations") or []}

    assert "NNS-029" in _ids("lower_bit_native")
    assert "QN-BINARY-INJURY" in _ids("lower_bit_native")
    assert "QN-BINARY-HEALING" in _ids("capability_sensitive_literal_islands")
    assert "NNS-013" in _ids("latent_routed_accumulation")
    assert "NNS-022" in _ids("entropy_coded_code_stream")
    # W-space shared-basis is a cousin of the code-array measurement.
    shared = by_id["shared_code_bases"]
    assert shared["status"] == mci.MEASURED_NEGATIVE
    assert shared.get("cousin_not_this_object") is True
    assert "QN-SHARED-BASIS-DENSITY" in _ids("shared_code_bases")


def test_unmeasured_families_are_explicit_not_supported():
    by_id = {c["id"]: c for c in _snap()["candidates"]}
    for cid in ("shared_input_transforms", "function_replacement"):
        assert by_id[cid]["status"] == mci.UNMEASURED, cid
        assert by_id[cid]["support"] == "UNMEASURED", cid
        assert "UNMEASURED" in by_id[cid]["cheapest_falsifier"] or "not a code-stream" in (
            by_id[cid]["cheapest_falsifier"].lower()
        )
    # Entropy coding is the only OPEN lever, and it is not a 4 GB win.
    ent = by_id["entropy_coded_code_stream"]
    assert ent["status"] == mci.OPEN
    assert ent["support"] == "SHANNON_GAP_MEASURED_KERNEL_UNMEASURED"
    assert ent["bytes_eliminated_if_true"] == _snap()["measurements"]["iid_redundant_bytes_rounded"]
    assert ent["dense_rematerialization"] == mci.DEPENDS_ON_LOWERING
    assert ent["bytes_eliminated_if_true"] < 350_000_000


def test_answers_section_matches_measurements():
    snap = _snap()
    ans = snap["answers"]
    indep = ans["how_much_of_the_code_body_is_independent"]
    assert indep["stored_code_bytes"] == mci.CODE_BYTES_TARGET
    assert "93" in indep["answer"] or "independent" in indep["answer"].lower()
    assert ans["is_entropy_coding_a_4gb_lever"]["status"] == mci.OPEN
    assert ans["do_the_codes_share_across_layers_or_organs"]["status"] == mci.MEASURED_NEGATIVE
    assert ans["can_a_dictionary_or_block_generator_win"]["dictionary_status"] == mci.MEASURED_NEGATIVE
    assert ans["what_the_codes_do_not_measure"]["function_replacement"] == mci.UNMEASURED


def test_negative_index_is_queried_and_does_not_launder_w_scars():
    snap = mci.snapshot(consult_index=True)
    by_id = {c["id"]: c for c in snap["candidates"]}
    shared = by_id["shared_code_bases"]
    assert shared["status"] == mci.MEASURED_NEGATIVE
    hits = shared.get("index_refusals") or []
    if hits:
        assert shared.get("index_hits_are_cousins") is True
        for h in hits:
            assert h.get("applies_to_this_object") is False
    # global_dense_lowrank is a refuse hit on this parent; it must not flip
    # the *code-matrix* factorization row to ALREADY_FALSIFIED.
    fact = by_id["factorized_programs"]
    assert fact["status"] == mci.MEASURED_NEGATIVE
    if fact.get("index_refusals"):
        assert fact.get("index_hits_are_cousins") is True


def test_build_emits_sealed_receipt():
    out = mci.build(consult_index=True)
    assert out.parent == RECEIPTS
    assert out.name == "MLP_CODE_INFORMATION.json"
    doc = json.loads(out.read_text())
    assert doc["schema"] == mci.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    _assert_no_hardware_claims(doc)
    assert doc["accounting"]["code_bytes"] == mci.CODE_BYTES_TARGET
    assert doc["accounting"]["reconciled"] is True
    assert [c["id"] for c in doc["candidates"]] == list(mci.REQUIRED_CANDIDATE_IDS)
    assert doc["candidate_counts"]["n"] == len(mci.REQUIRED_CANDIDATE_IDS)
    assert doc["floor"]["stored_code_bytes"] == mci.CODE_BYTES_TARGET
    assert any(c["id"] == "entropy_coded_code_stream" for c in doc["open_byte_levers"])
    assert {c["id"] for c in doc["unmeasured_attractive"]} == {
        "shared_input_transforms",
        "function_replacement",
    }
    for dead in mci.AUXILIARY_DEAD_IDS:
        assert dead in doc["auxiliary_families_not_restated"]
        assert dead not in {c["id"] for c in doc["candidates"]}


def test_module_entrypoint_runs_and_emits_sealed_receipt():
    rc = mci.main(["--build"])
    assert rc == 0
    doc = json.loads((RECEIPTS / mci.RECEIPT).read_text())
    assert doc["schema"] == mci.SCHEMA
    assert doc["seal_sha256"]


def test_selftest_aliases_build():
    assert mci.selftest is mci.build or mci.selftest().name == mci.RECEIPT


def test_hardware_fields_stay_non_numeric_on_the_receipt():
    out = mci.build(consult_index=True)
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
        ["python3", "tools/future/mlp_code_information.py", "--accounting-only"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stderr[-800:]
    doc = json.loads(p.stdout)
    assert doc["code_bytes"] == mci.CODE_BYTES_TARGET
    assert doc["reconciled"] is True
