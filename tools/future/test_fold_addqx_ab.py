"""fold_addqx complete-token A/B: token-id bytes, not argmax; probe is not identity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import fold_addqx_ab as fab


def _hist(kernel: str, count: int = 64) -> list[dict]:
    return [{"kernel": kernel, "count": count}]


def _byte_ids(ids: list[int]) -> dict:
    return fab.byte_compare_u32(ids, ids)


def _layer0(*, identical: bool = True) -> dict:
    n = 17408 * 4
    mismatch = 0 if identical else 4
    row = {
        "n_bytes_compared": n,
        "n_mismatch_bytes": mismatch,
        "bit_identical": identical,
        "first_mismatch_index": None if identical else 0,
        "compared_against": "layer-0 named matvec output buffers",
    }
    return {
        "bit_identical": identical,
        "compared_against": "production geo_tpr64 named-matvec output buffers on the live session, same x",
        "gate": dict(row),
        "up": dict(row),
        "down": {
            **row,
            "n_bytes_compared": 5120 * 4,
        },
        "n_bytes_compared": n * 2 + 5120 * 4,
        "n_mismatch_bytes": 0 if identical else 12,
    }


def _run(
    *,
    ids: list[int],
    gpu_ns: int,
    swiglu: str,
    down: str,
    fallbacks: int = 0,
    dispatches: int = 580,
) -> dict:
    return {
        "new_token_ids": ids,
        "fallbacks": fallbacks,
        "complete_token_gpu_ns_median": gpu_ns,
        "complete_token_gpu_ns": [gpu_ns] * 8,
        "complete_token_dispatches_last": dispatches,
        "theoretical_dispatches": dispatches,
        "affine2_geo": "fold_addqx" if "fold_addqx" in swiglu else "tpr64",
        "dn_state_kernel": "widen_f4",
        "kernel_histogram": [
            {"kernel": swiglu, "count": 64 * 8},
            {"kernel": down, "count": 64 * 8},
            {"kernel": fab.DN_F4, "count": 48 * 8},
        ],
    }


def _iso(ns: int, dispatches: int, name: str) -> dict:
    return {
        "name": name,
        "gpu_ns_median": ns,
        "gpu_ns_reps": [ns] * 7,
        "dispatches": dispatches,
        "n_reps": 7,
    }


def _raw(**overrides) -> dict:
    ids = [11, 22, 33, 44, 55, 66, 77, 88]
    inc_ns = 26_382_083
    fold_ns = 24_637_083  # −1.745 ms vs incumbent, the projection
    doc = {
        "schema": "hawking.future.fold_addqx_ab.raw.v1",
        "git_head": "deadbeef",
        "artifact_root": "/tmp/artifact",
        "reps": 7,
        "warmup": 1,
        "max_new_tokens": 8,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "absolute_ms_are_measured_under_load": True,
        "incumbent_is_post_widen_f4_baseline": True,
        "concurrent_load": {"loadavg": "{ 2.0 2.0 2.0 }"},
        "concurrent_load_start": {"loadavg": "{ 1.5 1.5 1.5 }"},
        "dense_w_materialized": 0,
        "production_fusions": {
            "mlp": "GateUpSwiglu",
            "fuse_gqa_qkv": True,
            "fuse_dn_inproj": True,
            "fuse_add_rmsnorm": True,
            "fuse_ba_delta": False,
            "dn_state_kernel": "widen_f4",
            "affine2_geo_incumbent": "tpr64",
            "affine2_geo_candidate": "fold_addqx",
        },
        "layer0_byte_compare": _layer0(identical=True),
        "isolated_mlp_full": {
            "incumbent": _iso(15_541_000, 128, "mlp_full_incumbent"),
            "fold_addqx": _iso(13_796_000, 128, "mlp_full_fold_addqx"),
        },
        "isolated_mlp_matvecs": {
            "incumbent": _iso(14_000_000, 192, "matvec_incumbent"),
            "fold_addqx": _iso(12_255_000, 192, "matvec_fold"),
        },
        "decode": {
            "interleaved": True,
            "incumbent": [
                _run(
                    ids=ids,
                    gpu_ns=inc_ns,
                    swiglu=fab.INCUMBENT_SWIGLU,
                    down=fab.INCUMBENT_DOWN,
                )
                for _ in range(7)
            ],
            "fold_addqx": [
                _run(
                    ids=ids,
                    gpu_ns=fold_ns,
                    swiglu=fab.FOLD_SWIGLU,
                    down=fab.FOLD_DOWN,
                )
                for _ in range(7)
            ],
            "incumbent_complete_token_gpu_ns_median_reps": [inc_ns] * 7,
            "fold_addqx_complete_token_gpu_ns_median_reps": [fold_ns] * 7,
            "incumbent_complete_token_gpu_ns_median": inc_ns,
            "fold_addqx_complete_token_gpu_ns_median": fold_ns,
            "token_id_byte_compare": [_byte_ids(ids) for _ in range(7)],
        },
    }
    doc.update(overrides)
    return doc


def test_refuses_to_report_parity_from_argmax_agreement_alone():
    with pytest.raises(fab.ArgmaxIsNotParity, match="token-id"):
        fab.report_token_parity(argmax_agreement=1.0)
    with pytest.raises(fab.ArgmaxIsNotParity, match="not parity"):
        fab.report_token_parity(
            incumbent_token_ids=None,
            candidate_token_ids=None,
            argmax_agreement=1.0,
        )


def test_matching_ids_with_fallbacks_is_not_parity():
    p = fab.report_token_parity(
        incumbent_token_ids=[1, 2, 3],
        candidate_token_ids=[1, 2, 3],
        fallbacks=2,
        layer0_byte_compare=_layer0(identical=True),
    )
    assert p["token_ids_identical"] is True
    assert p["parity"] is False
    assert p["fallbacks"] == 2
    assert p["argmax_is_not_parity"] is True


def test_token_id_byte_compare_is_required_for_bit_identity():
    with pytest.raises(fab.NoByteComparison):
        fab.require_byte_identity(None, label="token_ids")
    with pytest.raises(fab.NoByteComparison, match="n_bytes_compared"):
        fab.require_byte_identity(
            {"n_bytes_compared": 0, "n_mismatch_bytes": 0, "bit_identical": True},
            label="token_ids",
        )
    ok = fab.require_byte_identity(
        {"n_bytes_compared": 16, "n_mismatch_bytes": 0, "bit_identical": True},
        label="token_ids",
    )
    assert ok["bit_identical"] is True


def test_raw_without_layer0_byte_compare_is_refused():
    raw = _raw()
    del raw["layer0_byte_compare"]
    with pytest.raises(fab.NoByteComparison, match="probe"):
        fab.measurement_from_raw(raw)


def test_measurement_from_happy_raw_is_token_identical_and_exact():
    measured = fab.measurement_from_raw(_raw())
    assert measured["parity"]["parity"] is True
    assert measured["parity"]["fallbacks"] == 0
    assert measured["parity"]["arithmetic_exact"] is True
    assert measured["parity"]["argmax_is_not_parity"] is True
    assert measured["parity"]["cited_probe_is_not_the_identity_proof"] is True
    assert measured["parity"]["parity_basis"] == "token_id_equality_and_byte_identity"
    assert measured["saving"]["reached_the_token"] is True
    assert measured["complete_token"]["incumbent_is_post_widen_f4_baseline"] is True
    doc = fab.build(measured)
    assert doc["schema"] == fab.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["took_gpu_lease"] is True
    assert doc["incumbent_is_post_widen_f4_baseline"] is True
    assert doc["tps_qualification"]["any_tps_labelled_qualified"] is False
    assert doc["tps_qualification"]["protected_window_required"] is True
    assert "loadavg" in (doc["concurrent_load"] or {})
    assert any(f["id"] == "INCUMBENT_IS_POST_WIDEN_F4_BASELINE" for f in doc["findings"])


def test_refuses_when_candidate_did_not_launch_fold_addqx():
    raw = _raw()
    for run in raw["decode"]["fold_addqx"]:
        run["kernel_histogram"] = _hist(fab.INCUMBENT_SWIGLU, 64 * 8)
    with pytest.raises(fab.ProductionDidNotLaunch, match="fold_addqx"):
        fab.measurement_from_raw(raw)


def test_refuses_when_candidate_still_launches_production_swiglu():
    raw = _raw()
    for run in raw["decode"]["fold_addqx"]:
        run["kernel_histogram"] = [
            {"kernel": fab.FOLD_SWIGLU, "count": 1},
            {"kernel": fab.INCUMBENT_SWIGLU, "count": 64 * 8},
        ]
    with pytest.raises(fab.ProductionDidNotLaunch, match="still dispatched"):
        fab.measurement_from_raw(raw)


def test_refuses_empty_complete_token_ns():
    raw = _raw()
    raw["decode"]["incumbent_complete_token_gpu_ns_median"] = 0
    for run in raw["decode"]["incumbent"]:
        run["complete_token_gpu_ns_median"] = 0
    with pytest.raises(fab.EmptyGpuSample):
        fab.measurement_from_raw(raw)


def test_mismatched_ids_is_not_parity_even_if_faster():
    raw = _raw()
    for run in raw["decode"]["fold_addqx"]:
        run["new_token_ids"] = [11, 22, 33, 44, 55, 66, 77, 1]
    raw["decode"]["token_id_byte_compare"] = [
        fab.byte_compare_u32([11, 22, 33, 44, 55, 66, 77, 88], [11, 22, 33, 44, 55, 66, 77, 1])
    ]
    measured = fab.measurement_from_raw(raw)
    assert measured["parity"]["parity"] is False
    assert measured["parity"]["token_ids_identical"] is False
    assert measured["saving"]["faster_not_exact"] is True
    assert measured["saving"]["class"] == "approx_candidate"
    assert "not blended" in measured["saving"]["where"].lower() or "FAST" in measured["saving"]["where"]


def test_projection_miss_is_the_result_not_a_failure_to_record():
    raw = _raw()
    # Token saves 0.2 ms, isolated still 1.745.
    raw["decode"]["fold_addqx_complete_token_gpu_ns_median"] = 26_182_083
    for run in raw["decode"]["fold_addqx"]:
        run["complete_token_gpu_ns_median"] = 26_182_083
    measured = fab.measurement_from_raw(raw)
    s = measured["saving"]
    assert s["reproduced_1p745_projection"] is False
    assert "1.745" in s["where"] or "projection" in s["where"].lower()
    doc = fab.build(measured)
    assert doc["saving"]["reproduced_1p745_projection"] is False


def test_record_refuses_none(tmp_path: Path):
    with pytest.raises(fab.FoldAbRefuse, match="without a measurement"):
        fab.record(None, path=tmp_path / "x.json")
    dest = tmp_path / "FOLD_ADDQX_AB.json"
    path = fab.record(fab.measurement_from_raw(_raw()), path=dest)
    assert path == dest
    doc = json.loads(dest.read_text())
    assert doc["schema"] == fab.SCHEMA
    assert doc["parity"]["argmax_is_not_parity"] is True
    assert doc["incumbent_is_post_widen_f4_baseline"] is True
    assert doc["tps_qualification"]["any_tps_labelled_qualified"] is False
    assert "complete_token" in doc
    assert doc["concurrent_load"]["loadavg"]


def test_committed_receipt_if_present_has_both_arms_and_no_qualified_tps():
    if not fab.RECEIPT.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(fab.RECEIPT.read_text())
    assert doc["schema"] == fab.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["took_gpu_lease"] is True
    assert doc["absolute_ms_are_measured_under_load"] is True
    assert doc["incumbent_is_post_widen_f4_baseline"] is True
    ct = doc["complete_token"]
    assert ct["incumbent_ms"] and ct["fold_addqx_ms"]
    assert doc["parity"]["parity_basis"] == "token_id_equality_and_byte_identity"
    assert doc["parity"]["argmax_is_not_parity"] is True
    assert doc["parity"]["cited_probe_is_not_the_identity_proof"] is True
    assert doc["tps_qualification"]["any_tps_labelled_qualified"] is False
    assert "loadavg" in (doc["concurrent_load"] or {})
    saving = doc["saving"]
    assert "where" in saving
    assert "reached_the_token" in saving
    if saving.get("faster_not_exact"):
        assert saving["class"] == "approx_candidate"
