"""Widen-f4 628-graph A/B: token-id parity, not argmax; saving must reach the token or be named."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import deltanet_widen_ab as dwa


def _hist(kernel: str, count: int = 48) -> list[dict]:
    return [{"kernel": kernel, "count": count}]


def _run(
    *,
    ids: list[int],
    gpu_ns: int,
    kernel: str,
    fallbacks: int = 0,
    dispatches: int = 628,
) -> dict:
    return {
        "new_token_ids": ids,
        "fallbacks": fallbacks,
        "complete_token_gpu_ns_median": gpu_ns,
        "complete_token_gpu_ns": [gpu_ns] * 8,
        "complete_token_dispatches_last": dispatches,
        "theoretical_dispatches": dispatches,
        "launched_gated_delta_kernel": kernel,
        "kernel_histogram": _hist(kernel, 48 * 8),
        "fuse_ba_delta": False,
        "dn_state_kernel": "widen_f4" if "f4" in kernel else "baseline",
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
    doc = {
        "schema": "hawking.future.deltanet_widen_ab.raw.v1",
        "git_head": "deadbeef",
        "artifact_root": "/tmp/artifact",
        "reps": 7,
        "warmup": 1,
        "max_new_tokens": 8,
        "timing": "MTLCommandBuffer GPUStartTime/GPUEndTime",
        "absolute_ms_are_measured_under_load": True,
        "concurrent_load": {"loadavg": "{ 2.0 2.0 2.0 }"},
        "concurrent_load_start": {"loadavg": "{ 1.5 1.5 1.5 }"},
        "dense_w_materialized": 0,
        "production_fusions": {
            "mlp": "GateUpSwiglu",
            "fuse_gqa_qkv": True,
            "fuse_dn_inproj": True,
            "fuse_add_rmsnorm": True,
            "fuse_ba_delta": False,
        },
        "isolated_gated_delta": {
            "unfused": _iso(1_582_600, 48, "unfused"),
            "fused_ba": _iso(1_580_600, 48, "fused_ba"),
            "widen_f4": _iso(879_000, 48, "widen_f4"),
        },
        "isolated_organ": {
            "incumbent": _iso(8_160_000, 288, "organ_incumbent"),
            "widen_f4": _iso(7_458_000, 240, "organ_f4"),
        },
        "decode": {
            "interleaved": True,
            "incumbent": [
                _run(
                    ids=ids,
                    gpu_ns=28_700_000,
                    kernel=dwa.INCUMBENT_KERNEL,
                    dispatches=628,
                )
                for _ in range(7)
            ],
            "widen_f4": [
                _run(
                    ids=ids,
                    gpu_ns=27_998_000,
                    kernel=dwa.WIDEN_KERNEL,
                    dispatches=580,
                )
                for _ in range(7)
            ],
            "incumbent_complete_token_gpu_ns_median_reps": [28_700_000] * 7,
            "widen_f4_complete_token_gpu_ns_median_reps": [27_998_000] * 7,
            "incumbent_complete_token_gpu_ns_median": 28_700_000,
            "widen_f4_complete_token_gpu_ns_median": 27_998_000,
        },
    }
    doc.update(overrides)
    return doc


def test_refuses_to_report_parity_from_argmax_agreement_alone():
    """NEGATIVE CONTROL: the campaign scar. Argmax 1.0 is not parity."""
    with pytest.raises(dwa.ArgmaxIsNotParity, match="token-id"):
        dwa.report_token_parity(argmax_agreement=1.0)
    with pytest.raises(dwa.ArgmaxIsNotParity, match="not parity"):
        dwa.report_token_parity(
            incumbent_token_ids=None,
            candidate_token_ids=None,
            argmax_agreement=1.0,
        )
    # The exact DELTANET_MULTISTEP shape: argmax holds, tokens do not.
    p = dwa.report_token_parity(
        incumbent_token_ids=[10, 20, 30, 40],
        candidate_token_ids=[10, 20, 30, 99],
        fallbacks=0,
        argmax_agreement=1.0,
    )
    assert p["token_ids_identical"] is False
    assert p["parity"] is False
    assert p["argmax_is_not_parity"] is True
    assert p["argmax_agreement_ignored"] is True
    assert p["parity_basis"] == "token_id_equality"
    assert p["first_divergence"]["index"] == 3


def test_matching_ids_with_fallbacks_is_not_parity():
    p = dwa.report_token_parity(
        incumbent_token_ids=[1, 2, 3],
        candidate_token_ids=[1, 2, 3],
        fallbacks=2,
        argmax_agreement=1.0,
    )
    assert p["token_ids_identical"] is True
    assert p["parity"] is False
    assert p["fallbacks"] == 2
    assert p["argmax_is_not_parity"] is True


def test_token_id_equality_is_parity():
    p = dwa.report_token_parity(
        incumbent_token_ids=[7, 8, 9],
        candidate_token_ids=[7, 8, 9],
        fallbacks=0,
    )
    assert p["parity"] is True
    assert p["token_ids_identical"] is True
    assert p["first_divergence"] is None
    assert p["argmax_is_not_parity"] is True


def test_locate_saving_names_a_token_that_did_not_keep_the_organ_cut():
    s = dwa.locate_saving(
        organ_unfused_ms=1.5826,
        organ_fused_ba_ms=1.5806,
        organ_f4_ms=0.879,
        token_incumbent_ms=28.7,
        token_f4_ms=28.69,
    )
    assert s["reached_the_token"] is False
    assert s["clears_materiality"] is False
    assert "kept only" in s["where"] or "did not appear" in s["where"]
    assert "displaced" in s["where"].lower()
    assert s["isolated_fair_cut_ms"] == pytest.approx(0.7016, abs=1e-3)
    assert s["complete_token_saving_ms"] == pytest.approx(0.01, abs=1e-3)


def test_locate_saving_reports_when_the_token_keeps_the_cut():
    s = dwa.locate_saving(
        organ_unfused_ms=1.5826,
        organ_fused_ba_ms=1.5806,
        organ_f4_ms=0.879,
        token_incumbent_ms=28.7,
        token_f4_ms=27.998,
    )
    assert s["reached_the_token"] is True
    assert s["complete_token_saving_ms"] == pytest.approx(0.702, abs=1e-3)
    assert "reached the complete token" in s["where"]
    assert "extra" in s["where"] or "fair cut" in s["where"]


def test_measurement_from_happy_raw_is_token_identical_and_reached():
    measured = dwa.measurement_from_raw(_raw())
    assert measured["parity"]["parity"] is True
    assert measured["parity"]["fallbacks"] == 0
    assert measured["parity"]["argmax_is_not_parity"] is True
    assert measured["saving"]["reached_the_token"] is True
    doc = dwa.build(measured)
    assert doc["schema"] == dwa.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["took_gpu_lease"] is True
    assert doc["absolute_ms_are_measured_under_load"] is True
    assert "loadavg" in (doc["concurrent_load"] or {})
    assert doc["complete_token"]["incumbent_ms"] == pytest.approx(28.7, abs=1e-6)
    assert doc["complete_token"]["widen_f4_ms"] == pytest.approx(27.998, abs=1e-6)
    assert doc["parity"]["parity_basis"] == "token_id_equality"
    assert any(f["id"] == "ORGAN_SAVING_VS_COMPLETE_TOKEN" for f in doc["findings"])


def test_refuses_when_candidate_did_not_launch_f4():
    raw = _raw()
    for run in raw["decode"]["widen_f4"]:
        run["launched_gated_delta_kernel"] = dwa.INCUMBENT_KERNEL
        run["kernel_histogram"] = _hist(dwa.INCUMBENT_KERNEL, 48 * 8)
    with pytest.raises(dwa.ProductionDidNotLaunch, match="f4"):
        dwa.measurement_from_raw(raw)


def test_refuses_empty_complete_token_ns():
    raw = _raw()
    raw["decode"]["incumbent_complete_token_gpu_ns_median"] = 0
    for run in raw["decode"]["incumbent"]:
        run["complete_token_gpu_ns_median"] = 0
    with pytest.raises(dwa.EmptyGpuSample):
        dwa.measurement_from_raw(raw)


def test_mismatched_ids_across_arms_is_not_parity():
    raw = _raw()
    for run in raw["decode"]["widen_f4"]:
        run["new_token_ids"] = [11, 22, 33, 44, 55, 66, 77, 1]
    measured = dwa.measurement_from_raw(raw)
    assert measured["parity"]["parity"] is False
    assert measured["parity"]["token_ids_identical"] is False
    assert measured["parity"]["first_divergence"]["index"] == 7


def test_record_refuses_none(tmp_path: Path):
    with pytest.raises(dwa.WidenAbRefuse, match="without a measurement"):
        dwa.record(None, path=tmp_path / "x.json")
    dest = tmp_path / "DELTANET_WIDEN_AB.json"
    path = dwa.record(dwa.measurement_from_raw(_raw()), path=dest)
    assert path == dest
    doc = json.loads(dest.read_text())
    assert doc["schema"] == dwa.SCHEMA
    assert doc["parity"]["argmax_is_not_parity"] is True
    assert "complete_token" in doc
    assert doc["concurrent_load"]["loadavg"]


def test_committed_receipt_if_present_has_both_arms_and_parity_basis():
    if not dwa.RECEIPT.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(dwa.RECEIPT.read_text())
    assert doc["schema"] == dwa.SCHEMA
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"
    assert doc["took_gpu_lease"] is True
    assert doc["absolute_ms_are_measured_under_load"] is True
    ct = doc["complete_token"]
    assert ct["incumbent_ms"] and ct["widen_f4_ms"]
    assert doc["parity"]["parity_basis"] == "token_id_equality"
    assert doc["parity"]["argmax_is_not_parity"] is True
    assert "loadavg" in (doc["concurrent_load"] or {})
    saving = doc["saving"]
    assert "where" in saving
    assert "reached_the_token" in saving
    if not saving["reached_the_token"]:
        assert "complete token" in saving["where"].lower() or "did not" in saving["where"].lower()
