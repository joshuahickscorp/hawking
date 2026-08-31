"""Addressing attribution: every rung sourced, 703 has no input-vector load."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import addressing_attribution as aa


def test_703_is_marked_measured_without_input_vector_load():
    doc = aa.build()
    assert doc["703_without_input_vector_load"] is True
    rung = next(r for r in doc["rungs"] if r["id"] == "catalog_addressing_single_gemv_addr")
    assert rung["loads_activation"] is False
    assert rung["comparable_to_production_decode"] is False
    prov = rung["provenance"].lower()
    assert "no input-vector load" in prov or "never loads the activation" in prov
    assert "nibble" in prov
    assert "not comparable" in prov
    assert abs(rung["gb_s_full"] - 703.6072736347875) < 1e-6
    assert doc["703_statistic"].startswith("max")


def test_every_rung_has_source_receipt_and_provenance():
    doc = aa.build()
    ids = {r["id"] for r in doc["rungs"]}
    assert "catalog_addressing_single_gemv_addr" in ids
    assert "production_catalog_401_gemvs" in ids
    assert "mlp_alu_roofline_arm_a" in ids
    assert "fold_addqx" in ids
    assert "production_affine2_q2" in ids
    assert "production_effective" in ids
    assert "datasheet_peak" in ids
    assert "deltanet_arm_a_residency" in ids
    for r in doc["rungs"]:
        assert r["source_receipt"], r["id"]
        assert r["json_path"], r["id"]
        assert r["provenance"], r["id"]
        assert r["statistic"], r["id"]
        assert isinstance(r["gb_s_full"], float)


def test_530_is_first_target_and_943_is_not_dram():
    doc = aa.build()
    assert doc["first_target"]["id"] == "production_catalog_401_gemvs"
    assert abs(doc["first_target"]["gb_s"] - 530.6544688491846) < 1e-6
    arm = next(r for r in doc["rungs"] if r["id"] == "deltanet_arm_a_residency")
    assert arm["as_dram_roof"] is False
    assert "residency" in arm["provenance"].lower()
    assert arm["gb_s"] == pytest.approx(943.2, abs=0.05)
    peak = next(r for r in doc["rungs"] if r["id"] == "datasheet_peak")
    assert peak["gb_s"] == pytest.approx(819.0)
    assert peak["as_dram_roof"] is False


def test_refuted_mechanisms_are_not_reproposed_as_removable():
    doc = aa.build()
    refuted_ids = {m["id"] for m in doc["refuted"]}
    assert "region_granularity" in refuted_ids
    assert "catalog_addressing_as_main_mechanism" in refuted_ids
    assert "raw_dispatch_count" in refuted_ids
    assert "decode_fusion" in refuted_ids
    assert "stream_count_at_fixed_bytes_per_thread" in refuted_ids
    assert "dependency_chains" in refuted_ids
    assert "register_pressure" in refuted_ids
    assert "occupancy" in refuted_ids
    for m in doc["refuted"]:
        assert m["kind"] == "refuted"
        assert m["status"] == "REFUTED"
        assert m["source_receipt"]
    removable_ids = {m["id"] for m in doc["removable"]}
    assert removable_ids.isdisjoint(refuted_ids)
    occ = next(m for m in doc["refuted"] if m["id"] == "occupancy")
    assert "worse" in occ["refutation_or_demonstration"].lower()
    dep = next(m for m in doc["refuted"] if m["id"] == "dependency_chains")
    assert dep["span"] == pytest.approx(1.062, abs=1e-3)
    stream = next(m for m in doc["refuted"] if m["id"] == "stream_count_at_fixed_bytes_per_thread")
    assert "hurt" in stream["refutation_or_demonstration"].upper() or "HURTS" in stream["refutation_or_demonstration"]


def test_fold_addqx_is_the_demonstrated_removable():
    doc = aa.build()
    assert doc["removable_count"] == 1
    m = doc["removable"][0]
    assert m["id"] == "decode_arithmetic_fold_addqx"
    assert m["status"] == "REMOVABLE_DEMONSTRATED"
    assert m["span"] == pytest.approx(1.1265, abs=1e-3)
    fold = next(r for r in doc["rungs"] if r["id"] == "fold_addqx")
    assert fold["gb_s"] == pytest.approx(370.9, abs=0.05)
    assert "bit-identical" in fold["provenance"].lower() or "BIT-IDENTICAL" in fold["what"]


def test_no_tps_is_labelled_qualified():
    doc = aa.build()
    assert doc["tps_qualification"]["any_tps_labelled_qualified"] is False
    assert doc["tps_qualification"]["protected_window_required"] is True
    for row in doc["dirty_token_ms"]:
        assert row["qualified"] is False
        assert row["qualification"] == "NOT_QUALIFIED"
    blob = json.dumps(doc)
    assert '"qualification": "QUALIFIED"' not in blob
    assert '"qualified": true' not in blob.lower()
    assert doc["tps_qualification"]["any_tps_labelled_qualified"] is False


def test_refuses_injected_missing_rung():
    with pytest.raises(aa.AttributionRefuse):
        aa.build(injected={"receipts/future/CATALOG_ADDRESSING.json": {}})


def test_arm_a_loads_activation_and_is_mlp_ceiling():
    doc = aa.build()
    arm = next(r for r in doc["rungs"] if r["id"] == "mlp_alu_roofline_arm_a")
    assert arm["loads_activation"] is True
    assert arm["gb_s"] == pytest.approx(497.4, abs=0.05)
    prod = next(r for r in doc["rungs"] if r["id"] == "production_affine2_q2")
    assert prod["gb_s"] == pytest.approx(329.6, abs=0.05)
    eff = next(r for r in doc["rungs"] if r["id"] == "production_effective")
    assert abs(eff["gb_s_full"] - 337.26568478304) < 1e-6


def test_record_writes_sealed_receipt(tmp_path: Path):
    dest = tmp_path / "ADDRESSING_ATTRIBUTION.json"
    path = aa.record(path=dest)
    assert path == dest
    doc = json.loads(dest.read_text())
    assert doc["schema"] == aa.SCHEMA
    assert "seal_sha256" in doc
    assert doc["703_without_input_vector_load"] is True
    assert doc["evidence_class"] == "SELF_MEASURED_DIRTY"


def test_committed_receipt_if_present():
    from tools.future._common import RECEIPTS

    path = RECEIPTS / aa.RECEIPT
    if not path.is_file():
        pytest.skip("receipt not recorded yet")
    doc = json.loads(path.read_text())
    assert doc["schema"] == aa.SCHEMA
    assert doc["703_without_input_vector_load"] is True
    assert doc["tps_qualification"]["any_tps_labelled_qualified"] is False
    assert doc["first_target"]["id"] == "production_catalog_401_gemvs"
    assert {r["id"] for r in doc["rungs"]} >= {
        "catalog_addressing_single_gemv_addr",
        "production_catalog_401_gemvs",
        "mlp_alu_roofline_arm_a",
        "fold_addqx",
        "production_effective",
    }
