"""There was no writer for physical receipts, so every one of them hand-rolled its own.

`_common.write_receipt` is for STATIC sidecars and REFUSES hardware fields on purpose --
its own docstring says measurement receipts therefore "hand-rolled their own json.dumps
and inherited no bench block at all". That is the root cause of the contract debt: an
audit of 443 receipts found 77 carrying a measured rate and exactly ONE identifying all
six fields S006 14 requires (specimen, NR, runtime, context, path, concurrency).

Not a style problem. O003's two decode figures 25% apart could not be reconciled by
re-reading them, because NEITHER recorded its mlx_lm version, so the slower one's
compute path was unreconstructable.

So physical receipts get a writer that REFUSES an incomplete contract, and the debt
stops growing. Old receipts are not retrofitted -- a contract reconstructed after the
fact is a guess wearing a receipt's clothes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "future"))
import _common  # noqa: E402
from tps_contract import REQUIRED, audit  # noqa: E402


FULL = {
    "specimen": "Qwen3-30B-A3B@abc", "nr": "q4 g64 affine", "runtime": "mlx_lm 0.31.3",
    "context": 512, "path": "split_kv_b_proj greedy", "concurrency": 1,
}


def test_a_measurement_writer_exists_at_all():
    assert hasattr(_common, "write_measurement_receipt"), (
        "physical receipts still have no writer, so each one hand-rolls its contract")


def test_it_refuses_a_rate_with_no_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(_common, "RECEIPTS", tmp_path)
    with pytest.raises(_common.MeasurementContractError) as e:
        _common.write_measurement_receipt("X.json", {"decode_tps": 85.64}, contract={},
                                          recorded_by="test")
    for field in REQUIRED:
        assert field in str(e.value), f"the refusal does not name {field}: {e.value}"


def test_it_refuses_when_ONE_field_is_missing(tmp_path, monkeypatch):
    """Each field alone must be enough to refuse, or the check is decorative."""
    monkeypatch.setattr(_common, "RECEIPTS", tmp_path)
    for field in REQUIRED:
        partial = {k: v for k, v in FULL.items() if k != field}
        with pytest.raises(_common.MeasurementContractError) as e:
            _common.write_measurement_receipt("X.json", {"decode_tps": 1.0},
                                              contract=partial, recorded_by="test")
        assert field in str(e.value), f"missing {field} was not named: {e.value}"


def test_a_complete_contract_writes_and_the_result_is_COMPARABLE(tmp_path, monkeypatch):
    """The point of the writer: what it produces must pass the campaign's own audit."""
    monkeypatch.setattr(_common, "RECEIPTS", tmp_path)
    out = _common.write_measurement_receipt(
        "G_TEST_RATE.json", {"decode_tps": 85.64, "finding": "x"},
        contract=dict(FULL), recorded_by="test")
    doc = json.loads(out.read_text())
    assert doc["measurement_contract"] == FULL
    assert audit(out)["verdict"] == "COMPARABLE", audit(out)
    # and it must carry provenance -- WHEN, under what load, lane lock or not
    assert doc.get("measurement_provenance"), "no provenance stamped"
    assert "measured_at" in doc["measurement_provenance"]


def test_a_receipt_with_NO_rate_is_not_forced_through_this_door(tmp_path, monkeypatch):
    """Static sidecars keep their own writer. This must not become a tax on everything."""
    monkeypatch.setattr(_common, "RECEIPTS", tmp_path)
    out = _common.write_measurement_receipt(
        "G_TEST_NORATE.json", {"note": "no numbers here"}, contract=None, recorded_by="test")
    assert out.is_file()
