"""Focused regression for P15_AUTOTUNING_FOCUSED_CONTRACT_V4.

Exercises the newly added ``frontier_scheduler._read_json_receipt`` helper:
a receipt that is absent, empty, malformed, or not a JSON object must read as
None (probe-level UNKNOWN) instead of raising, so one broken receipt can never
sink the other frontiers in the same snapshot.

Runnable two ways::

    python3 -m pytest hawking/test_auto_p15_autotuning_focused_contract_v4_regression.py -q
    python3 hawking/test_auto_p15_autotuning_focused_contract_v4_regression.py
"""
from __future__ import annotations

import json

from hawking import frontier_scheduler as fs


def test_missing_receipt_reads_as_none(tmp_path):
    assert fs._read_json_receipt(tmp_path / "absent.json") is None


def test_empty_receipt_reads_as_none(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("   \n", encoding="utf-8")
    assert fs._read_json_receipt(p) is None


def test_malformed_receipt_reads_as_none(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("{not json", encoding="utf-8")
    assert fs._read_json_receipt(p) is None


def test_non_object_receipt_reads_as_none(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert fs._read_json_receipt(p) is None


def test_valid_object_receipt_is_returned(tmp_path):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps({"state": "WAITING"}), encoding="utf-8")
    got = fs._read_json_receipt(p)
    assert got == {"state": "WAITING"}


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_missing_receipt_reads_as_none(Path(d))
        test_empty_receipt_reads_as_none(Path(d))
        test_malformed_receipt_reads_as_none(Path(d))
        test_non_object_receipt_reads_as_none(Path(d))
        test_valid_object_receipt_is_returned(Path(d))
    print("P15_AUTOTUNING_FOCUSED_CONTRACT_V4 regression: OK")