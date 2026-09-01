"""G129: a receipt path belongs to one producer.

tps_budget.py and causal_budget_71.py both wrote RESIDENT_71TPS_CAUSAL_BUDGET.json
with different schemas. The later writer won. Four rows of the roof-anchor audit
stopped resolving, and the audit honestly reported "field is not a resolvable
path in this receipt" about a field that HAD been resolvable the day it was
written. Nothing raised, because an overwrite is a write and writes succeed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402


def _write(tmp_path, monkeypatch, name, doc, recorded_by):
    monkeypatch.setattr(c, "RECEIPTS", tmp_path)
    return c.write_receipt(name, dict(doc), recorded_by)


# Neutral fields: write_receipt refuses hardware numbers in a sidecar, and that
# guard fires before this one. The collision is about SHAPE, not about values.
A = {"schema": "hawking.future.alpha.v1", "recorded_by": "tools/future/alpha.py",
     "ladder": [{"rung": "71 TPS", "verdict": "NOT_REACHABLE"}]}
B = {"schema": "hawking.future.beta.v1", "recorded_by": "tools/future/beta.py",
     "milestone_ladder": {"roof_name": "lane_established_clean_gemv"}}


def test_a_foreign_producer_with_a_different_schema_is_refused(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, "R.json", A, "tools/future/alpha.py")
    with pytest.raises(c.ReceiptPathCollision, match="belongs to one producer"):
        _write(tmp_path, monkeypatch, "R.json", B, "tools/future/beta.py")


def test_the_earlier_receipt_survives_the_refusal(tmp_path, monkeypatch):
    """A guard that raises AFTER truncating the file would be worse than none."""
    p = _write(tmp_path, monkeypatch, "R.json", A, "tools/future/alpha.py")
    with pytest.raises(c.ReceiptPathCollision):
        _write(tmp_path, monkeypatch, "R.json", B, "tools/future/beta.py")
    kept = json.loads(p.read_text())
    assert kept["schema"] == "hawking.future.alpha.v1"
    assert kept["ladder"][0]["verdict"] == "NOT_REACHABLE"


def test_the_same_producer_may_regenerate(tmp_path, monkeypatch):
    """Regeneration is the normal case and must not be blocked."""
    _write(tmp_path, monkeypatch, "R.json", A, "tools/future/alpha.py")
    moved = dict(A, ladder=[{"rung": "71 TPS", "verdict": "REACHED"}])
    p = _write(tmp_path, monkeypatch, "R.json", moved, "tools/future/alpha.py")
    assert json.loads(p.read_text())["ladder"][0]["verdict"] == "REACHED"


def test_a_different_producer_with_the_SAME_schema_is_allowed(tmp_path, monkeypatch):
    """Schema is the ownership signal, not the filename of the writer. Two tools
    that agree on the shape are not clobbering each other."""
    _write(tmp_path, monkeypatch, "R.json", A, "tools/future/alpha.py")
    same = dict(A, recorded_by="tools/future/gamma.py")
    p = _write(tmp_path, monkeypatch, "R.json", same, "tools/future/gamma.py")
    assert p.is_file()


def test_a_first_write_is_never_blocked(tmp_path, monkeypatch):
    p = _write(tmp_path, monkeypatch, "NEW.json", B, "tools/future/beta.py")
    assert p.is_file()


def test_an_unreadable_prior_is_not_treated_as_ownership(tmp_path, monkeypatch):
    (tmp_path / "R.json").write_text("{ this is not json")
    p = _write(tmp_path, monkeypatch, "R.json", B, "tools/future/beta.py")
    assert json.loads(p.read_text())["schema"] == "hawking.future.beta.v1"


def test_a_prior_without_a_recorded_by_does_not_claim_the_path(tmp_path, monkeypatch):
    (tmp_path / "R.json").write_text(json.dumps({"schema": "x.v1"}))
    p = _write(tmp_path, monkeypatch, "R.json", B, "tools/future/beta.py")
    assert p.is_file()


def test_the_two_real_producers_no_longer_share_a_path():
    import causal_budget_71 as cb
    import tps_budget as tb
    assert Path(cb.RECEIPT).name != tb.RECEIPT, (
        "tps_budget must not write causal_budget_71's receipt"
    )
    assert Path(cb.RECEIPT).name == "RESIDENT_71TPS_CAUSAL_BUDGET.json"
