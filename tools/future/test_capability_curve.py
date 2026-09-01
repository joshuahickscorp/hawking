"""The cliff is an interval found by search, not a typed 10/20/30/40 list.

The two ways this module could lie are inventing a cliff on a flat curve
and choosing which organ to sweep. Both are pinned here. Every measurement
in these tests is an injected function - no GPU, no model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.future import capability_curve as cc
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


STEP = 0.37


def _target(**over):
    d = dict(
        component="test.component",
        layer=7,
        axis="output_rows",
        perturbation_type="zero_fraction",
        lo=0.0,
        hi=1.0,
        resolution=0.02,
        budget=16,
        n_coarse=5,
    )
    d.update(over)
    return d


def _step(spec, *, at=STEP, pre=1.0, post=0.0):
    return pre if float(spec["level"]) < at else post


def _flat(spec):
    return 0.5


def test_a_step_function_is_bracketed_not_point_estimated():
    r = cc.sweep(**_target(), measure=_step)
    assert r["cliff_found"] is True
    box = r["bracket"]
    assert box is not None
    assert set(box) >= {"lo", "hi"}
    assert box["lo"] < box["hi"], "a cliff between samples is an interval"
    assert box["lo"] <= STEP <= box["hi"]
    assert box["hi"] - box["lo"] <= r["resolution"] + 1e-12
    assert r["resolution_met"] is True
    assert not isinstance(r["bracket"], (int, float))


def test_an_inverted_step_is_still_bracketed():
    r = cc.sweep(**_target(), measure=lambda s: _step(s, pre=0.0, post=1.0))
    assert r["cliff_found"] is True
    assert r["bracket"]["lo"] <= STEP <= r["bracket"]["hi"]


def test_a_flat_function_reports_no_cliff_rather_than_inventing_one():
    r = cc.sweep(**_target(), measure=_flat)
    assert r["cliff_found"] is False
    assert r["bracket"] is None
    assert r["message"] == "no cliff found in [0, 1]"
    assert "flat" in r["why"]


def test_a_linear_ramp_is_not_reported_as_a_cliff():
    """A change spread evenly across the range is not a concentrated cliff."""
    r = cc.sweep(**_target(), measure=lambda s: float(s["level"]))
    assert r["cliff_found"] is False
    assert r["bracket"] is None
    assert "no cliff found" in r["message"]
    assert "spread" in r["why"]


def test_the_point_budget_is_honoured():
    calls: list[float] = []

    def counted(spec):
        calls.append(float(spec["level"]))
        return _step(spec)

    r = cc.sweep(
        **_target(budget=7, resolution=0.001),
        measure=counted,
    )
    assert len(calls) <= 7
    assert len(calls) == 7, "a tight resolution should spend the whole budget"
    assert r["n_measured"] == 7
    assert r["budget_honoured"] is True
    assert r["cliff_found"] is True
    assert r["resolution_met"] is False
    assert r["bracket"]["lo"] <= STEP <= r["bracket"]["hi"]


def test_cached_points_are_not_remeasured():
    calls: list[float] = []
    cache: dict = {}

    def counted(spec):
        calls.append(float(spec["level"]))
        return _step(spec)

    first = cc.sweep(**_target(), measure=counted, cache=cache)
    n = len(calls)
    assert n == first["n_measured"]
    assert n > 0
    second = cc.sweep(**_target(), measure=counted, cache=cache)
    assert len(calls) == n, "a re-run must not repay for the same measurement"
    assert second["n_measured"] == 0
    assert second["n_cache_hits"] >= n
    assert second["bracket"] == first["bracket"]


def test_cache_is_keyed_by_identity_not_only_level():
    calls: list[tuple] = []

    def counted(spec):
        calls.append((spec["component"], spec["level"]))
        return _step(spec)

    cache: dict = {}
    cc.sweep(**_target(component="A", budget=5, n_coarse=5),
             measure=counted, cache=cache)
    cc.sweep(**_target(component="B", budget=5, n_coarse=5),
             measure=counted, cache=cache)
    a_levels = [lv for comp, lv in calls if comp == "A"]
    b_levels = [lv for comp, lv in calls if comp == "B"]
    assert len(a_levels) == 5
    assert len(b_levels) == 5


def test_each_point_records_the_level_it_measured_at():
    r = cc.sweep(**_target(), measure=_step)
    assert r["points"]
    for p in r["points"]:
        assert "level" in p
        assert r["search_range"][0] <= p["level"] <= r["search_range"][1]
        assert "value" in p
        assert p["component"] == "test.component"
        assert p["layer"] == 7
        assert p["axis"] == "output_rows"
        assert p["perturbation_type"] == "zero_fraction"
        assert p["source"] in ("coarse", "refine")


def test_measured_at_level_is_recorded_when_the_measure_provides_it():
    def as_dict(spec):
        return {
            "value": _step(spec),
            "measured_at_level": "LOCAL_FUNCTIONAL_FIDELITY",
        }

    r = cc.sweep(**_target(), measure=as_dict)
    assert r["points"]
    assert all(p["measured_at_level"] == "LOCAL_FUNCTIONAL_FIDELITY"
               for p in r["points"])


def test_coarse_levels_are_derived_not_typed_percentages():
    r = cc.sweep(**_target(n_coarse=5, budget=5), measure=_step)
    coarse = [p["level"] for p in r["points"] if p["source"] == "coarse"]
    assert coarse == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    # The human-typed grid this module exists to replace.
    for typed in (0.1, 0.2, 0.3, 0.4):
        assert typed not in coarse


def test_the_module_does_not_choose_a_component():
    src = Path(cc.__file__).read_text()
    for needle in (
        "mlp.down", "mlp.gate", "gate_proj", "down_proj", "up_proj",
        "q_proj", "o_proj",
    ):
        assert needle not in src, needle
    assert cc.SYNTHETIC_IDENTITY["component"] == "SYNTHETIC"
    r = cc.sweep(**_target(component="caller.named"), measure=_flat)
    assert r["component"] == "caller.named"


def test_missing_component_refuses():
    with pytest.raises(cc.CurveRefused, match="component is missing"):
        cc.sweep(**_target(component=None), measure=_step)


def test_missing_layer_refuses():
    with pytest.raises(cc.CurveRefused, match="layer is missing"):
        cc.sweep(**_target(layer=None), measure=_step)


def test_missing_axis_refuses():
    with pytest.raises(cc.CurveRefused, match="axis is missing"):
        cc.sweep(**_target(axis=""), measure=_step)


def test_missing_perturbation_type_refuses():
    with pytest.raises(cc.CurveRefused, match="perturbation_type is missing"):
        cc.sweep(**_target(perturbation_type=None), measure=_step)


def test_missing_measure_refuses_rather_than_calling_the_gpu():
    with pytest.raises(cc.CurveRefused, match="will not default to a GPU call"):
        cc.sweep(**_target())


def test_missing_resolution_refuses():
    with pytest.raises(cc.CurveRefused, match="resolution is missing"):
        cc.sweep(**_target(resolution=None), measure=_step)


def test_missing_budget_refuses():
    with pytest.raises(cc.CurveRefused, match="point budget is missing"):
        cc.sweep(**_target(budget=None), measure=_step)


def test_budget_below_coarse_scan_refuses():
    with pytest.raises(cc.CurveRefused, match="silently shrink the scan"):
        cc.sweep(**_target(budget=4, n_coarse=5), measure=_step)


def test_empty_range_refuses():
    with pytest.raises(cc.CurveRefused, match="empty"):
        cc.sweep(**_target(lo=0.5, hi=0.5), measure=_step)


def test_a_step_outside_the_range_is_no_cliff():
    r = cc.sweep(
        **_target(),
        measure=lambda s: _step(s, at=1.5),
    )
    assert r["cliff_found"] is False
    assert "no cliff found" in r["message"]


def test_measure_returning_nothing_refuses():
    with pytest.raises(cc.CurveRefused, match="returned nothing"):
        cc.sweep(**_target(), measure=lambda _s: None)


def test_bind_workunit_refuses_when_the_workunit_is_absent(monkeypatch):
    def boom(name, *a, **k):
        raise ImportError(name)

    monkeypatch.setattr(cc.importlib, "import_module", boom)
    with pytest.raises(cc.CurveRefused, match="inject measure"):
        cc.bind_workunit()


def test_build_writes_a_receipt_that_parses():
    rc = cc.main(["--build"])
    assert rc == 0
    path = RECEIPTS / cc.RECEIPT_NAME
    doc = json.loads(path.read_text())
    assert doc["schema"] == cc.SCHEMA
    assert doc["version"] == cc.VERSION
    assert doc["seal_sha256"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert doc["seal_sha256"] == hashlib.sha256(blob).hexdigest()
    _assert_no_hardware_claims(doc)
    proofs = doc["synthetic_proofs"]
    assert proofs["step"]["cliff_found"] is True
    assert proofs["step"]["contains_step"] is True
    assert proofs["flat"]["cliff_found"] is False
    assert "no cliff found" in proofs["flat"]["message"]
    assert proofs["budget"]["honoured"] is True
    assert doc["live_sweep"]["ran"] is False
    assert doc["live_sweep"]["component"] is None
    assert doc["does_not_choose_science"] is True
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
