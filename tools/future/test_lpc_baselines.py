"""Tests for LPC baselines: neighbour, rule model, splits, abstention, authority."""
from __future__ import annotations

import json

import pytest

from tools.future import lpc_baselines as lb
from tools.future import lpc_dataset as ds
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def test_build_emits_sealed_receipt():
    out = lb.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "LPC_BASELINES.json"
    assert doc["schema"] == lb.SCHEMA
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["selftest"] == "passed"
    _assert_no_hardware_claims(doc)


def test_distance_symmetric_and_zero_to_self():
    a = lb._complete_fixture(row_id="a", organ_fingerprint="mlp")
    b = lb._complete_fixture(row_id="b", organ_fingerprint="attention")
    aa = lb.row_distance(a, a)
    ab = lb.row_distance(a, b)
    ba = lb.row_distance(b, a)
    assert aa is not None and aa < 0.05
    assert ab is not None and ba is not None
    assert abs(ab - ba) < 1e-12
    assert ab > aa


def test_nearest_neighbour_picks_same_organ():
    mlp = lb._complete_fixture(row_id="mlp", organ_fingerprint="mlp", latency=10.0)
    attn = lb._complete_fixture(row_id="attn", organ_fingerprint="attention", latency=50.0)
    query = lb._complete_fixture(
        row_id="q", organ_fingerprint="mlp", latency=None,
        absence_reasons={"latency": "UNMEASURED"},
    )
    neighbour = lb.nearest_measured_neighbour(query, [mlp, attn])
    assert neighbour is not None
    assert neighbour["row_id"] == "mlp"
    assert neighbour["label"] == 10.0


def test_null_numeric_is_not_distance_zero():
    measured = lb._complete_fixture(row_id="m", dispatches=100)
    missing = lb._complete_fixture(
        row_id="n",
        dispatches=None,
        absence_reasons={"dispatches": "UNMEASURED"},
    )
    # Comparable on every other axis; dispatches is skipped, not treated as 0.
    d = lb.row_distance(measured, missing)
    assert d is not None
    zeroed = lb._complete_fixture(row_id="z", dispatches=0)
    d0 = lb.row_distance(measured, zeroed)
    assert d0 is not None
    assert d0 > d


def test_rule_cost_abstains_on_null_active_bytes():
    """NEGATIVE CONTROL: a null input must not be treated as 0."""
    full = lb._complete_fixture(active_bytes=10, dispatches=2, synchronization=3)
    predicted = lb.rule_cost(full)
    assert predicted.status == lb.PREDICTED
    assert predicted.value == 10 * 1.0 + 2 * 1000.0 + 3 * 100.0

    null_active = lb._complete_fixture(
        active_bytes=None,
        dispatches=2,
        synchronization=3,
        absence_reasons={"active_bytes": "UNMEASURED"},
    )
    refused = lb.rule_cost(null_active)
    assert refused.status == lb.ABSTAIN
    assert refused.value is None
    assert "active_bytes" in (refused.reason or "")
    # The wrong implementation (impute 0) would equal w_d*2 + w_s*3.
    imputed_zero = 0.0 + 2 * 1000.0 + 3 * 100.0
    assert refused.value != imputed_zero


def test_rule_cost_monotone_in_dispatches():
    low = lb.rule_cost(lb._complete_fixture(dispatches=1))
    high = lb.rule_cost(lb._complete_fixture(dispatches=10))
    assert low.status == lb.PREDICTED and high.status == lb.PREDICTED
    assert high.value > low.value


def test_held_out_splits_by_model_organ_device_do_not_leak():
    rows = [
        lb._complete_fixture(row_id="a", model="qwen27", organ_fingerprint="mlp",
                             machine_genome="m3"),
        lb._complete_fixture(row_id="b", model="qwen27", organ_fingerprint="attention",
                             machine_genome="m3"),
        lb._complete_fixture(row_id="c", model="flash", organ_fingerprint="mlp",
                             machine_genome="m2"),
        lb._complete_fixture(row_id="d", model="flash", organ_fingerprint="attention",
                             machine_genome="m2"),
        # Null organ must not form a group and must not leak into train/holdout ids.
        lb._complete_fixture(
            row_id="null-organ",
            model="flash",
            organ_fingerprint=None,
            machine_genome="m2",
            absence_reasons={"organ_fingerprint": "NOT_IN_SOURCE"},
        ),
    ]
    for axis in ("architecture", "organ", "device"):
        splits = lb.held_out_splits(rows, axis=axis)
        assert splits, axis
        seen_hold = set()
        for split in splits:
            assert lb.split_has_no_leak(split), split
            assert split.holdout_key not in seen_hold
            seen_hold.add(split.holdout_key)
            assert split.holdout_ids
            if axis == "organ":
                assert "null-organ" not in split.holdout_ids
                assert "null-organ" not in split.train_ids
        if axis == "architecture":
            assert {s.holdout_key for s in splits} == {"qwen27", "flash"}
        if axis == "device":
            assert {s.holdout_key for s in splits} == {"m3", "m2"}


def test_architecture_holdout_abstains_on_unseen_model():
    train = [
        lb._complete_fixture(row_id="q0", model="qwen27", latency=10.0),
        lb._complete_fixture(row_id="q1", model="qwen27", organ_fingerprint="attention",
                             latency=11.0),
    ]
    holdout = lb._complete_fixture(
        row_id="flash",
        model="flash",
        organ_fingerprint="router",
        representation="source_bf16_exact",
        machine_genome="other",
        physical_graph_identity="flash-graph",
        backend="ane",
        layout="blocked",
        tile="tg32",
        grouping="moe",
        fusion="router-topk",
        persistent_resources="none",
        latency=None,
        absence_reasons={"latency": "UNMEASURED"},
    )
    refused = lb.predict(holdout, train, method="nearest")
    assert refused.status == lb.ABSTAIN
    assert refused.value is None


def test_diagnostic_neighbour_does_not_support_protected_query():
    diagnostic = lb._complete_fixture(
        row_id="diag",
        contamination_class="DIAGNOSTIC_RELATIVE",
        latency=3.0,
    )
    query = lb._complete_fixture(
        row_id="q",
        contamination_class="PROTECTED_ABSOLUTE",
        latency=None,
        absence_reasons={"latency": "UNMEASURED"},
    )
    neighbour = lb.nearest_measured_neighbour(query, [diagnostic])
    assert neighbour is None
    refused = lb.predict(query, [diagnostic], method="nearest")
    assert refused.status == lb.ABSTAIN
    assert refused.reason == "no_measured_neighbour"


def test_authority_rejects_static_and_ignores_model_even_when_equal():
    model = lb.Prediction(status=lb.PREDICTED, value=5.0, uncertainty=0.1)
    protected = {"contamination_class": "PROTECTED_ABSOLUTE", "value": 5.0}
    decided = lb.resolve_authority(model, protected)
    assert decided["model_prediction_ignored"] is True
    assert decided["source"] == "PROTECTED_ABSOLUTE"
    with pytest.raises(lb.AuthorityError):
        lb.resolve_authority(model, {"contamination_class": "STATIC_ONLY", "value": 5.0})


def test_describe_exports_the_metric_and_radius():
    info = lb.describe()
    assert info["nearest_measured_neighbour"]["support_radius"] == lb.SUPPORT_RADIUS
    assert info["rule_cost_model"]["null_input"] == lb.ABSTAIN
    assert "architecture" in info["held_out_splits"]


def test_complete_fixture_is_a_valid_lpc_row():
    row = lb._complete_fixture()
    assert set(ds.REQUIRED_FIELDS).issubset(row)
    assert ds.validate_row(row)["complete"] is True
