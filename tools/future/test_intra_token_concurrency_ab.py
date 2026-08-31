"""Wrapping independent pairs in a Metal concurrent group changes nothing.

S025 §7-8 asks for a matched SERIAL vs CONCURRENT probe at exact parity, and
treats multiple encoders and queues as experiments rather than dogma. The knob
already existed - HAWKING_QWEN38_CONCURRENT - so the probe is an environment A/B
on the production path.

The result confirms SINGLE_TOKEN_PARALLEL_SLACK from the other direction: the
independent operations are already fused, so a concurrent group has nothing to
wrap.
"""
from __future__ import annotations

import pytest

from tools.future import intra_token_concurrency_ab as ab


def test_the_arms_are_token_identical():
    """A timing difference between different outputs is not a speedup."""
    got = ab.compare()
    assert got["token_identical"] is True
    assert got["decode_steps"] == 48


def test_the_verdict_is_no_measurable_effect():
    got = ab.compare()
    assert got["verdict"] == "NO_MEASURABLE_EFFECT"
    assert abs(got["delta_ms"]) < got["run_to_run_bar_ms"]


def test_a_delta_inside_the_run_to_run_spread_is_not_a_result():
    """0.0465 ms would be a 0.17% 'win' if read carelessly."""
    got = ab.compare()
    assert 0 < abs(got["delta_ms"]) < 0.15
    assert "reading noise" in got["why"]


def test_the_gpu_medians_agree_to_microseconds():
    """The strongest form of the negative: the GPU did the same work."""
    got = ab.compare()
    assert abs(got["gpu_median_delta_us"]) < 5.0


def test_it_confirms_the_dag_prediction_rather_than_restating_it():
    r = ab.reading()
    assert "SINGLE_TOKEN_PARALLEL_SLACK" in r["predicted_by"]
    assert "ALREADY" in r["prediction"] and "FUSED" in r["prediction"]
    assert r["outcome"].startswith("confirmed")


def test_it_does_not_overclaim_what_it_killed():
    """Occupancy, memory-level parallelism, the ALU chain and multiple QUEUES
    all survive this probe."""
    survives = " ".join(ab.reading()["what_it_does_not_kill"])
    for word in ("occupancy", "memory-level parallelism", "dependency chain",
                 "command QUEUES"):
        assert word in survives


def test_the_contaminated_window_is_declared_and_the_ratio_is_the_result():
    r = ab.reading()
    assert "contaminated window" in r["evidence_class_note"]
    assert "not offered as a protected absolute" in r["evidence_class_note"]
    assert ab.build()["evidence_class"] == "DIAGNOSTIC_RELATIVE"


def test_missing_arms_refuse_rather_than_assembling(monkeypatch, tmp_path):
    monkeypatch.setattr(ab, "SERIAL", tmp_path / "nope.json")
    with pytest.raises(ab.ProbeRefused, match="not on disk"):
        ab.compare()
