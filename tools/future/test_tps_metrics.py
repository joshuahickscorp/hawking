"""71 is single-stream latency, and nothing else may claim progress toward it.

S025 §45-46. The failure mode is specific: a speculative or parallel decoding
scheme raises accepted throughput without touching single-token latency, and the
concurrency ladder raises aggregate throughput while each session gets SLOWER.
Either could be reported as progress toward 71 by anyone not paying attention.
"""
from __future__ import annotations

import pytest

from tools.future import tps_metrics as m


def test_71_is_the_latency_metric():
    assert m.which_metric_is_71() == m.LATENCY
    assert m.TARGET_MS == pytest.approx(14.085, abs=1e-3)


def test_accepted_token_tps_may_not_claim_progress_toward_71():
    with pytest.raises(m.MetricConfusion, match="cannot be progress toward 71"):
        m.assert_not_confused(m.ACCEPTED, claimed_progress_toward_71=True)


def test_aggregate_multistream_may_not_claim_progress_toward_71():
    """It rises while the 71 metric falls. That is the whole point."""
    with pytest.raises(m.MetricConfusion, match="gets WORSE"):
        m.assert_not_confused(m.AGGREGATE, claimed_progress_toward_71=True)


def test_the_latency_metric_may():
    m.assert_not_confused(m.LATENCY, claimed_progress_toward_71=True)


def test_reporting_a_metric_without_claiming_71_is_always_fine():
    for metric in (m.LATENCY, m.ACCEPTED, m.AGGREGATE):
        m.assert_not_confused(metric, claimed_progress_toward_71=False)


def test_an_unknown_metric_is_refused():
    with pytest.raises(m.MetricConfusion, match="unknown metric"):
        m.assert_not_confused("TOKENS_PER_VIBE", claimed_progress_toward_71=False)


def test_the_current_latency_reads_the_measured_body():
    cur = m.current()[m.LATENCY]
    assert cur["tps"] == pytest.approx(36.644, abs=1e-2)
    assert cur["is_the_71_target"] is True
    assert cur["gap_ms"] == pytest.approx(27.2896 - m.TARGET_MS, abs=1e-3)


def test_accepted_is_null_not_zero():
    """Nothing has run. Zero would be a measurement."""
    acc = m.current()[m.ACCEPTED]
    assert acc["tps"] is None
    assert acc["status"] == "NOT_MEASURED"
    assert "never as progress toward 71" in acc["may_never_claim"]


def test_aggregate_carries_the_reason_it_is_not_the_target():
    agg = m.current()[m.AGGREGATE]
    if agg is None:
        pytest.skip("no concurrency receipt on disk")
    assert agg["is_the_71_target"] is False
    assert "SLOWER" in agg["why_not"]
