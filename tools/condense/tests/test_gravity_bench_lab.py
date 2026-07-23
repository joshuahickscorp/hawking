"""Tests for the matched-benchmark harness - enforce the matched benchmark law.

Each test corresponds to a way the retracted 35.9x claim could be produced again:
comparing unmatched specs, reporting a bare mean over a contended sample, crediting an
unmeasured component as zero, or quoting a refuted number.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gravity_bench_lab as bl  # noqa: E402


def _spec(**over):
    base = dict(rows=64, cols=128, batch=1, input_seed=7, input_dtype="float32",
                output_dtype="float32", warmup=1, reps=5,
                sync_boundary="none_cpu_wall_clock", dependency_shape="independent_calls",
                pack_in_timed_region=False, unpack_in_timed_region=True)
    base.update(over)
    return bl.BenchSpec(**base)


def _result(baseline, samples, spec=None, **over):
    return bl.BenchResult(
        baseline=baseline,
        spec=spec or _spec(),
        timings=bl.ComponentTimings(end_to_end=bl.TimingStats(tuple(samples))),
        **over,
    )


def test_matched_specs_are_field_identical():
    assert bl.matched(_spec(), _spec())
    assert bl.mismatched_fields(_spec(), _spec()) == ()
    bl.require_matched(_spec(), _spec())


@pytest.mark.parametrize("field,value", [
    ("rows", 65), ("cols", 129), ("batch", 2), ("input_seed", 8),
    ("input_dtype", "float16"), ("output_dtype", "float16"),
    ("warmup", 2), ("reps", 6),
    ("sync_boundary", "per_call_host_sync"),
    ("dependency_shape", "serial_dependent_chain"),
    ("pack_in_timed_region", True), ("unpack_in_timed_region", False),
])
def test_any_differing_field_breaks_the_match(field, value):
    """Every field participates; none of them is a 'detail' that may drift."""
    other = _spec(**{field: value})
    assert not bl.matched(_spec(), other)
    assert bl.mismatched_fields(_spec(), other) == (field,)
    with pytest.raises(bl.MatchedBenchmarkError):
        bl.require_matched(_spec(), other)
    assert _spec().fingerprint != other.fingerprint


def test_speedup_refuses_unmatched_specs():
    base = _result("cpu_authority", [1.0, 1.0, 1.0, 1.0, 1.0])
    cand = _result("custom_v2", [0.5, 0.5, 0.5, 0.5, 0.5], spec=_spec(batch=2))
    with pytest.raises(bl.MatchedBenchmarkError, match="unmatched BenchSpecs"):
        bl.speedup(base, cand)


def test_speedup_refuses_unreproduced_baseline():
    base = _result("dense_fp16_mps", [1.0, 1.0, 1.0, 1.0, 1.0], reproduced=False)
    cand = _result("custom_v2", [0.5, 0.5, 0.5, 0.5, 0.5])
    with pytest.raises(bl.MatchedBenchmarkError, match="unreproduced"):
        bl.speedup(base, cand)


def test_speedup_refuses_mismatched_timed_region():
    """GPU-only time on one side and wall clock on the other is not a matched comparison."""
    base = bl.BenchResult("cpu_authority", _spec(),
                          bl.ComponentTimings(end_to_end=bl.TimingStats((1.0, 1.0, 1.1))))
    cand = bl.BenchResult("custom_v2", _spec(),
                          bl.ComponentTimings(gpu_execution=bl.TimingStats((0.5, 0.5, 0.5))))
    with pytest.raises(bl.MatchedBenchmarkError, match="component mismatch"):
        bl.speedup(base, cand)


def test_speedup_on_matched_specs_reports_the_ratio_and_direction():
    base = _result("dense_fp16_mps", [0.3674] * 5)
    cand = _result("custom_v2", [0.5057] * 5)
    out = bl.speedup(base, cand)
    assert out["specs_matched"] and out["baseline"] == "dense_fp16_mps"
    assert out["candidate"] == "custom_v2"
    assert math.isclose(out["speedup"], 0.3674 / 0.5057, rel_tol=1e-9)
    assert out["slower_than_baseline"]           # the corrected truth, not 35.9x


def test_unknown_baseline_name_is_rejected():
    with pytest.raises(bl.MatchedBenchmarkError, match="unknown baseline"):
        _result("my_fast_kernel", [1.0, 1.0])


def test_raw_sample_statistics_are_correct_and_no_mean_is_exposed():
    samples = (1.0, 2.0, 3.0, 4.0, 100.0)
    st = bl.TimingStats(samples)
    assert st.count == 5
    assert st.min_ms == 1.0 and st.max_ms == 100.0
    assert st.median_ms == 3.0
    assert st.p95_ms == 100.0                     # nearest rank: ceil(0.95*5)=5 -> last
    # sample stddev (n-1): mean 22, sum sq dev 7610, 7610/4 = 1902.5, sqrt = 43.617657
    assert math.isclose(st.stddev_ms, math.sqrt(1902.5), rel_tol=1e-12)
    assert math.isclose(st.coefficient_of_variation, math.sqrt(1902.5) / 3.0, rel_tol=1e-12)
    assert list(st.to_json()["raw_samples_ms"]) == list(samples)
    assert not any("mean" in k for k in st.to_json())
    assert not hasattr(st, "mean_ms")


def test_p95_nearest_rank_on_twenty_samples():
    st = bl.TimingStats(tuple(float(i) for i in range(1, 21)))
    assert st.median_ms == 10.5
    assert st.p95_ms == 19.0                      # ceil(0.95*20)=19 -> 19th smallest


def test_timing_stats_reject_degenerate_input():
    with pytest.raises(bl.MatchedBenchmarkError):
        bl.TimingStats((1.0,))
    with pytest.raises(bl.MatchedBenchmarkError):
        bl.TimingStats((1.0, float("nan")))
    with pytest.raises(bl.MatchedBenchmarkError):
        bl.TimingStats((1.0, -1.0))


def test_contention_flag_tracks_the_documented_threshold():
    quiet = bl.TimingStats((1.00, 1.01, 0.99, 1.00, 1.02))
    assert quiet.coefficient_of_variation < bl.CONTENTION_CV_THRESHOLD
    assert not quiet.is_contended

    noisy = bl.TimingStats((1.0, 1.0, 1.0, 1.0, 5.0))
    assert noisy.coefficient_of_variation > bl.CONTENTION_CV_THRESHOLD
    assert noisy.is_contended
    assert noisy.median_ms == 1.0 and noisy.max_ms == 5.0     # tail is load, not hardware
    assert noisy.to_json()["contention_cv_threshold"] == 0.15


def test_unmeasured_components_serialise_as_unmeasured_not_zero():
    res = _result("cpu_authority", [1.0, 1.0, 1.0])
    timings = res.to_json()["timings"]
    assert set(timings) == set(bl.COMPONENTS)
    assert timings["end_to_end"]["median_ms"] == 1.0
    for name in ("gpu_execution", "host_encode", "command_buffer", "cold_start",
                 "warm_steady_state"):
        assert timings[name] == "UNMEASURED"
        assert timings[name] != 0


def test_at_least_one_component_must_be_measured():
    with pytest.raises(bl.MatchedBenchmarkError):
        bl.ComponentTimings()


def test_roofline_bills_against_the_measured_roofs():
    res = _result("cpu_authority", [1.0, 1.0, 1.0])
    res = replace(res, bytes_moved=736_000_000, flops=17_703_000_000)
    roof = res.roofline()
    assert roof["timing_source"] == "end_to_end"
    # 1.0 ms for exactly 1/1000 of each roof's per-second figure => fraction 1.0
    assert math.isclose(roof["fraction_of_bandwidth_roof"], 1.0, rel_tol=1e-9)
    assert math.isclose(roof["fraction_of_compute_roof"], 1.0, rel_tol=1e-9)
    assert bl.BANDWIDTH_ROOF_GB_S == 736.0 and bl.COMPUTE_ROOF_GFLOP_S == 17703.0


def test_roofline_without_counters_is_unmeasured():
    roof = _result("cpu_authority", [1.0, 1.0, 1.0]).roofline()
    assert roof["achieved_gb_s"] == "UNMEASURED"
    assert roof["fraction_of_compute_roof"] == "UNMEASURED"


@pytest.mark.parametrize("name", [c.name for c in bl.REFUTED_CLAIMS])
def test_refuted_claims_are_rejected_by_name(name):
    with pytest.raises(bl.MatchedBenchmarkError, match="REFUTED"):
        bl.assert_not_refuted(name=name)


@pytest.mark.parametrize("kind,value", [
    ("milliseconds", 9.012), ("ratio", 35.9), ("parity", 1.4e-6),
])
def test_refuted_claims_are_rejected_by_value(kind, value):
    with pytest.raises(bl.MatchedBenchmarkError, match="REFUTED"):
        bl.assert_not_refuted(kind=kind, value=value)


def test_the_retracted_headline_cannot_be_rebuilt():
    """9.012 ms / 0.2511 ms = 35.9x. Both the baseline and the ratio are rejected."""
    base = _result("dense_fp16_mps", [9.012] * 5)
    cand = _result("custom_v2", [9.012 / 35.9] * 5)
    with pytest.raises(bl.MatchedBenchmarkError, match="dense_fp16_9.012ms"):
        bl.speedup(base, cand)
    # and even reaching 35.9x from other numbers is rejected on the ratio
    base2 = _result("dense_fp16_mps", [3.59] * 5)
    cand2 = _result("custom_v2", [0.1] * 5)
    with pytest.raises(bl.MatchedBenchmarkError, match="speedup_35.9x"):
        bl.speedup(base2, cand2)


def test_honest_numbers_pass_the_refuted_guard():
    bl.assert_not_refuted(kind="milliseconds", value=0.3674)
    bl.assert_not_refuted(kind="ratio", value=0.727)
    bl.assert_not_refuted(kind="parity", value=2.1e-4)


def test_spec_and_result_json_round_trip():
    spec = _spec()
    assert bl.BenchSpec.from_json(json.loads(json.dumps(spec.to_json()))) == spec

    res = replace(_result("custom_v2", [1.0, 2.0, 3.0]), bytes_moved=1024, flops=2048,
                  notes="n")
    back = bl.BenchResult.from_json(json.loads(json.dumps(res.to_json())))
    assert back == res
    assert back.to_json() == res.to_json()


def test_report_schema_round_trip():
    base = _result("cpu_authority", [1.0, 1.0, 1.0])
    cand = _result("custom_v2", [0.5, 0.5, 0.5])
    report = bl.build_report([base, cand], [bl.speedup(base, cand)], label="t")
    loaded = json.loads(json.dumps(report))
    assert loaded["schema"] == "hawking.glm52.matched_benchmark.v1"
    assert loaded["baselines"] == list(bl.BASELINES)
    assert loaded["machine"]["bandwidth_roof_gb_s"] == 736.0
    assert loaded["machine"]["gpu_cores"] == 60
    assert [c["name"] for c in loaded["refuted_claims"]] == [c.name for c in bl.REFUTED_CLAIMS]
    assert loaded["matched"][0]["baseline"] == "cpu_authority"
    assert loaded["matched"][0]["specs_matched"] is True
    assert [bl.BenchResult.from_json(r).to_json() for r in loaded["results"]] == loaded["results"]


def test_report_writer_refuses_a_foreign_schema(tmp_path):
    with pytest.raises(bl.MatchedBenchmarkError):
        bl.write_report(tmp_path / "x.json", {"schema": "something.else.v1"})


def test_build_report_refuses_an_unasserted_speedup():
    with pytest.raises(bl.MatchedBenchmarkError, match="specs_matched"):
        bl.build_report([], [{"baseline": "cpu_authority", "candidate": "custom_v2"}], label="t")


def test_measure_keeps_every_sample():
    spec = _spec(warmup=2, reps=7)
    calls = []
    stats = bl.measure(lambda: calls.append(1), spec)
    assert len(calls) == 9                       # warmup runs, but is not timed
    assert stats.count == 7 == len(stats.raw_samples_ms)


def test_selftest_runs_cpu_only_and_is_internally_consistent():
    report = bl.selftest(rows=64, cols=128)
    assert report["schema"] == "hawking.glm52.matched_benchmark.v1"
    assert {r["baseline"] for r in report["results"]} == {"cpu_authority", "custom_v2"}
    assert report["selftest"]["json_round_trip_stable"]
    assert report["selftest"]["unmatched_comparison_refused"]
    assert report["selftest"]["refuted_guard_live"]
    for res in report["results"]:
        assert res["timings"]["gpu_execution"] == "UNMEASURED"      # no GPU work in this phase
        assert res["timings"]["end_to_end"]["count"] == 15
