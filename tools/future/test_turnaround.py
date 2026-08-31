"""Tests for tools/future/turnaround.py.

The negative control is load-bearing: GPU/build phases must refuse a CPU
proxy, and the refusal has to actually raise.
"""
from __future__ import annotations

import json

import pytest

from tools.future import turnaround as ta
from tools.future._common import HARDWARE_FIELDS, RECEIPTS


@pytest.fixture(scope="module")
def measured_path():
    return ta.build(repeats=3)


@pytest.fixture(scope="module")
def doc(measured_path):
    return json.loads(measured_path.read_text())


def test_build_and_selftest_emit_sealed_receipt(measured_path, doc):
    assert measured_path.parent == RECEIPTS
    assert measured_path.name == "EXPERIMENT_TURNAROUND.json"
    assert doc["schema"] == ta.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    again = ta.selftest(repeats=2)
    body = json.loads(again.read_text())
    assert body["schema"] == ta.SCHEMA
    assert body["seal_sha256"]


def test_phases_match_contract_and_scoreboard_schema(doc):
    names = [row["name"] for row in doc["phases"]]
    assert names == list(ta.PHASES)
    assert set(doc["development_phases"]) == set(ta.SCOREBOARD_DEVELOPMENT_PHASES)
    assert doc["experiment_turnaround_ns"] is None
    assert doc["total_experiment_turnaround_ns"] is None
    for key, value in doc["development_phases"].items():
        assert value is None, f"{key} must stay null; CPU timings are not scoreboard ns"


def test_cpu_phases_are_repeated_medians(doc):
    cpu = [
        "source_discovery",
        "transform",
        "verify",
        "receipt",
        "ledger",
        "next_decision",
    ]
    by_name = {row["name"]: row for row in doc["phases"]}
    for name in cpu:
        row = by_name[name]
        assert row["state"] == "MEASURED_CPU_SIDE"
        assert row["n"] >= ta.MIN_REPEATS
        assert len(row["samples_ms"]) == row["n"]
        assert isinstance(row["phase_wall_ms_cpu_side"], (int, float))
        assert row["phase_wall_ms_cpu_side"] == row["median_ms"]
        assert row["min_ms"] <= row["median_ms"] <= row["max_ms"]
        assert row["range_ms"] == round(row["max_ms"] - row["min_ms"], 3)
        assert name in doc["phase_wall_ms_cpu_side"]
        assert doc["phase_wall_ms_cpu_side"][name] == row["median_ms"]


def test_source_discovery_is_a_real_repo_scan(doc):
    inv = doc["source_inventory"]
    assert inv["scan"] == "git ls-tree -r --name-only HEAD"
    assert inv["file_count"] > 0
    assert inv["rust_rs"] > 0
    assert inv["hawking_core_rs"] > 0
    assert inv["python"] > 0
    assert ".rs" in inv["by_suffix"]


def test_dominant_cost_splits_measured_from_hypothesized(doc):
    dominant = doc["dominant_cost"]
    measured = dominant["among_measured_cpu_side"]
    full = dominant["full_experiment_loop"]
    lever = dominant["lever"]
    assert measured["phase"] in {
        "source_discovery",
        "transform",
        "verify",
        "receipt",
        "ledger",
        "next_decision",
    }
    assert isinstance(measured["median_ms"], (int, float))
    assert full["hypothesized_dominant_phase"] == "compile"
    assert full["state"] == "UNKNOWN"
    assert full["not_a_measurement"] is True
    assert "static_evidence" in full and full["static_evidence"]
    assert lever["id"] == "target_isolation_plus_input_fingerprint"
    assert "does_not_weaken_reproducibility" in lever
    assert "PROTECTED_ABSOLUTE" in lever["does_not_weaken_reproducibility"]


def test_receipt_has_recovery_and_gaps(doc):
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    paths = [row["path"] for row in doc["recovered_implementation"]]
    assert "receipts/headless/ACCELERATOR_SCOREBOARD.json" in paths
    assert "tools/accelerator/scoreboard.py" in paths
    assert "crates/hawking-core/src/startup_timing.rs" in paths


def test_no_numeric_hardware_fields(doc):
    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                here = f"{path}.{key}" if path else key
                if key in HARDWARE_FIELDS:
                    assert not isinstance(value, (int, float)), here
                walk(value, here)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(doc, "")


def test_cpu_timer_table_has_no_gpu_proxy():
    assert ta.CPU_TIMERS.keys().isdisjoint(ta.GPU_OR_BUILD_PHASES)
    for phase in ta.GPU_OR_BUILD_PHASES:
        assert phase not in ta.CPU_TIMERS


def test_refuses_to_record_gpu_dependent_phase_as_measured():
    """Negative control: the refusal must fire, not merely exist."""
    for phase in sorted(ta.GPU_OR_BUILD_PHASES):
        with pytest.raises(ta.GpuDependentPhaseError, match=phase):
            ta.record_cpu_side(phase, 12.3)
        with pytest.raises(ta.GpuDependentPhaseError, match=phase):
            ta.record_cpu_side(phase, 0)


def test_refuses_to_launder_cpu_ms_into_scoreboard_ns():
    for name in (
        "experiment_turnaround_ns",
        "total_experiment_turnaround_ns",
        "compile_ns",
        "transform_ns",
        "receipt_ns",
    ):
        with pytest.raises(ta.HardwareLaunderingError, match=name):
            ta.record_scoreboard_ns(name, 1_000_000)


def test_gpu_phases_come_back_unknown_with_null_cpu_fields(doc):
    by_name = {row["name"]: row for row in doc["phases"]}
    for phase in ta.GPU_OR_BUILD_PHASES:
        row = by_name[phase]
        assert row["state"] == "UNKNOWN"
        assert row["phase_wall_ms_cpu_side"] is None
        assert row["median_ms"] is None
        assert row["samples_ms"] is None
        assert row["n"] == 0
        assert row["reason"]
        assert phase not in doc["phase_wall_ms_cpu_side"]


def test_measure_refuses_a_single_sample():
    with pytest.raises(ValueError, match="single sample"):
        ta.measure(repeats=1)


def test_summarize_median_and_spread():
    summary = ta._summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["n"] == 5
    assert summary["median_ms"] == 3.0
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 5.0
    assert summary["range_ms"] == 4.0
    assert "iqr_ms" in summary
