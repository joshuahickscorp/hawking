from __future__ import annotations

from types import SimpleNamespace

from hcli.agentos.native_gate import _stage
from hcli.latency import now_ns


def test_native_gate_stage_records_integer_nanoseconds() -> None:
    config = SimpleNamespace(identity=lambda: {"resident_identity": "fixture"})
    row = _stage("fixture", config=config, started_ns=now_ns() - 1_000_000, passed=True)

    assert row["timing_unit"] == "ns"
    assert isinstance(row["elapsed_ns"], int)
    assert row["elapsed_ns"] >= 1_000_000
    assert isinstance(row["elapsed_s"], float)


def test_native_gate_stage_compatibility_seconds_are_derived() -> None:
    config = SimpleNamespace(identity=lambda: {})
    row = _stage("fixture", config=config, started_ns=now_ns(), passed=False)

    assert row["timing_unit"] == "ns"
    assert row["elapsed_s"] == round(row["elapsed_ns"] / 1_000_000_000.0, 3)
