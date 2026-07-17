#!/usr/bin/env python3.12
"""Tests for the giant-parent systems layer: synthetic twins, bounded-stream Press, adapters."""
import pathlib
import sys

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import succ_twin as tw  # noqa: E402
import succ_press as pr  # noqa: E402
import succ_adapter_frontier as af  # noqa: E402
import succ_frontier as sf  # noqa: E402


# -- selftests -------------------------------------------------------------------------
def test_twin_selftest():
    assert tw.selftest()["ok"] is True


def test_press_selftest():
    assert pr.selftest()["ok"] is True


def test_adapter_selftest():
    assert af.selftest()["ok"] is True


# -- the acquisition gate: every giant twin must be green (systems path proven) ---------
def test_all_giant_twins_green():
    for p in sf.PARENTS:
        v = tw.validate_twin(p)
        assert v["all_green"] is True, f"{p.row_id} twin not green: {v.get('checks')}"
        # the mandated systems checks are all present and pass
        checks = v["checks"]
        for required in ("deterministic_conversion", "round_trip_integrity", "bounded_rss",
                         "source_range_resume", "expert_paging", "crash_recovery",
                         "duplicate_launch_prevention", "output_layout_validation"):
            assert required in checks, f"{p.row_id} missing check {required}"


# -- bounded-stream Press: peak disk must be a small fraction of the source -------------
def test_press_peak_disk_is_bounded():
    for p in sf.PARENTS:
        plan = pr.press_plan(p)
        assert plan.get("launches_nothing_while_legacy_running") is True
        assert plan.get("disk_walled") is True
        peak = (plan.get("peak_disk_bytes") or plan.get("peak_disk_gb", 0) * 1e9
                or plan.get("peak_disk", 0))
        assert peak > 0
        # peak must be well under the source (bounded stream, not whole-source coexistence)
        assert peak < p.source_bytes * 0.10, f"{p.row_id} peak disk not bounded: {peak}"
        # and it must fit the ~175 GB free disk with room
        assert peak < 60e9, f"{p.row_id} peak disk exceeds a safe fraction of free disk: {peak}"


# -- adapters are fail-closed contracts, never falsely 'ready' --------------------------
def test_adapters_refuse_and_split_claims():
    for mt in ("deepseek_v32", "kimi_k25", "deepseek_v4"):
        caps = af.capabilities(mt)
        assert caps["ready_for_execution"] is False
        assert caps["blockers"]
        refusal = af.run(mt, {"any": "request"})
        assert refusal["status"] == "refused" and refusal["exit"] == 78
    # kimi build_spec carries the text-core / full-multimodal claim split
    kimi = next(p for p in sf.PARENTS if p.row_id == "kimi-k2.6-1t")
    spec = af.build_spec(kimi)
    blob = str(spec)
    assert "TEXT_CORE" in blob and "MULTIMODAL" in blob
    # deepseek-v4 real geometry: 6 selected experts
    v4 = next(p for p in sf.PARENTS if p.row_id == "deepseek-v4-pro-1.6t")
    assert v4.experts_per_tok == 6
